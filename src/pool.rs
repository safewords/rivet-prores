//! A process-wide pool of worker threads for slice-parallel work.
//!
//! ProRes slices are independent by design (§4: each is decodable on its
//! own), so a picture's slices — or its rows of slices — are handed out as
//! numbered tasks. [`run`] calls a closure for every task number, on the
//! calling thread and on up to `threads − 1` pool workers, and returns once
//! every task has finished. What a task computes never depends on which
//! thread runs it, so the output is the same at any thread count.
//!
//! The workers are started on first use, one fewer than the CPU's
//! parallelism, and shared by every encoder and decoder in the process; a
//! caller whose workers are busy elsewhere still makes progress alone.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

/// The thread count `0` stands for: one per CPU.
pub(crate) fn resolve(threads: usize) -> usize {
    if threads == 0 { parallelism() } else { threads }
}

fn parallelism() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// The task function with its lifetime erased; see the SAFETY argument in
/// [`run`].
type TaskFn = dyn Fn(usize) + Sync;

struct Job {
    f: *const TaskFn,
    tasks: usize,
    /// Tasks claimed at a time.
    chunk: usize,
    /// The next task to claim, on a cache line of its own: every claim
    /// writes it, and every task reads the fields above.
    next: CacheLine<AtomicUsize>,
    max_helpers: usize,
    state: Mutex<JobState>,
    left: Condvar,
}

#[derive(Default)]
struct JobState {
    joined: usize,
    left: usize,
    panic: Option<Box<dyn Any + Send>>,
}

// SAFETY: `f` points to a `Sync` closure that outlives every use of it by
// another thread (see `run`); the rest of `Job` is `Send + Sync` already.
unsafe impl Send for Job {}
// SAFETY: as above.
unsafe impl Sync for Job {}

impl Job {
    /// Claims and runs tasks, `chunk` at a time, until none are left.
    fn work(&self) {
        loop {
            let first = self.next.fetch_add(self.chunk, Ordering::Relaxed);
            if first >= self.tasks {
                return;
            }
            for i in first..(first + self.chunk).min(self.tasks) {
                // SAFETY: a task number below `tasks` is only claimed while
                // the caller of `run` is still inside it, so `f` is alive.
                unsafe { (*self.f)(i) };
            }
        }
    }

    fn wants_help(&self, state: &JobState) -> bool {
        state.joined < self.max_helpers && self.next.load(Ordering::Relaxed) < self.tasks
    }
}

#[repr(align(128))]
struct CacheLine<T>(T);

impl<T> std::ops::Deref for CacheLine<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

struct Pool {
    queue: Mutex<Vec<Arc<Job>>>,
    work: Condvar,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panicking task is caught before it can poison anything; recover
    // the guard regardless.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool { queue: Mutex::new(Vec::new()), work: Condvar::new() }));
        for i in 0..parallelism().saturating_sub(1) {
            let spawned = std::thread::Builder::new().name(format!("prores-{i}")).spawn(move || worker(pool));
            if spawned.is_err() {
                break; // fewer workers; `run` still completes on its caller
            }
        }
        pool
    })
}

fn worker(pool: &'static Pool) {
    loop {
        let job = {
            let mut queue = lock(&pool.queue);
            loop {
                let found = queue.iter().find_map(|job| {
                    let mut st = lock(&job.state);
                    job.wants_help(&st).then(|| {
                        st.joined += 1;
                        job.clone()
                    })
                });
                if let Some(job) = found {
                    break job;
                }
                queue = pool.work.wait(queue).unwrap_or_else(|e| e.into_inner());
            }
        };
        let result = catch_unwind(AssertUnwindSafe(|| job.work()));
        let mut st = lock(&job.state);
        if let Err(p) = result {
            job.next.store(job.tasks, Ordering::Relaxed);
            st.panic.get_or_insert(p);
        }
        st.left += 1;
        job.left.notify_all();
    }
}

/// How tasks are claimed. One at a time shares out a few long tasks (a
/// decoder's rows of slices) most evenly; many short ones (an encoder's
/// slices, a couple of microseconds each at one quantiser) are claimed a few
/// at a time, as one counter claimed by every thread for every task costs
/// more than the evener share gains.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TaskSize {
    Large,
    Small,
}

/// Calls `f(i)` for every `i` in `0..tasks`, each exactly once, on the
/// calling thread and up to `threads − 1` workers (`threads` 0: one per
/// CPU). Returns when all have returned; a panic in any task is resumed
/// here.
pub(crate) fn run(threads: usize, tasks: usize, size: TaskSize, f: &(dyn Fn(usize) + Sync)) {
    let helpers = resolve(threads).min(tasks).saturating_sub(1);
    if helpers == 0 {
        (0..tasks).for_each(f);
        return;
    }
    let pool = pool();
    // SAFETY (lifetime erasure): workers dereference `f` only inside
    // `Job::work`, between joining and leaving the job. A worker joins only
    // while the job is in the queue, under the queue lock; below, the job
    // leaves the queue (under that lock) and this function then waits —
    // even when its own share of the tasks panicked — until every worker
    // that joined has left. So no use of `f` outlives this call.
    let f: *const TaskFn = unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const TaskFn>(f) };
    let chunk = match size {
        TaskSize::Large => 1,
        // At least four claims per thread, at most 8 tasks each.
        TaskSize::Small => (tasks / (4 * (helpers + 1))).clamp(1, 8),
    };
    let job = Arc::new(Job {
        f,
        tasks,
        chunk,
        next: CacheLine(AtomicUsize::new(0)),
        max_helpers: helpers,
        state: Mutex::new(JobState::default()),
        left: Condvar::new(),
    });
    lock(&pool.queue).push(job.clone());
    pool.work.notify_all();
    let mine = catch_unwind(AssertUnwindSafe(|| job.work()));
    lock(&pool.queue).retain(|j| !Arc::ptr_eq(j, &job));
    let mut st = lock(&job.state);
    while st.left < st.joined {
        st = job.left.wait(st).unwrap_or_else(|e| e.into_inner());
    }
    let panic = st.panic.take();
    drop(st);
    if let Err(p) = mine {
        resume_unwind(p);
    }
    if let Some(p) = panic {
        resume_unwind(p);
    }
}

/// [`run`] for tasks that produce a value: the values in task order.
/// Faults in the pages of a new buffer (one write per 4 KiB) on at most
/// four threads. Many threads faulting pages of one allocation at once
/// contend in the kernel: on a 16-core Windows machine, 8 MB took 0.6 ms on
/// one thread, 0.28 ms on four and 1 ms on 32 — and a decoder's workers,
/// writing a new picture, would all be doing it.
pub(crate) fn prefault(threads: usize, buffers: &mut [&mut [u16]]) {
    const PAGE: usize = 2048;
    let total: usize = buffers.iter().map(|b| b.len()).sum();
    let threads = resolve(threads).min(4).min(total.div_ceil(64 * PAGE)).max(1);
    let per = total.div_ceil(threads).next_multiple_of(PAGE);
    let mut parts: Vec<Mutex<Option<&mut [u16]>>> = Vec::new();
    for b in buffers.iter_mut() {
        for part in b.chunks_mut(per) {
            parts.push(Mutex::new(Some(part)));
        }
    }
    run(threads, parts.len(), TaskSize::Large, &|i| {
        if let Some(part) = lock(&parts[i]).take() {
            for page in part.chunks_mut(PAGE) {
                // Opaque: the compiler knows new memory is zero and would
                // drop a plain store of 0.
                page[0] = std::hint::black_box(0);
            }
        }
    });
}

pub(crate) fn map<T: Send>(threads: usize, tasks: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    map_sized(threads, tasks, TaskSize::Large, f)
}

/// [`map`] for many short tasks ([`TaskSize::Small`]).
pub(crate) fn map_small<T: Send>(threads: usize, tasks: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    map_sized(threads, tasks, TaskSize::Small, f)
}

fn map_sized<T: Send>(threads: usize, tasks: usize, size: TaskSize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let slots: Vec<Mutex<Option<T>>> = (0..tasks).map(|_| Mutex::new(None)).collect();
    run(threads, tasks, size, &|i| *lock(&slots[i]) = Some(f(i)));
    slots.into_iter().map(|s| s.into_inner().unwrap_or_else(|e| e.into_inner()).expect("every task ran")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once_at_any_thread_count() {
        for threads in [0, 1, 2, 3, 8, 64] {
            for (tasks, size) in [0, 1, 2, 7, 100, 1000].into_iter().flat_map(|t| [(t, TaskSize::Large), (t, TaskSize::Small)]) {
                let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                run(threads, tasks, size, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
                assert_eq!(map_sized(threads, tasks, size, |i| i * 3), (0..tasks).map(|i| i * 3).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn concurrent_callers_share_the_pool() {
        std::thread::scope(|s| {
            for t in 0..6 {
                s.spawn(move || {
                    for _ in 0..50 {
                        let v = map(0, 64, |i| i + t);
                        assert_eq!(v[63], 63 + t);
                    }
                });
            }
        });
    }

    #[test]
    fn a_panicking_task_panics_the_caller_and_the_pool_survives() {
        for threads in [1, 4] {
            let r = catch_unwind(|| run(threads, 100, TaskSize::Small, &|i| assert_ne!(i, 57)));
            assert!(r.is_err());
        }
        assert_eq!(map(4, 10, |i| i).len(), 10);
    }
}
