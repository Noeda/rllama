// There is no semaphore in Rust standard library. wat??
// So I've made a simple one I can use out of a mutex and condition variable..

use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Clone)]
pub struct Semaphore {
    count: Arc<Mutex<usize>>,
    waiters: Arc<Condvar>,
}

pub struct SemaphoreGuard<'a> {
    mutex_guard: MutexGuard<'a, usize>,
}

impl<'a> Drop for SemaphoreGuard<'a> {
    fn drop(&mut self) {
        *self.mutex_guard += 1;
    }
}

impl Semaphore {
    pub fn new(count: usize) -> Semaphore {
        Semaphore {
            count: Arc::new(Mutex::new(count)),
            waiters: Arc::new(Condvar::new()),
        }
    }

    pub fn acquire(&self) -> SemaphoreGuard {
        let mut count = self.count.lock().unwrap();
        while *count == 0 {
            count = self.waiters.wait(count).unwrap();
        }
        *count -= 1;
        SemaphoreGuard { mutex_guard: count }
    }
}
