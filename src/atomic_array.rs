use std::mem::MaybeUninit;
use std::sync::atomic::Ordering;

use crossbeam_epoch::{Atomic, Guard, Owned};

pub(crate) struct AtomicArray<T> {
    inner: Atomic<[MaybeUninit<T>]>,
}

impl<T> AtomicArray<T> {
    pub unsafe fn load(&self, ordering: Ordering, guard: &Guard) -> &[T] {
        let arr: &[MaybeUninit<T>] = self.inner.load(ordering, guard).deref();
        std::mem::transmute(arr)
    }
}

pub(crate) struct AtomicArrayBuilder<T> {
    inner: Owned<[MaybeUninit<T>]>,
}

impl<T> AtomicArrayBuilder<T> {
    pub fn from_fn(val: impl Fn() -> T, len: usize) -> Self {
        let mut inner: Owned<[MaybeUninit<T>]> = Owned::init(len);

        for i in 0..len {
            inner[i] = MaybeUninit::new(val());
        }

        Self { inner }
    }

    pub fn as_ref(&self) -> &[T] {
        let arr: &[MaybeUninit<T>] = self.inner.as_ref();
        unsafe { std::mem::transmute(arr) }
    }

    pub fn build(self) -> AtomicArray<T> {
        AtomicArray {
            inner: Atomic::from(self.inner),
        }
    }
}
