use std::sync::atomic::Ordering;

use flize::Shield;

pub struct Atomic<T> {
    inner: flize::Atomic<T, flize::NullTag, flize::NullTag, 0, 0>,
}

impl<T> Atomic<T> {
    pub fn new(val: T) -> Self {
        unsafe {
            Atomic {
                inner: flize::Atomic::new(flize::Shared::from_ptr(Box::into_raw(Box::new(val)))),
            }
        }
    }

    pub fn null() -> Self {
        Self {
            inner: flize::Atomic::null(),
        }
    }

    pub fn store(&self, val: Shared<'_, T>, ordering: Ordering) {
        self.inner.store(val.inner, ordering)
    }

    pub fn load<'g>(&self, ordering: Ordering, guard: &'g Guard<'_>) -> Shared<'g, T> {
        Shared {
            inner: self.inner.load(ordering, &guard.inner),
        }
    }

    pub unsafe fn load_unprotected<'g>(&self, ordering: Ordering) -> Shared<'g, T> {
        Shared {
            inner: self.inner.load(ordering, unsafe { flize::unprotected() }),
        }
    }

    pub fn copy(&self, ordering: Ordering) -> Atomic<T> {
        Atomic {
            inner: flize::Atomic::new(self.inner.load(ordering, unsafe { flize::unprotected() })),
        }
    }

    pub fn from_shared(shared: Shared<'_, T>) -> Self {
        Atomic {
            inner: flize::Atomic::new(shared.inner),
        }
    }
}

pub struct Owned<T> {
    val: Box<T>,
}

impl<T> std::ops::Deref for Owned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.val
    }
}

impl<T> std::ops::DerefMut for Owned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.val
    }
}

impl<T> Owned<T> {
    pub fn new(val: T) -> Self {
        Self { val: Box::new(val) }
    }

    pub fn into_shared<'g>(self, _: &'g Guard<'_>) -> Shared<'g, T> {
        Shared {
            inner: unsafe { flize::Shared::from_ptr(Box::into_raw(self.val)) },
        }
    }
}

pub struct Shared<'a, T> {
    inner: flize::Shared<'a, T, flize::NullTag, flize::NullTag, 0, 0>,
}

impl<'a, T> Clone for Shared<'a, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'g, T> Copy for Shared<'g, T> {}

impl<'g, T> Shared<'g, T> {
    pub fn null() -> Self {
        Self {
            inner: flize::Shared::null(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }

    pub unsafe fn as_ref(self) -> Option<&'g T> {
        unsafe { self.inner.as_ref() }
    }

    pub unsafe fn from_ptr(ptr: *mut T) -> Self {
        Self {
            inner: unsafe { flize::Shared::from_ptr(ptr) },
        }
    }

    pub unsafe fn into_owned(self) -> Owned<T> {
        Owned {
            val: unsafe { Box::from_raw(self.as_ptr()) },
        }
    }

    pub unsafe fn deref(self) -> &'g T {
        unsafe { self.inner.as_ref_unchecked() }
    }

    pub fn as_ptr(&self) -> *mut T {
        self.inner.as_ptr()
    }
}

pub struct Guard<'a> {
    inner: flize::ThinShield<'a>,
}

impl<'a> Guard<'a> {
    pub fn retire<F>(&self, f: F)
    where
        F: FnOnce() + 'a,
    {
        self.inner.retire(f);
    }
}

pub struct Collector {
    inner: flize::Collector,
}

impl Collector {
    pub fn new() -> Self {
        Self {
            inner: flize::Collector::new(),
        }
    }

    pub fn guard(&self) -> Guard<'_> {
        Guard {
            inner: self.inner.thin_shield(),
        }
    }
}
