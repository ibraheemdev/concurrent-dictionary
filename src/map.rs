use crate::Pinned;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crossbeam_epoch::{Atomic, Collector, Guard, Owned, Shared};
use lock_api::RawMutex;
use parking_lot::Mutex;

pub struct HashMap<K, V, S = RandomState> {
    len: AtomicUsize,
    budget: AtomicUsize,
    hash_builder: S,
    tables: Atomic<Tables<K, V>>,
    // Instead of adding garbage to the default global collector
    // we add it to a local collector tied to this particular map.
    //
    // Since the global collector might destroy garbage arbitrarily
    // late in the future, we would have to add `K: 'static` and `V: 'static`
    // bounds. But a local collector will destroy all remaining garbage when
    // the map is dropped, so we can accept non-'static keys and values.
    collector: Collector,
}

struct Tables<K, V> {
    buckets: Box<[Atomic<Node<K, V>>]>,
    locks: Arc<[Arc<Mutex<()>>]>,
}

struct Node<K, V> {
    key: K,
    value: NonNull<V>,
    next: Atomic<Self>,
    hash: u64,
}

impl<K, V> HashMap<K, V, RandomState> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    /// let map: HashMap<&str, i32> = HashMap::new();
    /// ```
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(31, RandomState::new())
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    S: BuildHasher,
{
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The created map has the default initial capacity.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let map = HashMap::with_hasher(s);
    /// let pinned = map.pin();
    ///
    /// pinned.insert(1, 2);
    /// ```
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    /// Creates an empty `HashMap` with the specified capacity, using `hash_builder`
    /// to hash the keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = HashMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, 2);
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<K, V, S> {
        let num_cpus = num_cpus::get();
        let tables = Tables {
            buckets: std::iter::repeat_with(|| Atomic::null())
                .take(capacity)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            locks: std::iter::repeat_with(|| Arc::new(Mutex::new(())))
                .take(num_cpus)
                .collect::<Vec<_>>()
                .into(),
        };

        Self {
            len: AtomicUsize::new(0),
            budget: AtomicUsize::new(capacity / num_cpus),
            tables: Atomic::new(tables),
            collector: Collector::new(),
            hash_builder,
        }
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    V: Send + Sync,
    K: Hash + Eq + Send + Sync + Clone,
    S: BuildHasher,
{
    /// Returns a reference to the map pinned to the current thread.
    ///
    /// The only way to access a map is through a pinned reference, which,
    /// when dropped, will trigger garbage collection.
    pub fn pin(&self) -> Pinned<'_, K, V, S> {
        Pinned {
            map: self,
            guard: self.collector.register().pin(),
        }
    }

    pub(crate) fn insert<'g>(&'g self, key: K, value: V, guard: &'g Guard) -> Option<&'g V> {
        let hash = self.hash(&key);
        let mut should_resize = false;

        loop {
            let tables_ptr = self.tables.load(Ordering::Acquire, guard);
            let tables = unsafe { tables_ptr.deref() };

            let bucket_index = bucket_index(hash, tables.buckets.len() as _);
            let lock_index = lock_index(bucket_index, tables.locks.len() as _);

            {
                let _guard = tables.locks[lock_index as usize].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !std::ptr::eq(
                    tables_ptr.as_raw(),
                    self.tables.load(Ordering::Acquire, guard).as_raw(),
                ) {
                    continue;
                }

                let mut curr = tables.buckets.get(bucket_index as usize);

                loop {
                    let node_ptr = match curr {
                        Some(node_ptr) => node_ptr,
                        None => break,
                    };

                    let node_ref = node_ptr.load(Ordering::Acquire, guard);
                    let node = match unsafe { node_ref.as_ref() } {
                        Some(node) => node,
                        None => break,
                    };

                    if hash == node.hash && node.key == key {
                        let new_node = Node {
                            next: node.next.clone(),
                            key,
                            value: NonNull::from(Box::leak(Box::new(value))),
                            hash,
                        };

                        node_ptr.store(Owned::new(new_node), Ordering::Release);

                        let old_val = unsafe { node.value.as_ref() };

                        unsafe {
                            guard.defer_destroy(node_ref);
                            guard.defer_destroy(Shared::from(old_val as *const _));
                        };

                        return Some(old_val);
                    }

                    curr = Some(&node.next);
                }

                let node = &tables.buckets[bucket_index as usize];

                let new = Node {
                    key,
                    next: node.clone(),
                    value: NonNull::from(Box::leak(Box::new(value))),
                    hash,
                };

                node.store(Owned::new(new), Ordering::Release);

                let len = self.len.fetch_add(1, Ordering::SeqCst);

                if len + 1 > self.budget.load(Ordering::SeqCst) {
                    should_resize = true;
                }
            }

            // We just performed an insertion. If necessary, we will grow the table.
            // Notice that we are not holding any locks when calling grow_table.
            // This is necessary to prevent deadlocks. As a result, it is possible that
            // grow_table will be called unnecessarily. But, it will obtain lock 0 and
            // then verify that the table we passed to it as the argument is still the
            // current table.
            if should_resize {
                self.grow_table(tables_ptr, guard);
            }

            return None;
        }
    }

    pub(crate) fn get<'g, Q: ?Sized>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(key);
        let tables = unsafe { self.tables.load(Ordering::Acquire, guard).deref() };
        let bucket_index = bucket_index(hash, tables.buckets.len() as _);

        let mut node_ptr = tables.buckets.get(bucket_index as usize);

        while let Some(node) = node_ptr {
            // Lock-free reads!
            //
            // The atomic load ensures that we have a valid reference to
            // tables.buckets[bucket_index]. This protects us from reading
            // node fields of different instances.
            let node = match unsafe { node.load(Ordering::Acquire, guard).as_ref() } {
                Some(node) => node,
                None => break,
            };

            if node.hash == hash && node.key.borrow() == key {
                return Some(unsafe { node.value.as_ref() });
            }

            node_ptr = Some(&node.next);
        }

        None
    }

    pub(crate) fn remove<'g, Q: ?Sized>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(key);

        loop {
            let tables_ptr = self.tables.load(Ordering::Acquire, guard);
            let tables = unsafe { tables_ptr.deref() };

            let bucket_index = bucket_index(hash, tables.buckets.len() as _);
            let lock_index = lock_index(bucket_index, tables.locks.len() as _);

            {
                let _guard = tables.locks[lock_index as usize].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !std::ptr::eq(
                    tables_ptr.as_raw(),
                    self.tables.load(Ordering::Acquire, guard).as_raw(),
                ) {
                    continue;
                }

                let mut curr = tables.buckets.get(bucket_index as usize);

                loop {
                    let node_ptr = match curr {
                        Some(node_ptr) => node_ptr,
                        None => break,
                    };

                    let node_ref = node_ptr.load(Ordering::Acquire, guard);
                    let node = match unsafe { node_ref.as_ref() } {
                        Some(node) => node,
                        None => break,
                    };

                    if hash == node.hash && node.key.borrow() == key {
                        let next = node.next.load(Ordering::Acquire, guard);
                        node_ptr.store(next, Ordering::Release);

                        self.len.fetch_min(1, Ordering::SeqCst);

                        let val = unsafe { node.value.as_ref() };

                        unsafe {
                            guard.defer_destroy(node_ref);
                            guard.defer_destroy(Shared::from(val as *const _));
                        }

                        return Some(val);
                    }

                    curr = Some(&node.next);
                }
            }

            return None;
        }
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use flurry::HashMap;
    ///
    /// let map = HashMap::new();
    ///
    /// map.pin().insert(1, "a");
    /// map.pin().clear();
    /// assert!(map.pin().is_empty());
    /// ```
    fn clear(&self, guard: &Guard) {
        let _guard = self.lock_all(guard);

        let tables_ptr = self.tables.load(Ordering::Acquire, guard);
        let tables = unsafe { tables_ptr.deref() };

        let new_tables = Tables {
            buckets: std::iter::repeat_with(|| Atomic::null())
                .take(0)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            locks: tables.locks.clone(),
        };

        let new_budget = new_tables.buckets.len() / new_tables.locks.len();

        self.tables.store(Owned::new(new_tables), Ordering::Release);
        self.budget.store(1.max(new_budget), Ordering::SeqCst);

        unsafe {
            guard.defer_destroy(tables_ptr);
        };
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn contains_key<Q: ?Sized>(&self, key: &Q, guard: &Guard) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(key, &guard).is_some()
    }

    fn grow_table<'g>(&'g self, tables_ptr: Shared<'g, Tables<K, V>>, guard: &'g Guard) {
        const MAX_ARRAY_LENGTH: usize = isize::MAX as _;
        const MAX_LOCKS: usize = 1024;

        let tables = unsafe { tables_ptr.deref() };

        // The thread that first obtains the first will be the one doing the resize operation
        let _guard = self.lock_range(0..1, guard);

        // Make sure nobody resized the table while we were waiting for lock 0:
        if !std::ptr::eq(
            tables_ptr.as_raw(),
            self.tables.load(Ordering::Acquire, guard).as_raw(),
        ) {
            // We assume that since the table reference is different, it was already resized (or the budget
            // was adjusted). If we ever decide to do table shrinking, or replace the table for other reasons,
            // we will have to revisit this logic.
            return;
        }

        let len = self.len();

        // If the bucket array is too empty, double the budget instead of resizing the table
        if len < tables.buckets.len() / 4 {
            let budget = self.budget.load(Ordering::SeqCst);
            self.budget.store(budget * 2, Ordering::SeqCst);
        }

        let mut new_len = 0;
        let mut max_size = false;

        let mut _try = || {
            new_len = tables.buckets.len().checked_mul(2)?.checked_add(1)?;

            while new_len.checked_rem(3)? == 0
                || new_len.checked_rem(5)? == 0
                || new_len.checked_rem(7)? == 0
            {
                new_len = new_len.checked_add(2)?;
            }

            debug_assert!(new_len % 2 != 0);

            if new_len > MAX_ARRAY_LENGTH {
                max_size = true;
            }

            Some(())
        };

        match _try() {
            Some(()) => {}
            None => {
                max_size = true;
            }
        }

        if max_size {
            new_len = MAX_ARRAY_LENGTH;
            self.budget.store(usize::MAX, Ordering::SeqCst);
        }

        let _guard_rest = self.lock_range(1..tables.locks.len(), guard);

        let mut new_locks = None;
        let mut new_locks_len = tables.locks.len();

        // add more locks
        if tables.locks.len() < MAX_LOCKS {
            new_locks_len = tables.locks.len() * 2;
            let mut locks = Vec::with_capacity(new_locks_len);

            for i in 0..tables.locks.len() {
                locks.push(tables.locks[i].clone());
            }

            for _ in tables.locks.len()..locks.len() {
                locks.push(Arc::new(Mutex::new(())));
            }

            new_locks = Some(locks);
        }

        let mut new_buckets = std::iter::repeat_with(|| Atomic::null())
            .take(new_len)
            .collect::<Vec<_>>();

        for i in 0..tables.buckets.len() {
            let mut current = unsafe { tables.buckets[i].load(Ordering::Acquire, guard).as_ref() };

            while let Some(node) = current {
                let next = unsafe { node.next.load(Ordering::Acquire, guard).as_ref() };

                let new_bucket_index = bucket_index(node.hash, new_buckets.capacity() as _);

                new_buckets[new_bucket_index as usize] = Atomic::new(Node {
                    key: node.key.clone(),
                    value: node.value.clone(),
                    next: new_buckets[new_bucket_index as usize].clone(),
                    hash: node.hash,
                });

                current = next;
            }
        }

        self.budget.store(
            1.max(new_buckets.capacity() / new_locks_len),
            Ordering::SeqCst,
        );

        self.tables.store(
            Owned::new(Tables {
                buckets: new_buckets.into_boxed_slice(),
                locks: new_locks
                    .map(Arc::from)
                    .unwrap_or_else(|| tables.locks.clone()),
            }),
            Ordering::Release,
        );

        unsafe {
            guard.defer_destroy(tables_ptr);
        }
    }

    fn lock_all<'g>(&'g self, guard: &'g Guard) -> (impl Drop + 'g, impl Drop + 'g) {
        let tables = unsafe { self.tables.load(Ordering::Acquire, guard).deref() };
        let locks = &tables.locks;

        // First, acquire lock 0
        let _guard = self.lock_range(0..1, guard);

        // Now that we have lock 0, the locks array will not change (i.e., grow),
        // and so we can safely read locks.len().
        let _guard_rest = self.lock_range(1..locks.len(), guard);

        (_guard, _guard_rest)
    }

    fn lock_range<'g>(&'g self, range: Range<usize>, guard: &'g Guard) -> impl Drop + 'g {
        let tables = unsafe { self.tables.load(Ordering::Acquire, guard).deref() };
        let locks = &tables.locks;

        for i in range.clone() {
            unsafe {
                locks[i].raw().lock();
            }
        }

        struct UnlockRange<'g, K, V, S> {
            map: &'g HashMap<K, V, S>,
            guard: &'g Guard,
            range: Range<usize>,
        }

        impl<K, V, S> Drop for UnlockRange<'_, K, V, S> {
            fn drop(&mut self) {
                let tables = unsafe { self.map.tables.load(Ordering::Acquire, self.guard).deref() };
                let locks = &tables.locks;

                for i in self.range.clone() {
                    unsafe {
                        locks[i].raw().unlock();
                    }
                }
            }
        }

        UnlockRange {
            map: self,
            guard,
            range,
        }
    }

    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        let mut h = self.hash_builder.build_hasher();
        key.hash(&mut h);
        h.finish()
    }
}

// impl<K, V, S> Clone for HashMap<K, V, S>
// where
//     K: Sync + Send + Clone + Hash + Eq,
//     V: Sync + Send + Clone,
//     S: BuildHasher + Clone,
// {
//     fn clone(&self) -> HashMap<K, V, S> {
//         let cloned_map = Self::with_capacity_and_hasher(self.len(), self.build_hasher.clone());
//
//         {
//             let guard = crossbeam_epoch::pin();
//             for (k, v) in self.iter(&guard) {
//                 cloned_map.insert(k.clone(), v.clone(), &guard);
//             }
//         }
//
//         cloned_map
//     }
// }

fn lock_index(bucket_index: u64, lock_count: u64) -> u64 {
    let lock_index = bucket_index % lock_count;
    debug_assert!(lock_index < lock_count);

    lock_index
}

fn bucket_index(hashcode: u64, bucket_count: u64) -> u64 {
    let bucket_index = (hashcode & 0x7fffffff) % bucket_count;
    debug_assert!(bucket_index < bucket_count);

    bucket_index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let map = HashMap::new();
        let pinned = map.pin();
        pinned.insert(1, "a");
        pinned.insert(2, "b");
        assert_eq!(pinned.get(&1), Some(&"a"));
        assert_eq!(pinned.get(&2), Some(&"b"));
        assert_eq!(pinned.get(&3), None);

        assert_eq!(pinned.remove(&2), Some(&"b"));
        assert_eq!(pinned.remove(&2), None);

        for i in 0..100 {
            pinned.insert(i, "a");
        }

        for i in 0..100 {
            assert_eq!(pinned.get(&i), Some(&"a"));
        }
    }
}
