use crate::atomic_array::{AtomicArray, AtomicArrayBuilder};
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
    // The maximum number of elements per lock before we re-allocate.
    budget: AtomicUsize,
    // The builder used to hash map keys.
    build_hasher: S,
    // The internal state of this map.
    //
    // Wrapping this in a separate struct allows us
    // to atomically swap everything at once.
    table: Atomic<Table<K, V>>,
    // Instead of adding garbage to the default global collector
    // we add it to a local collector tied to this particular map.
    //
    // Since the global collector might destroy garbage arbitrarily
    // late in the future, we would have to add `K: 'static` and `V: 'static`
    // bounds. But a local collector will destroy all remaining garbage when
    // the map is dropped, so we can accept non-'static keys and values.
    collector: Collector,
}

struct Table<K, V> {
    // The hashtable.
    buckets: Box<[Atomic<Node<K, V>>]>,
    // A set of locks, each guarding a number of buckets.
    locks: Arc<[Arc<Mutex<()>>]>,
    // The number of elements guarded by each lock.
    count_per_lock: AtomicArray<AtomicUsize>,
}

//A singly-linked list representing a bucket in the hashtable
struct Node<K, V> {
    // The key of this node.
    key: K,
    // The value of this node, heap-allocated in order to be shared
    // across buckets instances during resize operations.
    value: NonNull<V>,
    // The next node in the linked-list.
    next: Atomic<Self>,
    // The hashcode of `key`.
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
        Self::with_capacity_and_hasher(1, RandomState::new())
    }
}

impl<K, V, S> HashMap<K, V, S> {
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
        let table = Table {
            buckets: std::iter::repeat_with(|| Atomic::null())
                .take(capacity)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            locks: std::iter::repeat_with(|| Arc::new(Mutex::new(())))
                .take(num_cpus)
                .collect::<Vec<_>>()
                .into(),
            count_per_lock: AtomicArrayBuilder::from_fn(|| AtomicUsize::new(0), num_cpus).build(),
        };

        Self {
            budget: AtomicUsize::new(capacity / num_cpus),
            table: Atomic::new(table),
            collector: Collector::new(),
            build_hasher: hash_builder,
        }
    }
}

impl<K, V, S> HashMap<K, V, S> {
    fn guard(&self) -> Guard {
        self.collector.register().pin()
    }

    /// Returns a reference to the map pinned to the current thread.
    ///
    /// The only way to access a map is through a pinned reference, which,
    /// when dropped, will trigger garbage collection.
    pub fn pin(&self) -> Pinned<'_, K, V, S> {
        Pinned {
            map: self,
            guard: self.guard(),
        }
    }

    /// Returns the number of elements in the map.
    pub(crate) fn len(&self, guard: &Guard) -> usize {
        let table = unsafe { self.table.load(Ordering::Acquire, guard).deref() };

        // TODO: is this too slow? should we store a cached total length?
        unsafe { table.count_per_lock.load(Ordering::Acquire, guard) }
            .iter()
            .fold(0, |acc, c| acc + c.load(Ordering::Relaxed))
    }

    /// Returns `true` if the map contains no elements.
    pub(crate) fn is_empty(&self, guard: &Guard) -> bool {
        self.len(guard) == 0
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    pub(crate) fn iter<'g>(&self, guard: &'g Guard) -> Iter<'g, K, V> {
        let table = unsafe { self.table.load(Ordering::Acquire, guard).deref() };
        Iter {
            size: self.len(guard),
            current_node: table.buckets.get(0),
            table,
            current_bucket: 0,
            guard,
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    pub(crate) fn keys<'g>(&self, guard: &'g Guard) -> Keys<'g, K, V> {
        Keys {
            iter: self.iter(guard),
        }
    }

    /// An iterator visiting all values in arbitrary order.
    pub(crate) fn values<'g>(&self, guard: &'g Guard) -> Values<'g, K, V> {
        Values {
            iter: self.iter(guard),
        }
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Returns a reference to the value corresponding to the key.
    pub(crate) fn get<'g, Q: ?Sized>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(key);
        let table = unsafe { self.table.load(Ordering::Acquire, guard).deref() };
        let bucket_index = bucket_index(hash, table.buckets.len() as _);

        let mut node_ptr = table.buckets.get(bucket_index as usize);

        while let Some(node) = node_ptr {
            // Look Ma, no lock!
            //
            // The atomic load ensures that we have a valid reference to
            // table.buckets[bucket_index]. This protects us from reading
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

    /// Returns `true` if the map contains a value for the specified key.
    pub(crate) fn contains_key<Q: ?Sized>(&self, key: &Q, guard: &Guard) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(key, &guard).is_some()
    }

    /// Acquires all locks for this hash buckets.
    fn lock_all<'g>(&'g self, guard: &'g Guard) -> (impl Drop + 'g, impl Drop + 'g) {
        let table = unsafe { self.table.load(Ordering::Acquire, guard).deref() };
        let locks = &table.locks;

        // Acquire the first lock.
        let _guard = self.lock_range(0..1, guard);

        // Now that we have the first lock, the locks array cannnot change (i.e., be resized),
        // and so we can safely read locks.len().
        let _guard_rest = self.lock_range(1..locks.len(), guard);

        (_guard, _guard_rest)
    }

    /// Acquires a contiguous range of locks for this hash buckets.
    fn lock_range<'g>(&'g self, range: Range<usize>, guard: &'g Guard) -> impl Drop + 'g {
        let table = unsafe { self.table.load(Ordering::Acquire, guard).deref() };
        let locks = &table.locks;

        for i in range.clone() {
            unsafe {
                locks[i].raw().lock();
            }
        }

        /// Unlocks the specified locks on drop.
        struct Unlock<'g, K, V, S> {
            map: &'g HashMap<K, V, S>,
            guard: &'g Guard,
            range: Range<usize>,
        }

        impl<K, V, S> Drop for Unlock<'_, K, V, S> {
            fn drop(&mut self) {
                let table = unsafe { self.map.table.load(Ordering::Acquire, self.guard).deref() };
                let locks = &table.locks;

                for i in self.range.clone() {
                    unsafe {
                        locks[i].raw().unlock();
                    }
                }
            }
        }

        Unlock {
            map: self,
            guard,
            range,
        }
    }

    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        let mut h = self.build_hasher.build_hasher();
        key.hash(&mut h);
        h.finish()
    }
}

// methods that required K/V: Send + Sync, as they potentially call
// `defer_destroy` on keys and values, and the garbage collector
// may execute the destructors on another thread
impl<K, V, S> HashMap<K, V, S>
where
    V: Send + Sync,
    K: Hash + Eq + Send + Sync + Clone,
    S: BuildHasher,
{
    /// The maximum # of buckets the buckets can hold.
    const MAX_BUCKETS: usize = isize::MAX as _;

    /// The maximum size of the `locks` array.
    const MAX_LOCKS: usize = 1024;

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical.
    pub(crate) fn insert<'g>(&'g self, key: K, value: V, guard: &'g Guard) -> Option<&'g V> {
        let hash = self.hash(&key);
        let mut should_resize = false;

        loop {
            let table_ptr = self.table.load(Ordering::Acquire, guard);
            let table = unsafe { table_ptr.deref() };

            let bucket_index = bucket_index(hash, table.buckets.len() as _);
            let lock_index = lock_index(bucket_index, table.locks.len() as _);

            {
                let _guard = table.locks[lock_index as usize].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !std::ptr::eq(
                    table_ptr.as_raw(),
                    self.table.load(Ordering::Acquire, guard).as_raw(),
                ) {
                    continue;
                }

                // Try to find this key in the bucket
                let mut curr = table.buckets.get(bucket_index as usize);
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

                    // If the key already exists, update the value.
                    if hash == node.hash && node.key == key {
                        // Here we create a new node instead of updating the value atomically.
                        //
                        // This is a tradeoff, we don't have to do atomic loads on reads,
                        // but we have to do an extra allocation here for the node.
                        let new_node = Node {
                            key,
                            next: node.next.clone(),
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

                let node = &table.buckets[bucket_index as usize];

                let new = Node {
                    key,
                    next: node.clone(),
                    value: NonNull::from(Box::leak(Box::new(value))),
                    hash,
                };

                node.store(Owned::new(new), Ordering::Release);

                let count = unsafe {
                    table.count_per_lock.load(Ordering::Acquire, guard)[lock_index as usize]
                        .fetch_add(1, Ordering::AcqRel)
                        + 1
                };

                // If the number of elements guarded by this lock has exceeded the budget, resize
                // the hash table.
                //
                // It is also possible that GrowTable will increase the budget but won't resize the
                // hash table, if it is being poorly utilized due to a bad hash function.
                if count > self.budget.load(Ordering::SeqCst) {
                    should_resize = true;
                }
            }

            // We just performed an insertion. If necessary, we will resize the buckets.
            // Notice that we are not holding any locks when calling resize.
            // This is necessary to prevent deadlocks. As a result, it is possible that
            // resize will be called unnecessarily. But, it will obtain lock 0 and
            // then verify that the table we passed to it as the argument is still the
            // current table.
            if should_resize {
                self.resize(table_ptr, guard);
            }

            return None;
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    pub(crate) fn remove<'g, Q: ?Sized>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(key);

        loop {
            let table_ptr = self.table.load(Ordering::Acquire, guard);
            let table = unsafe { table_ptr.deref() };

            let bucket_index = bucket_index(hash, table.buckets.len() as _);
            let lock_index = lock_index(bucket_index, table.locks.len() as _);

            {
                let _guard = table.locks[lock_index as usize].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !std::ptr::eq(
                    table_ptr.as_raw(),
                    self.table.load(Ordering::Acquire, guard).as_raw(),
                ) {
                    continue;
                }

                let mut curr = table.buckets.get(bucket_index as usize);

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

                        unsafe {
                            table.count_per_lock.load(Ordering::Acquire, guard)[lock_index as usize]
                                .fetch_min(1, Ordering::AcqRel)
                        };

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
    pub(crate) fn clear(&self, guard: &Guard) {
        let _guard = self.lock_all(guard);

        let table_ptr = self.table.load(Ordering::Acquire, guard);
        let table = unsafe { table_ptr.deref() };

        let new_table = Table {
            buckets: std::iter::repeat_with(|| Atomic::null())
                .take(0)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            locks: table.locks.clone(),
            count_per_lock: AtomicArrayBuilder::from_fn(|| AtomicUsize::new(0), table.locks.len())
                .build(),
        };

        let new_budget = new_table.buckets.len() / new_table.locks.len();

        self.table.store(Owned::new(new_table), Ordering::Release);
        self.budget.store(1.max(new_budget), Ordering::SeqCst);

        // TODO: destroy everything
        unsafe {
            guard.defer_destroy(table_ptr);
        };
    }

    /// Replaces the bucket buckets with a larger one.
    ///
    /// To prevent multiple threads from resizing the buckets as a result of races,
    /// the `table` instance that holds the buckets of buckets deemed too small must
    /// be passed as an argument. `resize` obtains a lock, and then checks if the
    /// buckets has been replaced in the meantime or not.
    ///
    /// This function may not resize the table if it values are found to be poorly
    /// distributed across buckets. Instead the budget will be increased.
    fn resize<'g>(&'g self, table_ptr: Shared<'g, Table<K, V>>, guard: &'g Guard) {
        let table = unsafe { table_ptr.deref() };

        // The thread that first obtains lock 0 will be the one doing the resize operation
        let _guard = self.lock_range(0..1, guard);

        // Make sure nobody resized the table while we were waiting for lock 0:
        if !std::ptr::eq(
            table_ptr.as_raw(),
            self.table.load(Ordering::Acquire, guard).as_raw(),
        ) {
            // We assume that since the table reference is different,
            // it was already resized (or the budget was adjusted).
            // If we ever decide to do table shrinking, or replace the table for other reasons,
            // we will have to revisit this logic.
            return;
        }

        let approx_len = unsafe { table.count_per_lock.load(Ordering::Acquire, guard) }
            .iter()
            .fold(0, |acc, c| acc + c.load(Ordering::Relaxed));

        // If the bucket array is too empty, double the budget instead of resizing the table
        if approx_len < table.buckets.len() / 4 {
            let budget = self.budget.load(Ordering::SeqCst);
            self.budget.store(budget * 2, Ordering::SeqCst);
        }

        let mut new_len = 0;
        let mut max_size = false;

        // Compute the new buckets size.
        //
        // The current method is to find the smallest integer that is:
        //
        // 1) larger than twice the previous buckets size
        // 2) not divisible by 2, 3, 5 or 7.
        //
        // This may change in the future.
        let mut compute_new_size = || {
            // Double the size of the buckets buckets and add one, so that we have an odd integer.
            new_len = table.buckets.len().checked_mul(2)?.checked_add(1)?;

            // Now, we only need to check odd integers, and find the first that is not divisible
            // by 3, 5 or 7.
            while new_len.checked_rem(3)? == 0
                || new_len.checked_rem(5)? == 0
                || new_len.checked_rem(7)? == 0
            {
                new_len = new_len.checked_add(2)?;
            }

            debug_assert!(new_len % 2 != 0);

            if new_len > Self::MAX_BUCKETS {
                max_size = true;
            }

            Some(())
        };

        match compute_new_size() {
            Some(()) => {}
            None => {
                max_size = true;
            }
        }

        if max_size {
            new_len = Self::MAX_BUCKETS;

            // Set the budget to `usize::MAX` to make sure `resize_buckets`
            // is never called again (unless the buckets is shrunk), because
            // the buckets is at it's maximum size.
            self.budget.store(usize::MAX, Ordering::SeqCst);
        }

        // Now lock the rest of the buckets.
        //
        // TODO: could this go after creating the new locks?
        let _guard_rest = self.lock_range(1..table.locks.len(), guard);

        let mut new_locks = None;
        let mut new_locks_len = table.locks.len();

        // Add more locks to account for the new buckets.
        if table.locks.len() < Self::MAX_LOCKS {
            new_locks_len = table.locks.len() * 2;
            let mut locks = Vec::with_capacity(new_locks_len);

            for i in 0..table.locks.len() {
                locks.push(table.locks[i].clone());
            }

            for _ in table.locks.len()..locks.len() {
                locks.push(Arc::new(Mutex::new(())));
            }

            new_locks = Some(locks);
        }

        let new_count_per_lock = AtomicArrayBuilder::from_fn(|| AtomicUsize::new(0), new_locks_len);

        let mut new_buckets = std::iter::repeat_with(|| Atomic::null())
            .take(new_len)
            .collect::<Vec<_>>();

        // Copy all data into a new buckets, creating new nodes for all elements.
        for i in 0..table.buckets.len() {
            let mut current = unsafe { table.buckets[i].load(Ordering::Acquire, guard).as_ref() };

            while let Some(node) = current {
                let next = unsafe { node.next.load(Ordering::Acquire, guard).as_ref() };

                let new_bucket_index = bucket_index(node.hash, new_buckets.capacity() as _);
                let new_lock_index = lock_index(new_bucket_index, table.locks.len() as _);

                new_buckets[new_bucket_index as usize] = Atomic::new(Node {
                    key: node.key.clone(),
                    value: node.value.clone(),
                    next: new_buckets[new_bucket_index as usize].clone(),
                    hash: node.hash,
                });

                new_count_per_lock.as_ref()[new_lock_index as usize]
                    .fetch_add(1, Ordering::Relaxed);

                current = next;
            }
        }

        // Update the budget.
        self.budget.store(
            1.max(new_buckets.capacity() / new_locks_len),
            Ordering::SeqCst,
        );

        // Replace `self.table` with the updated version atomically.
        self.table.store(
            Owned::new(Table {
                buckets: new_buckets.into_boxed_slice(),
                locks: new_locks
                    .map(Arc::from)
                    .unwrap_or_else(|| table.locks.clone()),
                count_per_lock: new_count_per_lock.build(),
            }),
            Ordering::Release,
        );

        unsafe {
            guard.defer_destroy(table_ptr);
        }
    }
}

/// Computes the bucket index for a particular key.
fn bucket_index(hashcode: u64, bucket_count: u64) -> u64 {
    let bucket_index = (hashcode & 0x7fffffff) % bucket_count;
    debug_assert!(bucket_index < bucket_count);

    bucket_index
}

/// Computes the lock index for a particular bucket.
fn lock_index(bucket_index: u64, lock_count: u64) -> u64 {
    let lock_index = bucket_index % lock_count;
    debug_assert!(lock_index < lock_count);

    lock_index
}

/// An iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`iter`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`iter`]: HashMap::iter
///
/// # Example
///
/// ```
/// use concurrent_dictionary::HashMap;
///
/// let mut map = HashMap::new();
/// map.insert("a", 1);
/// let iter = map.iter();
/// ```
pub struct Iter<'g, K, V> {
    table: &'g Table<K, V>,
    guard: &'g Guard,
    current_node: Option<&'g Atomic<Node<K, V>>>,
    current_bucket: usize,
    size: usize,
}

impl<'g, K, V> Iterator for Iter<'g, K, V> {
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(node) = &self
                .current_node
                .and_then(|n| unsafe { n.load(Ordering::Acquire, self.guard).as_ref() })
            {
                self.size -= 1;
                self.current_node = Some(&node.next);
                return Some((&node.key, unsafe { node.value.as_ref() }));
            }

            self.current_bucket += 1;

            if self.current_bucket >= self.table.buckets.len() {
                return None;
            }

            self.current_node = Some(&self.table.buckets[self.current_bucket]);
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            table: self.table,
            guard: self.guard,
            current_node: self.current_node,
            current_bucket: self.current_bucket,
            size: self.size,
        }
    }
}

/// An iterator over the keys of a `HashMap`.
///
/// This `struct` is created by the [`keys`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`keys`]: HashMap::keys
///
/// # Example
///
/// ```
/// use concurrent_dictionary::HashMap;
///
/// let mut map = HashMap::new();
/// map.insert("a", 1);
/// let iter_keys = map.keys();
/// ```
pub struct Keys<'g, K, V> {
    iter: Iter<'g, K, V>,
}

impl<'g, K, V> Iterator for Keys<'g, K, V> {
    type Item = &'g K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }
}

impl<K, V> Clone for Keys<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

/// An iterator over the values of a `HashMap`.
///
/// This `struct` is created by the [`values`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`values`]: HashMap::values
///
/// # Example
///
/// ```
/// use concurrent_dictionary::HashMap;
///
/// let mut map = HashMap::new();
/// map.insert("a", 1);
/// let iter_values = map.values();
/// ```
pub struct Values<'g, K, V> {
    iter: Iter<'g, K, V>,
}

impl<'g, K, V> Iterator for Values<'g, K, V> {
    type Item = &'g V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }
}

impl<K, V> Clone for Values<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<K, V, S> Clone for HashMap<K, V, S>
where
    K: Sync + Send + Clone + Hash + Eq,
    V: Sync + Send + Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> HashMap<K, V, S> {
        let self_pinned = self.pin();

        let clone = Self::with_capacity_and_hasher(self_pinned.len(), self.build_hasher.clone());
        let clone_pinned = clone.pin();

        for (k, v) in self_pinned.iter() {
            clone_pinned.insert(k.clone(), v.clone());
        }

        clone
    }
}

impl<K, V, S> Default for HashMap<K, V, S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    pub(crate) fn eq_pinned(this: &Pinned<'_, K, V, S>, other: &Pinned<'_, K, V, S>) -> bool {
        if this.len() != other.len() {
            return false;
        }

        this.iter()
            .all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S> PartialEq for HashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.pin() == other.pin()
    }
}

impl<K, V, S> Eq for HashMap<K, V, S>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{
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

        assert!([
            pinned.iter().count(),
            pinned.len(),
            pinned.iter().size_hint().0,
            pinned.iter().size_hint().1.unwrap()
        ]
        .iter()
        .all(|&l| l == 100));
    }
}
