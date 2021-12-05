use crate::reclaim::{Atomic, Collector, Guard, Owned, Shared};
use crate::resize;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::{fmt, iter};

use lock_api::RawMutex;
use parking_lot::Mutex;

pub struct RawTable<K, V, S = RandomState> {
    // The maximum number of elements per lock before we re-allocate.
    budget: AtomicUsize,
    // The builder used to hash map keys.
    hash_builder: S,
    // The inner state of this map.
    //
    // Wrapping this in a separate struct allows us
    // to atomically swap everything at once.
    table: Atomic<Inner<K, V>>,
    // Instead of adding garbage to the default global collector
    // we add it to a local collector tied to this particular map.
    //
    // Since the global collector might destroy garbage arbitrarily
    // late in the future, we would have to add `K: 'static` and `V: 'static`
    // bounds. But a local collector will destroy all remaining garbage when
    // the map is dropped, so we can accept non-'static keys and values.
    collector: Collector,
    // The lock acquired when resizing.
    resize: Arc<Mutex<()>>,
}

struct Inner<K, V> {
    // The hashtable.
    buckets: Box<[Atomic<Node<K, V>>]>,
    // A set of locks, each guarding a number of buckets.
    //
    // Locks are shared across table instances during
    // resizing, hence the `Arc`s.
    locks: Arc<[Arc<Mutex<()>>]>,
    // The number of elements guarded by each lock.
    counts: Box<[AtomicUsize]>,
}

impl<K, V> Inner<K, V> {
    fn new(buckets: usize, locks: usize) -> Self {
        Self {
            buckets: iter::repeat_with(Atomic::null).take(buckets).collect(),
            locks: iter::repeat_with(Arc::default).take(locks).collect(),
            counts: iter::repeat_with(AtomicUsize::default)
                .take(locks)
                .collect(),
        }
    }

    fn len(&self) -> usize {
        self.counts
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum()
    }

    fn lock_buckets(&self) -> impl Drop + '_ {
        for lock in self.locks.iter() {
            unsafe { lock.raw().lock() }
        }

        struct Guard<'a, K, V>(&'a Inner<K, V>);

        impl<K, V> Drop for Guard<'_, K, V> {
            fn drop(&mut self) {
                for lock in self.0.locks.iter() {
                    unsafe { lock.raw().unlock() }
                }
            }
        }

        Guard(self)
    }

    pub fn bucket<'a>(
        &'a self,
        hash: u64,
        guard: &'a Guard<'a>,
    ) -> impl Iterator<
        Item = (
            &'a Atomic<Node<K, V>>,
            Shared<'a, Node<K, V>>,
            &'a Node<K, V>,
        ),
    > {
        let bucket = bucket_index(hash, self.buckets.len() as _) as usize;

        let mut atomic = &self.buckets[bucket];
        std::iter::from_fn(move || {
            let ptr = atomic.load(Ordering::Acquire, guard);
            if ptr.is_null() {
                return None;
            }

            let node = unsafe { ptr.deref() };
            let val = Some((atomic, ptr, node));
            atomic = &node.next;
            val
        })
    }
}

//A singly-linked list representing a bucket in the hashtable
struct Node<K, V> {
    // The key of this node.
    key: K,
    // The hashcode of `key`.
    hash: u64,
    // The value of this node, heap-allocated in order to be shared
    // across buckets instances during resize operations.
    value: NonNull<V>,
    // The next node in the linked-list.
    next: Atomic<Self>,
}

impl<K, V, S> RawTable<K, V, S> {
    // Returns a new `RawTable` with the specified hasher.
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            resize: Arc::new(Mutex::new(())),
            budget: AtomicUsize::new(0),
            table: Atomic::null(),
            collector: Collector::new(),
        }
    }

    // Returns a new `RawTable` with the specified capacity and hasher.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> RawTable<K, V, S> {
        if capacity == 0 {
            return Self::with_hasher(hash_builder);
        }

        let lock_count = resize::initial_locks(capacity);

        Self {
            resize: Arc::new(Mutex::new(())),
            budget: AtomicUsize::new(resize::budget(capacity, lock_count)),
            table: Atomic::new(Inner::new(capacity, lock_count)),
            collector: Collector::new(),
            hash_builder,
        }
    }
}

impl<K, V, S> RawTable<K, V, S> {
    // Returns the number of elements in the map.
    pub fn len(&self, guard: &Guard<'_>) -> usize {
        unsafe { self.table.load(Ordering::Acquire, guard).as_ref() }
            .map(Inner::len)
            .unwrap_or_default()
    }

    // Returns `true` if the map contains no elements.
    pub fn is_empty(&self, guard: &Guard<'_>) -> bool {
        self.len(guard) == 0
    }

    // An iterator visiting all key-value pairs in arbitrary order.
    pub fn iter<'a>(&'a self, guard: &'a Guard<'a>) -> Iter<'a, K, V> {
        let table = self.table.load(Ordering::Acquire, guard);
        let first_node = unsafe { table.as_ref().and_then(|t| t.buckets.get(0)) };

        Iter {
            size: self.len(guard),
            current_node: first_node,
            table,
            current_bucket: 0,
            guard,
        }
    }

    // Initializes the table if it has not been already.
    fn init_table<'a>(&'a self, capacity: usize, guard: &'a Guard<'a>) -> Shared<'a, Inner<K, V>> {
        let _resizing = self.resize.lock();
        let table = self.table.load(Ordering::Acquire, guard);

        if !table.is_null() {
            return table;
        }

        let table =
            Owned::new(Inner::new(capacity, resize::initial_locks(capacity))).into_shared(guard);
        self.table.store(table, Ordering::Release);
        table
    }
}

impl<K, V, S> RawTable<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    // Returns a reference to the value corresponding to the key.
    pub fn get<'a, Q: ?Sized>(&'a self, key: &Q, guard: &'a Guard<'a>) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(key);
        let table = unsafe { self.table.load(Ordering::Acquire, guard).as_ref()? };

        for (_, _, node) in table.bucket(hash, guard) {
            if node.hash == hash && node.key.borrow() == key {
                return Some(unsafe { node.value.as_ref() });
            }
        }

        None
    }

    // Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q: ?Sized>(&self, key: &Q, guard: &Guard<'_>) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(key, &guard).is_some()
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

// methods that required K/V: Send + Sync, as they potentially call
// `defer_destroy` on keys and values, and the garbage collector
// may execute the destructors on another thread
impl<K, V, S> RawTable<K, V, S>
where
    V: Send + Sync,
    K: Hash + Eq + Send + Sync + Clone,
    S: BuildHasher,
{
    // Inserts a key-value pair into the map.
    pub fn insert<'a>(&'a self, key: K, value: V, guard: &'a Guard<'a>) -> Option<&'a V> {
        let hash = self.hash(&key);
        let mut should_resize = false;

        loop {
            let mut table_ptr = unsafe { self.table.load(Ordering::Acquire, guard) };
            if table_ptr.is_null() {
                table_ptr = self.init_table(resize::DEFAULT_BUCKETS, guard);
            }

            let table = unsafe { table_ptr.deref() };
            let bucket = bucket_index(hash, table.buckets.len() as _);
            let lock = lock_index(bucket, table.locks.len() as _) as usize;

            {
                let _guard = table.locks[lock].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !ptr::eq(
                    table_ptr.as_ptr(),
                    self.table.load(Ordering::Acquire, guard).as_ptr(),
                ) {
                    continue;
                }

                // Try to find this key in the bucket
                for (atomic, ptr, node) in table.bucket(hash, guard) {
                    if node.hash == hash && node.key == key {
                        // If the key already exists, update the value.
                        let new = Node {
                            key,
                            next: node.next.copy(Ordering::Relaxed),
                            value: NonNull::from(Box::leak(Box::new(value))),
                            hash,
                        };
                        atomic.store(Owned::new(new).into_shared(guard), Ordering::Release);

                        guard.retire(move || unsafe {
                            let node = ptr.into_owned();
                            let _ = Box::from_raw(node.value.as_ptr());
                        });

                        return Some(unsafe { node.value.as_ref() });
                    }
                }

                // Otherwise insert a new one.
                let node = &table.buckets[bucket as usize];
                let new = Node {
                    key,
                    next: node.copy(Ordering::Relaxed),
                    value: NonNull::from(Box::leak(Box::new(value))),
                    hash,
                };

                node.store(Owned::new(new).into_shared(guard), Ordering::Release);

                // If the number of elements guarded by this lock has exceeded the budget, resize
                // the hash table.
                //
                // It is also possible that `resize` will increase the budget instead of resizing the
                // hashtable if keys are poorly distributed across buckets.
                let count = unsafe { table.counts[lock].fetch_add(1, Ordering::AcqRel) + 1 };
                if count > self.budget.load(Ordering::SeqCst) {
                    should_resize = true;
                }
            }

            // Resize the table if we're passed our budget.
            //
            // Note that we are not holding the lock anymore.
            if should_resize {
                unsafe { self.resize(table_ptr, None, guard) };
            }

            return None;
        }
    }

    // Removes a key from the map, returning the value at the key if the key
    // was previously in the map.
    pub fn remove<'a, Q: ?Sized>(&'a self, key: &Q, guard: &'a Guard<'a>) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(key);

        loop {
            let table = unsafe { self.table.load(Ordering::Acquire, guard) };
            let table_ref = unsafe { table.as_ref() }?;

            let bucket_index = bucket_index(hash, table_ref.buckets.len() as _);
            let lock_index = lock_index(bucket_index, table_ref.locks.len() as _);

            {
                let _guard = table_ref.locks[lock_index as usize].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !ptr::eq(
                    table.as_ptr(),
                    self.table.load(Ordering::Acquire, guard).as_ptr(),
                ) {
                    continue;
                }

                let mut walk = table_ref.buckets.get(bucket_index as usize);
                while let Some(node_ptr) = walk {
                    let node = unsafe { node_ptr.load(Ordering::Acquire, guard) };
                    let node_ref = match unsafe { node.as_ref() } {
                        Some(x) => x,
                        None => break,
                    };

                    if hash == node_ref.hash && node_ref.key.borrow() == key {
                        let next = node_ref.next.load(Ordering::Acquire, guard);
                        node_ptr.store(next, Ordering::Release);

                        unsafe {
                            table_ref.counts[lock_index as usize].fetch_min(1, Ordering::AcqRel)
                        };

                        let val = unsafe { node_ref.value.as_ref() };

                        guard.retire(move || unsafe {
                            let node = node.into_owned();
                            let _ = Box::from_raw(node.value.as_ptr());
                        });

                        return Some(val);
                    }

                    walk = Some(&node_ref.next);
                }
            }

            return None;
        }
    }

    // Clears the map, removing all key-value pairs.
    pub fn clear(&self, guard: &Guard<'_>) {
        let _resizing = self.resize.lock();

        // Now that we have the resize lock, we can safely load the table and acquire it's
        // locks, knowing that it cannot change.
        let table = self.table.load(Ordering::Acquire, guard);
        let table_ref = match unsafe { table.as_ref() } {
            Some(table) => table,
            None => return,
        };

        let _buckets = table_ref.lock_buckets();

        // walk through the table buckets, and set each initial node to null, destroying the rest
        // of the nodes and values.
        for bucket_ptr in table_ref.buckets.iter() {
            match unsafe { bucket_ptr.load(Ordering::Acquire, guard).as_ref() } {
                Some(bucket) => {
                    bucket_ptr.store(Shared::null(), Ordering::Release);

                    guard.retire(move || unsafe {
                        let _ = Box::from_raw(bucket.value.as_ptr());
                    });

                    let mut node = bucket.next.load(Ordering::Relaxed, guard);
                    while !node.is_null() {
                        guard.retire(move || unsafe {
                            let node = node.into_owned();
                            let _ = Box::from_raw(node.value.as_ptr());
                        });

                        node = unsafe { node.deref().next.load(Ordering::Relaxed, guard) };
                    }
                }
                None => break,
            }
        }

        // reset the counts
        for i in 0..table_ref.counts.len() {
            table_ref.counts[i].store(0, Ordering::Release);
        }

        if let Some(budget) = resize::clear_budget(table_ref.buckets.len(), table_ref.locks.len()) {
            // store the new budget, which might be different if the map was
            // badly distributed
            self.budget.store(budget, Ordering::SeqCst);
        }
    }

    // Reserves capacity for at least additional more elements to be inserted in the `HashMap`.
    pub fn reserve<'a>(&'a self, additional: usize, guard: &'a Guard<'a>) {
        let table = self.table.load(Ordering::Acquire, guard);

        if table.is_null() {
            self.init_table(additional, guard);
            return;
        }

        let capacity = unsafe { table.deref() }.len() + additional;
        unsafe { self.resize(table, Some(capacity), guard) };
    }

    // Replaces the hash-table with a larger one.
    //
    // To prevent multiple threads from resizing the table as a result of races,
    // the `table` instance that holds the buckets of table deemed too small must
    // be passed as an argument. We then acquire the resize lock and check if the
    // table has been replaced in the meantime or not.
    //
    // This function may not resize the table if it values are found to be poorly
    // distributed across buckets. Instead the budget will be increased.
    //
    // If a `new_len` is given, that length will be used instead of the default
    // resizing behavior.
    unsafe fn resize<'a>(
        &'a self,
        table: Shared<'a, Inner<K, V>>,
        new_len: Option<usize>,
        guard: &'a Guard<'a>,
    ) {
        let _resizing = self.resize.lock();

        // Make sure nobody resized the table while we were waiting for the resize lock.
        if !ptr::eq(
            table.as_ptr(),
            self.table.load(Ordering::Acquire, guard).as_ptr(),
        ) {
            // We assume that since the table reference is different, it was
            // already resized (or the budget was adjusted). If we ever decide
            // to do table shrinking, or replace the table for other reasons,
            // we will have to revisit this logic.
            return;
        }

        let table_ref = unsafe { table.deref() };
        let current_locks = table_ref.locks.len();
        let (new_len, new_locks, new_budget) = match new_len {
            Some(len) => {
                let locks = resize::new_locks(len, current_locks);
                let budget = resize::budget(len, locks.unwrap_or(current_locks));
                (len, locks, budget)
            }
            None => match resize::resize(table_ref.buckets.len(), current_locks, table_ref.len()) {
                resize::Resize::Resize {
                    buckets,
                    locks,
                    budget,
                } => (buckets, locks, budget),
                resize::Resize::DoubleBudget => {
                    let budget = self.budget.load(Ordering::SeqCst);
                    self.budget.store(budget * 2, Ordering::SeqCst);
                    return;
                }
            },
        };

        // Add more locks to account for the new buckets.
        let new_locks = new_locks
            .map(|len| {
                table_ref
                    .locks
                    .iter()
                    .cloned()
                    .chain((current_locks..len).map(|_| Arc::default()))
                    .collect()
            })
            .unwrap_or_else(|| table_ref.locks.clone());

        let new_counts = std::iter::repeat_with(AtomicUsize::default)
            .take(new_locks.len())
            .collect::<Vec<_>>();

        let mut new_buckets = std::iter::repeat_with(Atomic::null)
            .take(new_len)
            .collect::<Vec<_>>();

        // Now lock the rest of the buckets to make sure nothing is modified while resizing.
        let _buckets = table_ref.lock_buckets();

        // Copy all data into a new buckets, creating new nodes for all elements.
        for bucket in table_ref.buckets.iter() {
            let mut node = bucket.load(Ordering::Relaxed, guard);
            while !node.is_null() {
                let node_ref = unsafe { node.deref() };
                let new_bucket_index = bucket_index(node_ref.hash, new_len as _);
                let new_lock_index = lock_index(new_bucket_index, new_locks.len() as _);

                let next = node_ref.next.load(Ordering::Relaxed, guard);
                let head = new_buckets[new_bucket_index as usize].load(Ordering::Relaxed, guard);

                // if `next` is the same as the new head, we can skip an allocation and just copy
                // the pointer
                if ptr::eq(next.as_ptr(), head.as_ptr()) {
                    new_buckets[new_bucket_index as usize] = Atomic::from_shared(node);
                } else {
                    new_buckets[new_bucket_index as usize] = Atomic::new(Node {
                        key: node_ref.key.clone(),
                        value: node_ref.value,
                        next: new_buckets[new_bucket_index as usize].copy(Ordering::Relaxed),
                        hash: node_ref.hash,
                    });

                    guard.retire(move || unsafe {
                        let _ = node.into_owned();
                    });
                }

                new_counts[new_lock_index as usize].fetch_add(1, Ordering::Relaxed);

                node = next;
            }
        }

        self.budget.store(new_budget, Ordering::SeqCst);

        self.table.store(
            Owned::new(Inner {
                buckets: new_buckets.into_boxed_slice(),
                locks: new_locks,
                counts: new_counts.into_boxed_slice(),
            })
            .into_shared(guard),
            Ordering::Release,
        );

        guard.retire(move || unsafe {
            let _ = table.into_owned();
        });
    }
}

// Computes the bucket index for a particular key.
fn bucket_index(hashcode: u64, bucket_count: u64) -> u64 {
    let bucket_index = (hashcode & 0x7fffffff) % bucket_count;
    debug_assert!(bucket_index < bucket_count);

    bucket_index
}

// Computes the lock index for a particular bucket.
fn lock_index(bucket_index: u64, lock_count: u64) -> u64 {
    let lock_index = bucket_index % lock_count;
    debug_assert!(lock_index < lock_count);

    lock_index
}

impl<K, V, S> Drop for RawTable<K, V, S> {
    fn drop(&mut self) {
        let table = unsafe { self.table.load_unprotected(Ordering::Relaxed) };

        // table was never allocated
        if table.is_null() {
            return;
        }

        let mut table = unsafe { table.into_owned() };

        // drop all nodes and values
        for bucket in table.buckets.iter_mut() {
            let mut node = unsafe { bucket.load_unprotected(Ordering::Relaxed) };
            while !node.is_null() {
                let owned = unsafe { node.into_owned() };
                let _value = unsafe { Box::from_raw(owned.value.as_ptr()) };
                node = unsafe { owned.next.load_unprotected(Ordering::Relaxed) };
            }
        }
    }
}

unsafe impl<K, V, S> Send for RawTable<K, V, S>
where
    K: Send + Sync,
    V: Send + Sync,
    S: Send,
{
}

unsafe impl<K, V, S> Sync for RawTable<K, V, S>
where
    K: Send + Sync,
    V: Send + Sync,
    S: Sync,
{
}

impl<K, V, S> Clone for RawTable<K, V, S>
where
    K: Sync + Send + Clone + Hash + Eq,
    V: Sync + Send + Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> RawTable<K, V, S> {
        let guard = self.collector.guard();

        let clone = Self::with_capacity_and_hasher(self.len(&guard), self.hash_builder.clone());
        {
            for (k, v) in clone.iter(&guard) {
                clone.insert(k.clone(), v.clone(), &guard);
            }
        }

        clone
    }
}

impl<K, V, S> fmt::Debug for RawTable<K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = self.collector.guard();
        f.debug_map().entries(self.iter(&guard)).finish()
    }
}

impl<K, V, S> Default for RawTable<K, V, S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> RawTable<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    pub fn eq(&self, guard: &Guard<'_>, other: &Self, other_guard: &Guard<'_>) -> bool {
        if self.len(guard) != other.len(other_guard) {
            return false;
        }

        self.iter(guard)
            .all(|(key, value)| other.get(key, other_guard).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S> PartialEq for RawTable<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        let guard = self.collector.guard();
        let other_guard = other.collector.guard();

        Self::eq(self, &guard, other, &other_guard)
    }
}

impl<K, V, S> Eq for RawTable<K, V, S>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{
}

// An iterator over the entries of a `HashMap`.
pub struct Iter<'a, K, V> {
    table: Shared<'a, Inner<K, V>>,
    guard: &'a Guard<'a>,
    current_node: Option<&'a Atomic<Node<K, V>>>,
    current_bucket: usize,
    size: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.table.is_null() {
                return None;
            }

            let table = unsafe { self.table.deref() };

            if let Some(node) = &self
                .current_node
                .and_then(|n| unsafe { n.load(Ordering::Acquire, self.guard).as_ref() })
            {
                self.size -= 1;
                self.current_node = Some(&node.next);
                return Some((&node.key, unsafe { node.value.as_ref() }));
            }

            self.current_bucket += 1;

            if self.current_bucket >= table.buckets.len() {
                return None;
            }

            self.current_node = Some(&table.buckets[self.current_bucket]);
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<K, V> fmt::Debug for Iter<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.clone()).finish()
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
