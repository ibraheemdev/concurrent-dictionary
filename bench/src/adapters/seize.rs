use std::hash::{BuildHasher, Hash};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

use bustle::*;

struct Value(AtomicU64);

impl From<u64> for Value {
    fn from(val: u64) -> Self {
        Value(AtomicU64::new(val))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.0.load(Relaxed) == other.0.load(Relaxed)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.load(Relaxed))
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(AtomicU64::new(self.0.load(Relaxed)))
    }
}

#[derive(Clone)]
pub struct SeizeTable<K, H>(Arc<seize::HashMap<K, Value, H>>);

pub struct SeizeHandle<K: 'static, H: 'static>(seize::Pinned<'static, K, Value, H>);

impl<H> Collection for SeizeTable<u64, H>
where
    H: BuildHasher + Default + Send + Sync + 'static + Clone,
{
    type Handle = SeizeHandle<u64, H>;

    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(seize::HashMap::with_capacity_and_hasher(
            capacity,
            H::default(),
        )))
    }

    fn pin(&self) -> Self::Handle {
        SeizeHandle(unsafe { std::mem::transmute(self.0.pin()) })
    }
}

impl<H> CollectionHandle for SeizeHandle<u64, H>
where
    H: BuildHasher + Default + Send + Sync + 'static + Clone,
{
    type Key = u64;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.insert(*key, Value(AtomicU64::new(0))).is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).map(|x| x.0.fetch_add(1, Relaxed)).is_some()
    }
}

pub struct ShardedSeizeTable<K, H>(Arc<seize::Sharded<K, Value, H>>);

pub struct ShardedSeizeHandle<K: 'static, H: 'static>(
    &'static seize::Sharded<K, Value, H>,
    seize::Guard,
);

impl<H> Collection for ShardedSeizeTable<u64, H>
where
    H: BuildHasher + Default + Send + Sync + 'static + Clone,
{
    type Handle = ShardedSeizeHandle<u64, H>;

    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(seize::Sharded::with_capacity_and_hasher(
            capacity,
            H::default(),
        )))
    }

    fn pin(&self) -> Self::Handle {
        let map: &'static seize::Sharded<_, _, _> = unsafe { std::mem::transmute(&*self.0) };
        ShardedSeizeHandle(map, self.0.pin())
    }
}

impl<H> CollectionHandle for ShardedSeizeHandle<u64, H>
where
    H: BuildHasher + Default + Send + Sync + 'static + Clone,
{
    type Key = u64;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key, &self.1).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0
            .insert(*key, Value(AtomicU64::new(0)), &self.1)
            .is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.remove(key, &self.1).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        self.0
            .get(key, &self.1)
            .map(|x| x.0.fetch_add(1, Relaxed))
            .is_some()
    }
}
