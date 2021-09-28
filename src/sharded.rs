use std::{
    borrow::Borrow,
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hash, Hasher},
};

use crossbeam_epoch::{Collector, Guard};

use crate::HashMap;

pub struct Sharded<K, V, S = RandomState> {
    shift: usize,
    shards: Box<[HashMap<K, V, S>]>,
    hasher: S,
    collector: Collector,
}

fn ncb(shard_amount: usize) -> usize {
    shard_amount.trailing_zeros() as usize
}

pub fn ptr_size_bits() -> usize {
    std::mem::size_of::<usize>() * 8
}

impl<K, V, S> Sharded<K, V, S>
where
    S: Clone + BuildHasher,
    K: Hash + Eq + Send + Sync + Clone,
    V: Send + Sync,
{
    pub fn pin(&self) -> Guard {
        self.collector.register().pin()
    }

    pub fn with_capacity_and_hasher(mut capacity: usize, hasher: S) -> Self {
        let shard_amount = (num_cpus::get() * 4).next_power_of_two();
        let shift = ptr_size_bits() - ncb(shard_amount);

        if capacity != 0 {
            capacity = (capacity + (shard_amount - 1)) & !(shard_amount - 1);
        }

        let cps = capacity / shard_amount;

        let collector = Collector::new();

        let shards = (0..shard_amount)
            .map(|_| HashMap::build(cps, hasher.clone(), collector.clone()))
            .collect();

        Self {
            collector,
            shift,
            shards,
            hasher,
        }
    }

    pub fn get<'g, Q>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash(&key);
        let idx = self.determine_shard(hash);
        self.shards[idx].get_hashed(key, guard, hash)
    }

    pub fn remove<'g, Q: ?Sized>(&'g self, key: &Q, guard: &'g Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(&key);
        let idx = self.determine_shard(hash);
        self.shards[idx].remove_hashed(key, guard, hash)
    }

    pub fn insert<'g>(&'g self, key: K, value: V, guard: &'g Guard) -> Option<&'g V> {
        let hash = self.hash(&key);
        let idx = self.determine_shard(hash);
        self.shards[idx].insert_hashed(key, value, guard, hash)
    }

    pub fn determine_shard(&self, hash: u64) -> usize {
        // Leave the high 7 bits for the HashBrown SIMD tag.
        ((hash as usize) << 7) >> self.shift
    }

    pub fn hash<T: Hash>(&self, item: &T) -> u64 {
        let mut hasher = self.hasher.build_hasher();
        item.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let map =
            Sharded::with_capacity_and_hasher(0, std::collections::hash_map::RandomState::new());
        {
            let guard = map.pin();
            map.insert(1, "a", &guard);
            map.insert(2, "b", &guard);
            assert_eq!(map.get(&1, &guard), Some(&"a"));
            assert_eq!(map.get(&2, &guard), Some(&"b"));
            assert_eq!(map.get(&3, &guard), None);

            assert_eq!(map.remove(&2, &guard), Some(&"b"));
            assert_eq!(map.remove(&2, &guard), None);
        }

        let guard = map.pin();
        for i in 0..100 {
            map.insert(i, "a", &guard);
        }

        for i in 0..100 {
            assert_eq!(map.get(&i, &guard), Some(&"a"));
        }
    }
}
