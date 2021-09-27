use crate::{HashMap, Iter, Keys, Values};

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::ops::Index;

use crossbeam_epoch::Guard;

/// A reference to a [`HashMap`] pinned to the current thread.
///
/// The only way to access a map is through a pinned reference.
/// Dropping a pinned reference will trigger garbage collection.
pub struct Pinned<'map, K, V, S = RandomState> {
    pub(crate) map: &'map HashMap<K, V, S>,
    pub(crate) guard: Guard,
}

impl<'map, K, V, S> Pinned<'map, K, V, S>
where
    V: Send + Sync,
    K: Hash + Eq + Send + Sync + Clone,
    S: BuildHasher,
{
    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let map = HashMap::new();
    /// let pinned = map.pin();
    ///
    /// pinned.insert(1, "a");
    /// assert_eq!(pinned.get(&1), Some(&"a"));
    /// assert_eq!(pinned.get(&2), None);
    /// ```
    pub fn get<'p, Q: ?Sized>(&'p self, key: &Q) -> Option<&'p V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.get(key, &self.guard)
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.map.iter(&self.guard)
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    pub fn keys(&self) -> Keys<'_, K, V> {
        self.map.keys(&self.guard)
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values(&self) -> Values<'_, K, V> {
        self.map.values(&self.guard)
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.contains_key(key, &self.guard)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let map = HashMap::new();
    /// let pinned = map.pin();
    ///
    /// assert_eq!(pinned.insert(37, "a"), None);
    /// assert_eq!(pinned.is_empty(), false);
    ///
    /// pinned.insert(37, "b");
    /// assert_eq!(pinned.insert(37, "c"), Some("b"));
    /// assert_eq!(pinned[&37], "c");
    /// ```
    pub fn insert(&self, key: K, value: V) -> Option<&'_ V> {
        self.map.insert(key, value, &self.guard)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let map = HashMap::new();
    /// let pinned = map.pin();
    ///
    /// pinned.insert(1, "a");
    /// assert_eq!(pinned.remove(&1), Some(&"a"));
    /// assert_eq!(pinned.remove(&1), None);
    /// ```
    pub fn remove<'p, Q: ?Sized>(&'p self, key: &Q) -> Option<&'p V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.remove(key, &self.guard)
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
    pub fn clear(&self) {
        self.map.clear(&self.guard)
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let a = HashMap::new();
    /// let pinned = a.pin();
    ///
    /// assert_eq!(pinned.len(), 0);
    /// pinned.insert(1, "a");
    /// assert_eq!(pinned.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.map.len(&self.guard)
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use concurrent_dictionary::HashMap;
    ///
    /// let a = HashMap::new();
    /// let pinned = a.pin();
    ///
    /// assert!(pinned.is_empty());
    /// pinned.insert(1, "a");
    /// assert!(!pinned.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.map.is_empty(&self.guard)
    }
}

impl<K, V, S> Clone for Pinned<'_, K, V, S> {
    fn clone(&self) -> Self {
        self.map.pin()
    }
}

impl<K, Q, V, S> Index<&'_ Q> for Pinned<'_, K, V, S>
where
    K: Borrow<Q> + Hash + Eq + Send + Sync + Clone,
    V: Send + Sync,
    Q: Hash + Ord + ?Sized,
    S: BuildHasher,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}
