/// The hash-table will be resized to this amount on the first insert
/// unless a non-zero capacity is specified upon creation.
pub const DEFAULT_BUCKETS: usize = 31;

/// The maximum # of buckets the buckets can hold.
const MAX_BUCKETS: usize = isize::MAX as _;

/// The maximum size of the `locks` array.
/// TODO max this out like dashmap
const MAX_LOCKS: usize = 1024;

pub enum Resize {
    /// The table is poorly distributed, double the budget instead of resizing.
    DoubleBudget,
    /// Resize the table.
    Resize {
        /// The new number of buckets.
        buckets: usize,
        /// The new number of locks, `None` if no locks should be added.
        locks: Option<usize>,
        /// The new budget.
        budget: usize,
    },
}

pub fn resize(current_buckets: usize, current_locks: usize, current_len: usize) -> Resize {
    // If the bucket array is too empty, double the budget instead of resizing the table
    if current_len < current_buckets / 4 {
        return Resize::DoubleBudget;
    }

    // Compute the new buckets size.
    //
    // The current method is to find the smallest integer that is:
    //
    // 1) larger than twice the previous buckets size
    // 2) not divisible by 2, 3, 5 or 7.
    //
    // This may change in the future.
    let compute_new_len = || {
        // Double the size of the buckets buckets and add one, so that we have an odd integer.
        let mut new_len = current_buckets.checked_mul(2)?.checked_add(1)?;

        // Now, we only need to check odd integers, and find the first that is not divisible
        // by 3, 5 or 7.
        while new_len.checked_rem(3)? == 0
            || new_len.checked_rem(5)? == 0
            || new_len.checked_rem(7)? == 0
        {
            new_len = new_len.checked_add(2)?;
        }

        debug_assert!(new_len % 2 != 0);

        Some(new_len)
    };

    let len = match compute_new_len() {
        Some(len) => len,
        None => MAX_BUCKETS,
    };

    let buckets = len.min(MAX_BUCKETS);
    let locks = new_locks(buckets, current_locks);
    let budget = budget(buckets, locks.unwrap_or(current_locks));

    Resize::Resize {
        buckets,
        locks,
        budget,
    }
}

pub fn clear_budget(buckets: usize, locks: usize) -> Option<usize> {
    // the capacity doesn't change when clearing, so if
    // we're already at max, this is a no-op.
    if buckets == MAX_BUCKETS {
        return None;
    }

    Some(1.max(buckets / locks))
}

pub fn budget(buckets: usize, locks: usize) -> usize {
    // We are at `MAX_BUCKETS`, make sure `resize` is never called again.
    if buckets == MAX_BUCKETS {
        return usize::MAX;
    }

    1.max(buckets / locks)
}

pub fn initial_locks(buckets: usize) -> usize {
    new_locks(buckets, 0).unwrap()
}

pub fn new_locks(buckets: usize, current_locks: usize) -> Option<usize> {
    (current_locks != MAX_LOCKS).then(|| match buckets {
        0..=67 => 8,
        68..=137 => 16,
        138..=277 => 32,
        278..=557 => 64,
        558..=1117 => 128,
        1118..=2237 => 512,
        _ => 1024,
    })
}
