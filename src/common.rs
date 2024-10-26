//! Common utility functions used across different modules
use crate::RGBA;
use std::{
    collections::{HashMap, VecDeque},
    io::{BufRead, Read, Write},
    str::FromStr,
};

/// Clamp value to the value between `min` and `max`
#[inline]
pub fn clamp<T>(val: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

pub trait LockExt {
    type Value;

    fn with<Scope, Out>(&self, scope: Scope) -> Out
    where
        Scope: FnOnce(&Self::Value) -> Out;

    fn with_mut<Scope, Out>(&self, scope: Scope) -> Out
    where
        Scope: FnOnce(&mut Self::Value) -> Out;
}

impl<V> LockExt for std::sync::Mutex<V> {
    type Value = V;

    fn with<Scope, Out>(&self, scope: Scope) -> Out
    where
        Scope: FnOnce(&Self::Value) -> Out,
    {
        let value = self.lock().expect("lock poisoned");
        scope(&*value)
    }

    fn with_mut<Scope, Out>(&self, scope: Scope) -> Out
    where
        Scope: FnOnce(&mut Self::Value) -> Out,
    {
        let mut value = self.lock().expect("lock poisoned");
        scope(&mut *value)
    }
}

impl<V> LockExt for std::sync::RwLock<V> {
    type Value = V;

    fn with<Scope, Out>(&self, scope: Scope) -> Out
    where
        Scope: FnOnce(&Self::Value) -> Out,
    {
        let value = self.read().expect("lock poisoned");
        scope(&*value)
    }

    fn with_mut<Scope, Out>(&self, scope: Scope) -> Out
    where
        Scope: FnOnce(&mut Self::Value) -> Out,
    {
        let mut value = self.write().expect("lock poisoned");
        scope(&mut *value)
    }
}

lazy_static::lazy_static! {
    static ref ENV_CONFIG: HashMap<String, String> = {
        let mut config = HashMap::new();
        let config_str = match std::env::var("SURFNTERM") {
            Ok(config_str) => config_str,
            _ => return config,
        };
        for kv in config_str.split(',') {
            let mut kv = kv.trim().splitn(2, '=');
            if let Some(key) = kv.next() {
                config.insert(key.trim().to_string(), kv.next().unwrap_or("").trim().to_string());
            }
        }
        config
    };
}

pub fn env_cfg<V: FromStr>(key: &str) -> Option<V> {
    let value = ENV_CONFIG.get(key)?;
    value.parse().ok()
}

/// Readable and writable IO queue
pub struct IOQueue {
    chunks: VecDeque<Vec<u8>>,
    offset: usize,
    length: usize,
}

impl IOQueue {
    /// Create new empty queue
    pub fn new() -> Self {
        Self {
            chunks: Default::default(),
            offset: 0,
            length: 0,
        }
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Number of bytes available to be read from the queue
    pub fn len(&self) -> usize {
        self.length
    }

    /// Drop all but last chunks
    pub fn clear_but_last(&mut self) {
        if self.chunks.len() > 1 {
            self.chunks.drain(1..);
        }
    }

    /// Number of available chunks
    pub fn chunks_count(&self) -> usize {
        self.chunks.len()
    }

    /// Remaining slice at the front of the queue
    pub fn as_slice(&self) -> &[u8] {
        match self.chunks.front() {
            Some(chunk) => &chunk.as_slice()[self.offset..],
            None => &[],
        }
    }

    /// Consume bytes from the front of the queue
    pub fn consume(&mut self, amt: usize) {
        if self.chunks.front().map(|chunk| chunk.len()).unwrap_or(0) > self.offset + amt {
            self.offset += amt;
            self.length -= amt;
        } else {
            if let Some(chunk) = self.chunks.pop_front() {
                self.length -= chunk.len() - self.offset
            }
            self.offset = 0;
        }
    }

    /// Consume single chunk from the queue. Function `consumer` returns number of bytes
    /// to be removed from the queue.
    pub fn consume_with<F, FE>(&mut self, consumer: F) -> Result<usize, FE>
    where
        F: FnOnce(&[u8]) -> Result<usize, FE>,
    {
        let size = consumer(self.as_slice())?;
        self.consume(size);
        Ok(size)
    }
}

impl Write for IOQueue {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.chunks.is_empty() {
            self.chunks.push_back(Default::default());
        }
        let chunk = self.chunks.back_mut().unwrap();
        self.length += buf.len();
        chunk.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.as_slice().is_empty() {
            self.chunks.push_back(Default::default());
        }
        Ok(())
    }
}

impl Read for IOQueue {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let buf_queue = self.as_slice();
        let size = std::cmp::min(buf.len(), buf_queue.len());
        let src = &buf_queue[..size];
        let dst = &mut buf[..size];
        dst.copy_from_slice(src);
        self.consume(size);
        Ok(size)
    }
}

impl BufRead for IOQueue {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        Ok(self.as_slice())
    }

    fn consume(&mut self, amt: usize) {
        Self::consume(self, amt)
    }
}

impl Default for IOQueue {
    fn default() -> Self {
        IOQueue::new()
    }
}

/// Simple random number generator
///
/// Implemented as linear congruential generator LCG. Used here instead
/// of pooling in `rand` crate to reduce dependencies.
/// Formula:
///     X(n) = ((X(n - 1) * a + c) % m) / d
///     a = 214_013
///     c = 2_531_011
///     m = 2 ^ 31
///     d = 2 ^ 16
/// This formula produces 16-bit random number.
pub struct Rnd {
    state: u32,
}

impl Default for Rnd {
    fn default() -> Self {
        Self::new()
    }
}

impl Rnd {
    /// Create new random number generator with seed `0`
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Create new random number generator with provided `seed` value
    pub fn with_seed(seed: u32) -> Self {
        Self { state: seed }
    }

    fn step(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(214_013).wrapping_add(2_531_011) & 0x7fff_ffff;
        self.state >> 16
    }

    /// Generate random u32 value
    pub fn next_u32(&mut self) -> u32 {
        (self.step() & 0xffff) << 16 | (self.step() & 0xffff)
    }

    pub fn next_u8x4(&mut self) -> [u8; 4] {
        let value = self.next_u32();
        [
            (value & 0xff) as u8,
            ((value >> 8) & 0xff) as u8,
            ((value >> 16) & 0xff) as u8,
            ((value >> 24) & 0xff) as u8,
        ]
    }

    /// Generate random u64 value
    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }
}

pub trait Random: Sized {
    fn random(rnd: &mut Rnd) -> Self;

    fn random_iter() -> Box<dyn Iterator<Item = Self>> {
        let mut rnd = Rnd::new();
        Box::new(std::iter::from_fn(move || Some(Self::random(&mut rnd))))
    }
}

impl Random for RGBA {
    fn random(rnd: &mut Rnd) -> Self {
        let [r, g, b, _] = rnd.next_u8x4();
        RGBA::new(r, g, b, 255)
    }
}

#[cfg(test)]
mod tests {
    use super::IOQueue;
    use std::io::{BufRead, Read, Write};

    #[test]
    fn test_io_queue() -> std::io::Result<()> {
        let mut queue = IOQueue::new();
        assert!(queue.is_empty());

        queue.write_all(b"on")?;
        queue.write_all(b"e")?;
        queue.flush()?;
        queue.write_all(b",two")?;
        queue.flush()?;
        assert!(!queue.is_empty());

        // make sure flush creates separate chunks
        assert_eq!(queue.as_slice(), b"one");

        // check `Read` implementation
        let mut out = String::new();
        queue.read_to_string(&mut out)?;
        assert_eq!(out, String::from("one,two"));
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());

        // make `BufRead` implementation
        queue.write_all(b"one\nt")?;
        queue.flush()?;
        queue.write_all(b"wo\nth")?;
        queue.flush()?;
        queue.write_all(b"ree\n")?;
        let lines = queue.lines().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            lines,
            vec!["one".to_string(), "two".to_string(), "three".to_string()]
        );
        Ok(())
    }
}
