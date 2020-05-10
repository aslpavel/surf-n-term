use std::{
    collections::VecDeque,
    io::{BufRead, Read, Write},
};

pub struct IOQueue {
    chunks: VecDeque<Vec<u8>>,
    offset: usize,
}

impl IOQueue {
    pub fn new() -> Self {
        Self {
            chunks: Default::default(),
            offset: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
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
            self.offset += amt
        } else {
            self.offset = 0;
            self.chunks.pop_front();
        }
    }

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

#[cfg(test)]
mod tests {
    use super::IOQueue;
    use std::io::{BufRead, Read, Write};

    #[test]
    fn test_io_queue() -> std::io::Result<()> {
        let mut queue = IOQueue::new();
        assert!(queue.is_empty());

        queue.write(b"on")?;
        queue.write(b"e")?;
        queue.flush()?;
        queue.write(b",two")?;
        queue.flush()?;
        assert!(!queue.is_empty());

        // make sure flush creates separate chunks
        assert_eq!(queue.as_slice(), b"one");

        // check `Read` implementation
        let mut out = String::new();
        queue.read_to_string(&mut out)?;
        assert_eq!(out, String::from("one,two"));
        assert!(queue.is_empty());

        // make `BufRead` implementation
        queue.write(b"one\nt")?;
        queue.flush()?;
        queue.write(b"wo\nth")?;
        queue.flush()?;
        queue.write(b"ree\n")?;
        let lines = queue.lines().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            lines,
            vec!["one".to_string(), "two".to_string(), "three".to_string()]
        );
        Ok(())
    }
}
