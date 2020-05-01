use crate::cell::Cell;
use std::ops::{Bound, RangeBounds};

#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub offset: usize,
    pub row_stride: usize,
    pub col_stride: usize,
    pub height: usize,
    pub width: usize,
}

impl Shape {
    pub fn index(&self, row: usize, col: usize) -> usize {
        self.offset + row * self.row_stride + col * self.col_stride
    }
}

#[derive(Clone)]
pub struct Surface {
    shape: Shape,
    data: Vec<Cell>,
}

pub struct SurfaceView<'a> {
    shape: Shape,
    data: &'a mut [Cell],
}

impl Surface {
    pub fn new(height: usize, width: usize) -> Self {
        let mut data = Vec::new();
        data.resize_with(height * width, Default::default);
        Self {
            shape: Shape {
                offset: 0,
                row_stride: width,
                col_stride: 1,
                height,
                width,
            },
            data,
        }
    }
}

/// Resolve bounds the same way python or numpy does for ranges when indexing arrays
fn resolve_bound(bound: impl RangeBounds<i32>, size: usize) -> Option<(usize, usize)> {
    if size == 0 {
        return None;
    }

    let start = match bound.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(start) => *start,
        Bound::Excluded(-1) => return None,
        Bound::Excluded(start) => *start + 1,
    };
    let start = if start.abs() > size as i32 {
        return None;
    } else if start >= 0 {
        start as usize
    } else {
        size - start.abs() as usize
    };

    let end = match bound.end_bound() {
        Bound::Unbounded => size as i32,
        Bound::Excluded(end) => *end,
        Bound::Included(-1) => size as i32,
        Bound::Included(end) => *end + 1,
    };
    let end = if end.abs() > size as i32 {
        return None;
    } else if end >= 0 {
        end as usize
    } else {
        size - end.abs() as usize
    };

    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

pub trait View {
    fn shape(&self) -> Shape;
    fn data(&self) -> &[Cell];
    fn data_mut(&mut self) -> &mut [Cell];

    fn hight(&self) -> usize {
        self.shape().height
    }

    fn width(&self) -> usize {
        self.shape().width
    }

    fn get(&self, row: usize, col: usize) -> Option<&Cell> {
        let shape = self.shape();
        if row >= shape.height || col >= shape.width {
            None
        } else {
            self.data().get(shape.index(row, col))
        }
    }

    fn view<'a, RS, CS>(&'a mut self, rows: RS, cols: CS) -> SurfaceView<'a>
    where
        RS: RangeBounds<i32>,
        CS: RangeBounds<i32>,
    {
        let shape = self.shape();
        match (
            resolve_bound(cols, shape.width),
            resolve_bound(rows, shape.height),
        ) {
            (Some((col_start, col_end)), Some((row_start, row_end))) => SurfaceView {
                shape: Shape {
                    offset: shape.offset + col_start + row_start * shape.row_stride,
                    col_stride: shape.col_stride,
                    row_stride: shape.row_stride,
                    width: col_end - col_start,
                    height: row_end - row_start,
                },
                data: self.data_mut(),
            },
            _ => SurfaceView {
                shape: Shape {
                    height: 0,
                    width: 0,
                    ..shape
                },
                data: self.data_mut(),
            },
        }
    }

    fn fill<F>(&mut self, mut fill: F)
    where
        F: FnMut(usize, usize, &mut Cell),
    {
        let shape = self.shape();
        let data = self.data_mut();
        for row in 0..shape.height {
            for col in 0..shape.width {
                fill(row, col, &mut data[shape.index(row, col)])
            }
        }
    }
}

impl View for Surface {
    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Cell] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [Cell] {
        &mut self.data
    }
}

impl<'a> View for SurfaceView<'a> {
    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Cell] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [Cell] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_bound;

    #[test]
    fn test_resolve_range() {
        assert_eq!(resolve_bound(.., 10), Some((0, 10)));
        assert_eq!(resolve_bound(..-1, 10), Some((0, 9)));
        assert_eq!(resolve_bound(..=-1, 10), Some((0, 10)));
        assert_eq!(resolve_bound(-5..8, 10), Some((5, 8)));
        assert_eq!(resolve_bound(-10.., 10), Some((0, 10)));
    }
}
