use crate::cell::Cell;
use std::ops::{Bound, Range, RangeBounds};

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Shape {
    pub row_stride: usize,
    pub col_stride: usize,
    pub height: usize,
    pub width: usize,
}

impl Shape {
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.row_stride + col * self.col_stride
    }
}

pub trait View {
    fn shape(&self) -> Shape;

    fn data(&self) -> &[Cell];

    fn is_empty(&self) -> bool {
        self.data().is_empty()
    }

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

    fn view<RS, CS>(&self, rows: RS, cols: CS) -> SurfaceView<'_>
    where
        RS: RangeBounds<i32>,
        CS: RangeBounds<i32>,
    {
        let (range, shape) = view_shape(self.shape(), rows, cols);
        let data = &self.data()[range];
        SurfaceView { shape, data }
    }

    fn iter(&self) -> ViewIter<'_> {
        ViewIter {
            row: 0,
            col: 0,
            shape: self.shape(),
            data: self.data(),
        }
    }
}

pub struct ViewIter<'a> {
    row: usize,
    col: usize,
    shape: Shape,
    data: &'a [Cell],
}

impl<'a> Iterator for ViewIter<'a> {
    type Item = &'a Cell;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.shape.height || self.col >= self.shape.width {
            None
        } else {
            let index = self.shape.index(self.row, self.col);
            self.col += 1;
            if self.col > self.shape.width {
                self.col = 0;
                self.row += 1;
            }
            Some(&self.data[index])
        }
    }
}

pub trait ViewMut: View {
    fn data_mut(&mut self) -> &mut [Cell];

    fn view_mut<RS, CS>(&mut self, rows: RS, cols: CS) -> SurfaceViewMut<'_>
    where
        RS: RangeBounds<i32>,
        CS: RangeBounds<i32>,
    {
        let (range, shape) = view_shape(self.shape(), rows, cols);
        let data = &mut self.data_mut()[range];
        SurfaceViewMut { shape, data }
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

#[derive(Clone)]
pub struct Surface {
    shape: Shape,
    data: Vec<Cell>,
}

impl Surface {
    pub fn new(height: usize, width: usize) -> Self {
        let mut data = Vec::new();
        data.resize_with(height * width, Default::default);
        Self {
            shape: Shape {
                row_stride: width,
                col_stride: 1,
                height,
                width,
            },
            data,
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
}

impl ViewMut for Surface {
    fn data_mut(&mut self) -> &mut [Cell] {
        &mut self.data
    }
}

pub struct SurfaceView<'a> {
    shape: Shape,
    data: &'a [Cell],
}

impl<'a> View for SurfaceView<'a> {
    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Cell] {
        &self.data
    }
}

pub struct SurfaceViewMut<'a> {
    shape: Shape,
    data: &'a mut [Cell],
}

impl<'a> View for SurfaceViewMut<'a> {
    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Cell] {
        &self.data
    }
}

impl<'a> ViewMut for SurfaceViewMut<'a> {
    fn data_mut(&mut self) -> &mut [Cell] {
        &mut self.data
    }
}

#[inline]
fn clip<T>(val: T, min: T, max: T) -> T
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

/// Resolve bounds the same way python or numpy does for ranges when indexing arrays
pub fn resolve_bound(bound: impl RangeBounds<i32>, size: usize) -> Option<(usize, usize)> {
    //  (index + size) % size - almost works
    //  0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9
    //-10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9
    let size = size as i32;

    let (start, offset) = match bound.start_bound() {
        Bound::Unbounded => (0, 0),
        Bound::Included(start) => (*start, 0),
        Bound::Excluded(start) => (*start, 1),
    };
    let offset = if start >= size { 1 } else { offset };
    let start = clip(start + size, 0, 2 * size - 1) % size + offset;

    let (end, offset) = match bound.end_bound() {
        Bound::Unbounded => (-1, 1),
        Bound::Included(end) => (*end, 1),
        Bound::Excluded(end) => (*end, 0),
    };
    let offset = if end >= size { 1 } else { offset };
    let end = clip(end + size, 0, 2 * size - 1) % size + offset;

    if end <= start {
        None
    } else {
        Some((start as usize, end as usize))
    }
}

/// Construt new offset and shape for
fn view_shape<RS, CS>(shape: Shape, rows: RS, cols: CS) -> (Range<usize>, Shape)
where
    RS: RangeBounds<i32>,
    CS: RangeBounds<i32>,
{
    match (
        resolve_bound(cols, shape.width),
        resolve_bound(rows, shape.height),
    ) {
        (Some((col_start, col_end)), Some((row_start, row_end))) => {
            let width = col_end - col_start;
            let height = row_end - row_start;
            let shape = Shape {
                width,
                height,
                ..shape
            };
            let start = col_start * shape.col_stride + row_start * shape.row_stride;
            let range = Range {
                start,
                end: start + shape.index(height - 1, width),
            };
            (range, shape)
        }
        _ => {
            let shape = Shape {
                height: 0,
                width: 0,
                row_stride: 0,
                col_stride: 0,
            };
            let range = Range { start: 0, end: 0 };
            (range, shape)
        }
    }
}

/*
use crate::TerminalCommand;

pub fn diff(src_surf: &Surface, dst_surf: &Surface) -> Vec<TerminalCommand> {
    use TerminalCommand::*

    let mut cmds = Vec::new();
    let mut cursor = CursorTo { row: 0, col: 0};

    cmds.push(cursor);
    let src_shape = src_surf.shape();
    for row in 0..shape.height {
        for col in 0..shape.width {
            match (src_surf.get(row, col), dst_surf.get(row, col)) {
                (Some(src), Some(dst)) => {
                    if src == dst {
                        continue;
                    }
                    if cursor
                }
                _ => break,
            }
        }
    }

    cmds
}
*/

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
        assert_eq!(resolve_bound(..20, 10), Some((0, 10)));
        assert_eq!(resolve_bound(10..20, 10), None);
        assert_eq!(resolve_bound(9..20, 10), Some((9, 10)));
    }
}
