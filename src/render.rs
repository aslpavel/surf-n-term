use crate::{clamp, ArrayIter, SurfaceMut, SurfaceOwned};
use std::{
    cmp::min,
    fmt,
    ops::{Add, Div, Mul, Sub},
    str::FromStr,
    usize,
};

pub type Scalar = f64;
const EPSILON: f64 = std::f64::EPSILON;
const PI: f64 = std::f64::consts::PI;

/// flatness of 0.05px gives good accuracy tradeoff
pub const FLATNESS: Scalar = 0.05;

#[derive(Clone, Copy)]
pub struct Point([Scalar; 2]);

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Point([x, y]) = self;
        write!(f, "(")?;
        scalar_fmt(f, *x)?;
        write!(f, ", ")?;
        scalar_fmt(f, *y)?;
        write!(f, ")")
    }
}

impl Point {
    #[inline]
    pub fn new(x: Scalar, y: Scalar) -> Self {
        Self([x, y])
    }

    #[inline]
    pub fn x(&self) -> Scalar {
        self.0[0]
    }

    #[inline]
    pub fn y(self) -> Scalar {
        self.0[1]
    }

    pub fn hypot(self) -> Scalar {
        let Self([x, y]) = self;
        x.hypot(y)
    }

    pub fn dot(self, other: Self) -> Scalar {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        x0 * x1 + y0 * y1
    }

    pub fn cross(self, other: Self) -> Scalar {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        x0 * y1 - y0 * x1
    }

    pub fn angle_between(self, other: Self) -> Scalar {
        let angle_cos = self.dot(other) / (self.hypot() * other.hypot());
        let angle = clamp(angle_cos, -1.0, 1.0).acos();
        if self.cross(other) < 0.0 {
            -angle
        } else {
            angle
        }
    }

    pub fn is_close_to(self, other: Point) -> bool {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        (x0 - x1).abs() < EPSILON && (y0 - y1).abs() < EPSILON
    }
}

impl From<(Scalar, Scalar)> for Point {
    fn from(xy: (Scalar, Scalar)) -> Self {
        Self([xy.0, xy.1])
    }
}

impl Mul<&Point> for Scalar {
    type Output = Point;

    #[inline]
    fn mul(self, other: &Point) -> Self::Output {
        let Point([x, y]) = other;
        Point([self * x, self * y])
    }
}

impl Mul<Point> for Scalar {
    type Output = Point;

    #[inline]
    fn mul(self, other: Point) -> Self::Output {
        let Point([x, y]) = other;
        Point([self * x, self * y])
    }
}

impl Div<Scalar> for Point {
    type Output = Point;

    fn div(self, rhs: Scalar) -> Self::Output {
        let Point([x, y]) = self;
        Point([x / rhs, y / rhs])
    }
}

impl Add for Point {
    type Output = Point;

    #[inline]
    fn add(self, other: Point) -> Self::Output {
        let Point([x0, y0]) = self;
        let Point([x1, y1]) = other;
        Point([x0 + x1, y0 + y1])
    }
}

impl Sub for Point {
    type Output = Point;

    #[inline]
    fn sub(self, other: Point) -> Self::Output {
        let Point([x0, y0]) = self;
        let Point([x1, y1]) = other;
        Point([x0 - x1, y0 - y1])
    }
}

impl Mul for Point {
    type Output = Point;

    #[inline]
    fn mul(self, other: Point) -> Self::Output {
        let Point([x0, y0]) = self;
        let Point([x1, y1]) = other;
        Point([x0 * x1, y0 * y1])
    }
}

pub trait Curve: Sized {
    /// Iterator returned by flatten method
    type FlattenIter: IntoIterator<Item = Line>;

    /// Convert curve to an iterator over line segments
    fn flatten(&self, tr: Transform, flatness: Scalar) -> Self::FlattenIter;

    /// Apply affine transformation to the curve
    fn transform(&self, tr: Transform) -> Self;

    /// Point at which curve starts
    fn start(&self) -> Point;

    /// Point at which curve ends
    fn end(&self) -> Point;

    /// Parametric representation of the curve at `t` (0.0 .. 1.0)
    fn at(&self, t: Scalar) -> Point;

    /// Split curve at specified parametric value `t` (0.0 .. 1.0)
    fn split_at(&self, t: Scalar) -> (Self, Self);

    /// Bounding box of the curve given initial bounding box.
    fn bbox(&self, init: Option<BBox>) -> BBox;

    /// Offset curve
    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>);
}

/// Line segment curve
#[derive(Clone, Copy)]
pub struct Line([Point; 2]);

impl Line {
    pub fn new(p0: impl Into<Point>, p1: impl Into<Point>) -> Self {
        Self([p0.into(), p1.into()])
    }

    pub fn len(&self) -> Scalar {
        let Self([p0, p1]) = self;
        (*p0 - *p1).hypot()
    }

    /// Find intersection of two lines
    ///
    /// Returns pair of parametric `t` parameters for this line and the other.
    /// Found by solving `self.at(t0) == self.at(t1)`. Real intersection can
    /// be found by making sure that `0.0 <= t0 <= 1.0 && 0.0 <= t1 <= 1.0`
    pub fn intersect(&self, other: Line) -> Option<(Scalar, Scalar)> {
        let Line([Point([x1, y1]), Point([x2, y2])]) = *self;
        let Line([Point([x3, y3]), Point([x4, y4])]) = other;
        let det = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3);
        if det.abs() < EPSILON {
            return None;
        }
        let t0 = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / det;
        let t1 = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / det;
        Some((t0, t1))
    }
}

fn line_offset(line: Line, dist: Scalar) -> Option<Line> {
    let Line([p0, p1]) = line;
    let len = line.len();
    if len < EPSILON {
        None
    } else {
        let v = p1 - p0;
        let d = Point::new(-v.y() * dist / len, v.x() * dist / len);
        Some(Line::new(p0 + d, p1 + d))
    }
}

fn three_point_offset(ps: [Point; 3], dist: Scalar) -> Option<[Point; 3]> {
    let [p0, p1, p2] = ps;
    let l0 = line_offset(Line::new(p0, p1), dist);
    let l1 = line_offset(Line::new(p1, p2), dist);
    match (l0, l1) {
        (Some(l0), None) => Some([l0.start(), l0.end(), l0.end()]),
        (None, Some(l1)) => Some([l1.start(), l1.start(), l1.end()]),
        (Some(l0), Some(l1)) => match l0.intersect(l1) {
            Some((t, _)) => Some([l0.start(), l0.at(t), l1.end()]),
            None => Some([l0.start(), l0.end(), l1.end()]),
        },
        _ => None,
    }
}

impl fmt::Debug for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Line([p0, p1]) = self;
        write!(f, "Line {:?} {:?}", p0, p1)
    }
}

impl Curve for Line {
    type FlattenIter = std::option::IntoIter<Self>;

    fn flatten(&self, tr: Transform, _: Scalar) -> Self::FlattenIter {
        Some(self.transform(tr)).into_iter()
    }

    fn transform(&self, tr: Transform) -> Self {
        let Line([p0, p1]) = self;
        Self([tr.apply(*p0), tr.apply(*p1)])
    }

    fn start(&self) -> Point {
        self.0[0]
    }

    fn end(&self) -> Point {
        self.0[1]
    }

    fn at(&self, t: Scalar) -> Point {
        let Self([p0, p1]) = self;
        (1.0 - t) * p0 + t * p1
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        let Self([p0, p1]) = self;
        let mid = self.at(t);
        (Self([*p0, mid]), Self([mid, *p1]))
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        let Self([p0, p1]) = self;
        BBox::new(*p0, *p1).union_opt(init)
    }

    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>) {
        out.extend(line_offset(*self, dist).map(Segment::from));
    }
}

/// Quadratic bezier curve
#[derive(Clone, Copy)]
pub struct Quad([Point; 3]);

impl fmt::Debug for Quad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Quad([p0, p1, p2]) = self;
        write!(f, "Quad {:?} {:?} {:?}", p0, p1, p2)
    }
}

impl Quad {
    pub fn new(p0: impl Into<Point>, p1: impl Into<Point>, p2: impl Into<Point>) -> Self {
        Self([p0.into(), p1.into(), p2.into()])
    }

    pub fn smooth(&self) -> Point {
        let Quad([_p0, p1, p2]) = self;
        2.0 * p2 - *p1
    }
}

impl Curve for Quad {
    type FlattenIter = CubicFlattenIter;

    fn flatten(&self, tr: Transform, flatness: Scalar) -> Self::FlattenIter {
        Cubic::from(*self).flatten(tr, flatness)
    }

    fn transform(&self, tr: Transform) -> Self {
        let Quad([p0, p1, p2]) = self;
        Self([tr.apply(*p0), tr.apply(*p1), tr.apply(*p2)])
    }

    fn start(&self) -> Point {
        self.0[0]
    }

    fn end(&self) -> Point {
        self.0[2]
    }

    fn at(&self, t: Scalar) -> Point {
        // at(t) =
        //   (1 - t) ^ 2 * p0 +
        //   2 * (1 - t) * t * p1 +
        //   t ^ 2 * p2
        let Self([p0, p1, p2]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        t_2 * p0 + 2.0 * t1 * t_1 * p1 + t2 * p2
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        // https://pomax.github.io/bezierinfo/#matrixsplit
        let Self([p0, p1, p2]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        let mid = t_2 * p0 + 2.0 * t1 * t_1 * p1 + t2 * p2;
        (
            Self([*p0, t_1 * p0 + t * p1, mid]),
            Self([mid, t_1 * p1 + t * p2, *p2]),
        )
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        let Self([p0, p1, p2]) = self;
        let mut bbox = BBox::new(*p0, *p2).union_opt(init);
        if bbox.contains(*p1) {
            return bbox;
        }
        let Point([a0, a1]) = *p2 - 2.0 * p1 + *p0;
        let Point([b0, b1]) = *p1 - *p0;
        // curve'(t)_x = 0
        if a0.abs() > EPSILON {
            let t0 = -b0 / a0;
            if t0 >= 0.0 && t0 <= 1.0 {
                bbox = bbox.extend(self.at(t0));
            }
        }
        // curve'(t)_y = 0
        if a1.abs() > EPSILON {
            let t1 = -b1 / a1;
            if t1 >= 0.0 && t1 <= 1.0 {
                bbox = bbox.extend(self.at(t1));
            }
        }

        bbox
    }

    fn offset(&self, _dist: Scalar, _out: &mut impl Extend<Segment>) {
        todo!()
    }
}

/// Cubic bezier curve
#[derive(Clone, Copy)]
pub struct Cubic([Point; 4]);

impl fmt::Debug for Cubic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Cubic([p0, p1, p2, p3]) = self;
        write!(f, "Cubic {:?} {:?} {:?} {:?}", p0, p1, p2, p3)
    }
}

impl Cubic {
    pub fn new(
        p0: impl Into<Point>,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
        p3: impl Into<Point>,
    ) -> Self {
        Self([p0.into(), p1.into(), p2.into(), p3.into()])
    }

    pub fn smooth(&self) -> Point {
        let Cubic([_p0, _p1, p2, p3]) = self;
        2.0 * p3 - *p2
    }

    /// Flattness criteria for a batch of bezier3 curves
    ///
    /// It is equal to `f = maxarg d(t) where d(t) = |b(t) - l(t)|, l(t) = (1 - t) * b0 + t * b3`
    /// for b(t) bezier3 curve with b{0..3} control points, in other words maximum distance
    /// from parametric line to bezier3 curve for the same parameter t. It is proven in the article
    /// that:
    ///     f^2 <= 1/16 (max{u_x^2, v_x^2} + max{u_y^2, v_y^2})
    /// where:
    ///     u = 3 * b1 - 2 * b0 - b3
    ///     v = 3 * b2 - b0 - 2 * b3
    /// `f == 0` means completely flat so estimating upper bound is sufficient as spliting more
    /// than needed is not a problem for rendering.
    ///
    /// [Linear Approximation of Bezier Curve](https://hcklbrrfnn.files.wordpress.com/2012/08/bez.pdf)
    fn flatness(&self) -> Scalar {
        let Self([p0, p1, p2, p3]) = self;
        let u = 3.0 * p1 - 2.0 * p0 - *p3;
        let v = 3.0 * p2 - *p0 - 2.0 * p3;
        (u.x() * u.x()).max(v.x() * v.x()) + (u.y() * u.y()).max(v.y() * v.y())
    }

    /// Optimized version of `split_at(0.5)`
    fn split(&self) -> (Self, Self) {
        let Self([p0, p1, p2, p3]) = self;
        let mid = 0.125 * p0 + 0.375 * p1 + 0.375 * p2 + 0.125 * p3;
        let c0 = Self([
            *p0,
            0.5 * p0 + 0.5 * p1,
            0.25 * p0 + 0.5 * p1 + 0.25 * p2,
            mid,
        ]);
        let c1 = Self([
            mid,
            0.25 * p1 + 0.5 * p2 + 0.25 * p3,
            0.5 * p2 + 0.5 * p3,
            *p3,
        ]);
        (c0, c1)
    }

    pub fn extremities(&self) -> impl Iterator<Item = Scalar> {
        let Self([p0, p1, p2, p3]) = self;
        // Solve for `curve'(t)_x = 0 || curve'(t)_y = 0`
        let Point([a0, a1]) = -1.0 * p0 + 3.0 * p1 - 3.0 * p2 + 1.0 * p3;
        let Point([b0, b1]) = 2.0 * p0 - 4.0 * p1 + 2.0 * p2;
        let Point([c0, c1]) = -1.0 * p0 + *p1;
        quadratic_solve(a0, b0, c0)
            .chain(quadratic_solve(a1, b1, c1))
            .filter(|t| *t >= 0.0 && *t <= 1.0)
            .collect::<ArrayIter<[Option<Scalar>; 6]>>()
    }
}

impl Curve for Cubic {
    type FlattenIter = CubicFlattenIter;

    fn flatten(&self, tr: Transform, flatness: Scalar) -> CubicFlattenIter {
        CubicFlattenIter {
            flatness: 16.0 * flatness * flatness,
            cubics: vec![self.transform(tr)],
        }
    }

    fn transform(&self, tr: Transform) -> Self {
        let Cubic([p0, p1, p2, p3]) = self;
        Self([tr.apply(*p0), tr.apply(*p1), tr.apply(*p2), tr.apply(*p3)])
    }

    fn start(&self) -> Point {
        self.0[0]
    }

    fn end(&self) -> Point {
        self.0[3]
    }

    fn at(&self, t: Scalar) -> Point {
        // at(t) =
        //   (1 - t) ^ 3 * p0 +
        //   3 * (1 - t) ^ 2 * t * p1 +
        //   3 * (1 - t) * t ^ 2 * p2 +
        //   t ^ 3 * p3
        let Self([p0, p1, p2, p3]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        let (t3, t_3) = (t2 * t1, t_2 * t_1);
        t_3 * p0 + 3.0 * t1 * t_2 * p1 + 3.0 * t2 * t_1 * p2 + t3 * p3
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        // https://pomax.github.io/bezierinfo/#matrixsplit
        let Self([p0, p1, p2, p3]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        let (t3, t_3) = (t2 * t1, t_2 * t_1);
        let mid = t_3 * p0 + 3.0 * t1 * t_2 * p1 + 3.0 * t2 * t_1 * p2 + t3 * p3;
        let c0 = Self([
            *p0,
            t_1 * p0 + t * p1,
            t_2 * p0 + 2.0 * t * t_1 * p1 + t2 * p2,
            mid,
        ]);
        let c1 = Self([
            mid,
            t_2 * p1 + 2.0 * t * t_1 * p2 + t2 * p3,
            t_1 * p2 + t * p3,
            *p3,
        ]);
        (c0, c1)
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        let Self([p0, p1, p2, p3]) = self;
        let bbox = BBox::new(*p0, *p3).union_opt(init);
        if bbox.contains(*p1) && bbox.contains(*p2) {
            return bbox;
        }
        self.extremities()
            .fold(bbox, |bbox, t| bbox.extend(self.at(t)))
    }

    /// Offset cubic bezier curve with a list of cubic curves
    ///
    /// Offset bezier curve using Tiller-Hanson method. In short, it will just offset
    /// line segment corresponding to control points, then find intersection of this
    /// lines and treat them as new control points.
    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>) {
        // TODO: recurse and limit the depth?
        let mut queue = Vec::new();
        queue.push(*self);
        while let Some(cubic) = queue.pop() {
            if cubic_offset_should_split(cubic) {
                let (c0, c1) = cubic.split();
                // queue in reverse order so curves would appear in order
                queue.push(c1);
                queue.push(c0);
                continue;
            }
            let Self([p0, p1, p2, p3]) = cubic;
            let result = if p1.is_close_to(p2) {
                match three_point_offset([p0, p1, p3], dist) {
                    None => continue,
                    Some([p0, p1, p2]) => Cubic::new(p0, p1, p1, p2),
                }
            } else {
                let o0 = three_point_offset([p0, p1, p2], dist);
                let o1 = three_point_offset([p1, p2, p3], dist);
                match (o0, o1) {
                    (Some([p0, p1, p2]), None) => Cubic::new(p0, p1, p2, p2),
                    (None, Some([p0, p1, p2])) => Cubic::new(p0, p0, p1, p2),
                    (Some([p0, p1, _]), Some([_, p2, p3])) => Cubic::new(p0, p1, p2, p3),
                    (None, None) => continue,
                }
            };
            out.extend(Some(Segment::from(result)));
        }
    }
}

/// Determine if cubic curve needs splitting before offsetting.
fn cubic_offset_should_split(cubic: Cubic) -> bool {
    let Cubic([p0, p1, p2, p3]) = cubic;
    // angle(p3 - p0, p2 - p1) > 90 or < -90
    if (p3 - p0).dot(p2 - p1) < 0.0 {
        return true;
    }
    // control points should be on the same side of the baseline.
    let a0 = (p3 - p0).cross(p1 - p0);
    let a1 = (p3 - p0).cross(p2 - p0);
    if a0 * a1 < 0.0 {
        return true;
    }
    // distance between center mass and midpoint of a cruve,
    // should be bigger then 0.1 of the bounding box diagonal
    let c_mass = (p0 + p1 + p2 + p3) / 4.0;
    let c_mid = cubic.at(0.5);
    let _dist = (c_mass - c_mid).hypot();
    todo!()
}

pub struct CubicFlattenIter {
    flatness: Scalar,
    cubics: Vec<Cubic>,
}

impl Iterator for CubicFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cubics.pop() {
                None => {
                    return None;
                }
                Some(cubic) if cubic.flatness() < self.flatness => {
                    let Cubic([p0, _p1, _p2, p3]) = cubic;
                    return Some(Line([p0, p3]));
                }
                Some(cubic) => {
                    let (c0, c1) = cubic.split();
                    self.cubics.push(c1);
                    self.cubics.push(c0);
                }
            }
        }
    }
}

impl From<Quad> for Cubic {
    fn from(quad: Quad) -> Self {
        let Quad([p0, p1, p2]) = quad;
        Self([
            p0,
            (1.0 / 3.0) * p0 + (2.0 / 3.0) * p1,
            (2.0 / 3.0) * p1 + (1.0 / 3.0) * p2,
            p2,
        ])
    }
}

#[derive(Clone, Copy)]
pub struct ElipArc {
    center: Point,
    rx: Scalar,
    ry: Scalar,
    phi: Scalar,
    eta: Scalar,
    eta_delta: Scalar,
}

impl fmt::Debug for ElipArc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Arc center:{:?} radius:{:?} phi:{:.3?} eta:{:.3?} eta_delta:{:.3?}",
            self.center,
            Point([self.rx, self.ry]),
            self.phi,
            self.eta,
            self.eta_delta
        )
    }
}

impl ElipArc {
    /// Convert arc from SVG arguments to parametric curve
    ///
    /// This code mostly comes from arc implementation notes from svg sepc
    /// (Arc to Parametric)[https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes]
    pub fn new_param(
        src: Point,
        dst: Point,
        rx: Scalar,
        ry: Scalar,
        x_axis_rot: Scalar,
        large_flag: bool,
        sweep_flag: bool,
    ) -> Self {
        let rx = rx.abs();
        let ry = ry.abs();
        let phi = x_axis_rot * PI / 180.0;

        // Eq 5.1
        let Point([x1, y1]) = Transform::default().rotate(-phi).apply(0.5 * (src - dst));
        // scale/normalize radii
        let s = (x1 / rx).powi(2) + (y1 / ry).powi(2);
        let (rx, ry) = if s > 1.0 {
            let s = s.sqrt();
            (rx * s, ry * s)
        } else {
            (rx, ry)
        };
        // Eq 5.2
        let sq = ((rx * ry).powi(2) / ((rx * y1).powi(2) + (ry * x1).powi(2)) - 1.0)
            .max(0.0)
            .sqrt();
        let sq = if large_flag == sweep_flag { -sq } else { sq };
        let center = sq * Point([rx * y1 / ry, -ry * x1 / rx]);
        let Point([cx, cy]) = center;
        // Eq 5.3 convert center to initail coordinates
        let center = Transform::default().rotate(phi).apply(center) + 0.5 * (dst + src);
        // Eq 5.5-6
        let v0 = Point([1.0, 0.0]);
        let v1 = Point([(x1 - cx) / rx, (y1 - cy) / ry]);
        let v2 = Point([(-x1 - cx) / rx, (-y1 - cy) / ry]);
        // initial angle
        let eta = v0.angle_between(v1);
        //delta angle to be covered when t changes from 0..1
        let eta_delta = v1.angle_between(v2).rem_euclid(2.0 * PI);
        let eta_delta = if !sweep_flag && eta_delta > 0.0 {
            eta_delta - 2.0 * PI
        } else if sweep_flag && eta_delta < 0.0 {
            eta_delta + 2.0 * PI
        } else {
            eta_delta
        };

        Self {
            center,
            rx,
            ry,
            phi,
            eta,
            eta_delta,
        }
    }

    pub fn to_cubic(&self) -> ElipArcCubicIter {
        ElipArcCubicIter::new(*self)
    }

    pub fn flatten(&self, tr: Transform, flatness: Scalar) -> ElipArcFlattenIter {
        ElipArcFlattenIter::new(*self, tr, flatness)
    }
}

impl Curve for ElipArc {
    type FlattenIter = ElipArcFlattenIter;

    fn flatten(&self, tr: Transform, flatness: Scalar) -> ElipArcFlattenIter {
        ElipArcFlattenIter::new(*self, tr, flatness)
    }

    fn transform(&self, _tr: Transform) -> ElipArc {
        todo!()
    }

    fn start(&self) -> Point {
        self.at(0.0)
    }

    fn end(&self) -> Point {
        self.at(1.0)
    }

    fn at(&self, t: Scalar) -> Point {
        let (angle_sin, angle_cos) = (self.eta + t * self.eta_delta).sin_cos();
        let point = Point([self.rx * angle_cos, self.ry * angle_sin]);
        Transform::default().rotate(self.phi).apply(point) + self.center
    }

    fn split_at(&self, _t: Scalar) -> (Self, Self) {
        todo!()
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        ElipArcCubicIter::new(*self)
            .fold(init, |bbox, cubic| Some(cubic.bbox(bbox)))
            .expect("ElipArcCubicIter is empty")
    }

    fn offset(&self, _dist: Scalar, _out: &mut impl Extend<Segment>) {
        todo!()
    }
}

/// Approximate arc with a sequnce of cubic bezier curves
///
/// [Drawing an elliptical arc using polylines, quadratic or cubic Bezier curves]
/// (http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf)
/// [Approximating Arcs Using Cubic Bézier Curves]
/// (https://www.joecridge.me/content/pdf/bezier-arcs.pdf)
///
/// We are using following formula to split arc segment from `eta_1` to `eta_2`
/// to achieve good approximation arc is split in segments smaller then `pi / 2`.
///     P0 = A(eta_1)
///     P1 = P0 + alpha * A'(eta_1)
///     P2 = P3 - alpha * A'(eta_2)
///     P3 = A(eta_2)
/// where
///     A - arc parametrized by angle
///     A' - derivative of arc parametrized by angle
///     eta_1 = eta
///     eta_2 = eta + eta_delta
///     alpha = sin(eta_2 - eta_1) * (sqrt(4 + 3 * tan((eta_2 - eta_1) / 2) ** 2) - 1) / 3
pub struct ElipArcCubicIter {
    arc: ElipArc,
    phi_tr: Transform,
    segment_delta: Scalar,
    segment_index: Scalar,
    segment_count: Scalar,
}

impl ElipArcCubicIter {
    fn new(arc: ElipArc) -> Self {
        let phi_tr = Transform::default().rotate(arc.phi);
        let segment_max_angle = PI / 4.0; // maximum `eta_delta` of a segment
        let segment_count = (arc.eta_delta.abs() / segment_max_angle).ceil();
        let segment_delta = arc.eta_delta / segment_count;
        Self {
            arc,
            phi_tr,
            segment_delta,
            segment_index: 0.0,
            segment_count: segment_count - 1.0,
        }
    }

    fn at(&self, alpha: Scalar) -> (Point, Point) {
        let (sin, cos) = alpha.sin_cos();
        let at = self
            .phi_tr
            .apply(Point([self.arc.rx * cos, self.arc.ry * sin]))
            + self.arc.center;
        let at_deriv = self
            .phi_tr
            .apply(Point([-self.arc.rx * sin, self.arc.ry * cos]));
        (at, at_deriv)
    }
}

impl Iterator for ElipArcCubicIter {
    type Item = Cubic;

    fn next(&mut self) -> Option<Self::Item> {
        if self.segment_index > self.segment_count {
            return None;
        }
        let eta_1 = self.arc.eta + self.segment_delta * self.segment_index;
        let eta_2 = eta_1 + self.segment_delta;
        self.segment_index += 1.0;

        let sq = (4.0 + 3.0 * ((eta_2 - eta_1) / 2.0).tan().powi(2)).sqrt();
        let alpha = (eta_2 - eta_1).sin() * (sq - 1.0) / 3.0;
        let (p0, d0) = self.at(eta_1);
        let (p3, d3) = self.at(eta_2);
        let p1 = p0 + alpha * d0;
        let p2 = p3 - alpha * d3;
        Some(Cubic([p0, p1, p2, p3]))
    }
}

pub struct ElipArcFlattenIter {
    tr: Transform,
    flatness: Scalar,
    cubics: ElipArcCubicIter,
    cubic: Option<CubicFlattenIter>,
}

impl ElipArcFlattenIter {
    fn new(arc: ElipArc, tr: Transform, flatness: Scalar) -> Self {
        Self {
            tr,
            flatness,
            cubics: arc.to_cubic(),
            cubic: None,
        }
    }
}

impl Iterator for ElipArcFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cubic.as_mut().and_then(Iterator::next) {
                line @ Some(_) => return line,
                None => self.cubic = Some(self.cubics.next()?.flatten(self.tr, self.flatness)),
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum Segment {
    Line(Line),
    Quad(Quad),
    Cubic(Cubic),
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Line(line) => line.fmt(f),
            Segment::Quad(quad) => quad.fmt(f),
            Segment::Cubic(cubic) => cubic.fmt(f),
        }
    }
}

impl Curve for Segment {
    type FlattenIter = SegmentFlattenIter;

    fn flatten(&self, tr: Transform, flatness: Scalar) -> SegmentFlattenIter {
        match self {
            Segment::Line(line) => SegmentFlattenIter::Line(line.flatten(tr, flatness)),
            Segment::Quad(quad) => SegmentFlattenIter::Cubic(quad.flatten(tr, flatness)),
            Segment::Cubic(cubic) => SegmentFlattenIter::Cubic(cubic.flatten(tr, flatness)),
        }
    }

    fn transform(&self, tr: Transform) -> Self {
        match self {
            Segment::Line(line) => line.transform(tr).into(),
            Segment::Quad(quad) => quad.transform(tr).into(),
            Segment::Cubic(cubic) => cubic.transform(tr).into(),
        }
    }

    fn start(&self) -> Point {
        match self {
            Segment::Line(line) => line.start(),
            Segment::Quad(quad) => quad.start(),
            Segment::Cubic(cubic) => cubic.start(),
        }
    }

    fn end(&self) -> Point {
        match self {
            Segment::Line(line) => line.end(),
            Segment::Quad(quad) => quad.end(),
            Segment::Cubic(cubic) => cubic.end(),
        }
    }

    fn at(&self, t: Scalar) -> Point {
        match self {
            Segment::Line(line) => line.at(t),
            Segment::Quad(quad) => quad.at(t),
            Segment::Cubic(cubic) => cubic.at(t),
        }
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        match self {
            Segment::Line(line) => {
                let (l0, l1) = line.split_at(t);
                (l0.into(), l1.into())
            }
            Segment::Quad(quad) => {
                let (q0, q1) = quad.split_at(t);
                (q0.into(), q1.into())
            }
            Segment::Cubic(cubic) => {
                let (c0, c1) = cubic.split_at(t);
                (c0.into(), c1.into())
            }
        }
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        match self {
            Segment::Line(line) => line.bbox(init),
            Segment::Quad(quad) => quad.bbox(init),
            Segment::Cubic(cubic) => cubic.bbox(init),
        }
    }

    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>) {
        match self {
            Segment::Line(line) => line.offset(dist, out),
            Segment::Quad(quad) => quad.offset(dist, out),
            Segment::Cubic(cubic) => cubic.offset(dist, out),
        }
    }
}

impl From<Line> for Segment {
    fn from(line: Line) -> Self {
        Self::Line(line)
    }
}

impl From<Quad> for Segment {
    fn from(quad: Quad) -> Self {
        Self::Quad(quad)
    }
}

impl From<Cubic> for Segment {
    fn from(cubic: Cubic) -> Self {
        Self::Cubic(cubic)
    }
}

pub enum SegmentFlattenIter {
    Line(<Line as Curve>::FlattenIter),
    Cubic(<Cubic as Curve>::FlattenIter),
}

impl Iterator for SegmentFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Line(line) => line.next(),
            Self::Cubic(cubic) => cubic.next(),
        }
    }
}

#[derive(Clone)]
pub struct SubPath {
    segments: Vec<Segment>,
    closed: bool,
}

impl fmt::Debug for SubPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for segment in self.segments.iter() {
            writeln!(f, "{:?}", segment)?;
        }
        if self.closed {
            writeln!(f, "Close")?;
        } else {
            writeln!(f, "End")?
        }
        Ok(())
    }
}

impl SubPath {
    pub fn new(segments: Vec<Segment>, closed: bool) -> Option<Self> {
        if segments.is_empty() {
            None
        } else {
            Some(Self { segments, closed })
        }
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn flatten<'a>(
        &'a self,
        tr: Transform,
        flatness: Scalar,
        close: bool,
    ) -> impl Iterator<Item = Line> + 'a {
        let last = if self.closed || close {
            Some(Line::new(self.end(), self.start()).transform(tr))
        } else {
            None
        };
        self.segments
            .iter()
            .flat_map(move |segment| segment.flatten(tr, flatness))
            .chain(last)
    }

    fn start(&self) -> Point {
        self.segments
            .first()
            .expect("SubPath is never emtpy")
            .start()
    }

    fn end(&self) -> Point {
        self.segments.last().expect("SubPath is never empty").end()
    }

    pub fn bbox(&self, init: Option<BBox>, tr: Transform) -> BBox {
        self.segments
            .iter()
            .fold(init, |bbox, seg| Some(seg.transform(tr).bbox(bbox)))
            .expect("SubPath is never empty")
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone)]
pub struct Path {
    subpaths: Vec<SubPath>,
}

impl fmt::Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.subpaths.is_empty() {
            write!(f, "Empty")?;
        } else {
            for subpath in self.subpaths.iter() {
                subpath.fmt(f)?
            }
        }
        Ok(())
    }
}

impl Path {
    /// Create path from the list of subpaths
    pub fn new(subpaths: Vec<SubPath>) -> Self {
        Self { subpaths }
    }

    pub fn subpaths(&self) -> &[SubPath] {
        &self.subpaths
    }

    /// Convenience method to create `PathBuilder`
    pub fn builder() -> PathBuilder {
        PathBuilder::new()
    }

    /// Convert path to an iterator over line segments
    pub fn flatten(
        &self,
        tr: Transform,
        flatness: Scalar,
        close: bool,
    ) -> impl Iterator<Item = Line> + '_ {
        PathFlattenIter::new(self, tr, flatness, close)
    }

    /// Bounding box of the path after provided transformation is applied.
    pub fn bbox(&self, tr: Transform) -> Option<BBox> {
        self.subpaths
            .iter()
            .fold(None, |bbox, subpath| Some(subpath.bbox(bbox, tr)))
    }

    /// Rasterize mast for the path in into a provided surface.
    ///
    /// Everything that is outside of the surface will be cropped. Surface is assumed
    /// to contain zeros.
    pub fn rasterize_to<S: SurfaceMut<Item = Scalar>>(
        &self,
        tr: Transform,
        fill_rule: FillRule,
        mut surf: S,
    ) -> S {
        for line in self.flatten(tr, FLATNESS, true) {
            signed_difference_line(&mut surf, line);
        }
        signed_difference_to_mask(&mut surf, fill_rule);
        surf
    }

    /// Rasterize fitted mask for the path into a provided sruface.
    ///
    /// Path is rescaled and centered appropriately to fit into a provided surface.
    pub fn rasterize_fit<S: SurfaceMut<Item = Scalar>>(
        &self,
        tr: Transform,
        fill_rule: FillRule,
        align: Align,
        surf: S,
    ) -> S {
        if surf.height() < 3 || surf.height() < 3 {
            return surf;
        }
        let src_bbox = match self.bbox(tr) {
            Some(bbox) if bbox.width() > 0.0 && bbox.height() > 0.0 => bbox,
            _ => return surf,
        };
        let dst_bbox = BBox::new(
            Point::new(1.0, 1.0),
            Point::new((surf.width() - 1) as Scalar, (surf.height() - 1) as Scalar),
        );
        let tr = Transform::fit(src_bbox, dst_bbox, align) * tr;
        self.rasterize_to(tr, fill_rule, surf)
    }

    /// Rasteraize mask for the path into an allocated surface.
    ///
    /// Surface of required size will be allocated.
    pub fn rasterize(&self, tr: Transform, fill_rule: FillRule) -> SurfaceOwned<Scalar> {
        let bbox = match self.bbox(tr) {
            Some(bbox) => bbox,
            None => return SurfaceOwned::new(0, 0),
        };
        // one pixel border to account for anti-aliasing
        let width = (bbox.width() + 2.0).ceil() as usize;
        let height = (bbox.height() + 2.0).ceil() as usize;
        let surf = SurfaceOwned::new(height, width);
        let shift = Transform::default().translate(1.0 - bbox.x(), 1.0 - bbox.y());
        self.rasterize_to(shift * tr, fill_rule, surf)
    }
}

pub struct PathFlattenIter<'a> {
    path: &'a Path,
    transform: Transform,
    flatness: Scalar,
    close: bool,
    subpath: usize,
    segment: usize,
    stack: Vec<Cubic>,
}

impl<'a> PathFlattenIter<'a> {
    fn new(path: &'a Path, transform: Transform, flatness: Scalar, close: bool) -> Self {
        Self {
            path,
            transform,
            flatness: 16.0 * flatness * flatness,
            close,
            subpath: 0,
            segment: 0,
            stack: Default::default(),
        }
    }
}

impl<'a> Iterator for PathFlattenIter<'a> {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stack.pop() {
                Some(cubic) => {
                    if cubic.flatness() < self.flatness {
                        return Some(Line::new(cubic.start(), cubic.end()));
                    }
                    let (c0, c1) = cubic.split();
                    self.stack.push(c1);
                    self.stack.push(c0);
                }
                None => {
                    let subpath = self.path.subpaths.get(self.subpath)?;
                    match subpath.segments().get(self.segment) {
                        None => {
                            self.subpath += 1;
                            self.segment = 0;
                            if subpath.closed || self.close {
                                let line = Line::new(subpath.end(), subpath.start())
                                    .transform(self.transform);
                                return Some(line);
                            }
                        }
                        Some(segment) => {
                            self.segment += 1;
                            match segment {
                                Segment::Line(line) => return Some(line.transform(self.transform)),
                                Segment::Quad(quad) => {
                                    self.stack.push(quad.transform(self.transform).into());
                                }
                                Segment::Cubic(cubic) => {
                                    self.stack.push(cubic.transform(self.transform));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct PathBuilder {
    position: Point,
    subpath: Vec<Segment>,
    subpaths: Vec<SubPath>,
}

impl Default for PathBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PathBuilder {
    pub fn new() -> Self {
        Self {
            position: Point::new(0.0, 0.0),
            subpath: Default::default(),
            subpaths: Default::default(),
        }
    }

    /// Build path
    pub fn build(self) -> Path {
        let PathBuilder {
            subpath,
            mut subpaths,
            ..
        } = self;
        subpaths.extend(SubPath::new(subpath, false));
        Path::new(subpaths)
    }

    /// Extend path from string, which is specified in the same format as SVGs path element.
    pub fn from_str(self, string: impl AsRef<[u8]>) -> Result<Self, PathParseError> {
        let parser = PathParser::new(string.as_ref());
        parser.parse(self)
    }

    /// Move current position, ending current subpath
    pub fn move_to(mut self, p: impl Into<Point>) -> Self {
        let subpath = std::mem::replace(&mut self.subpath, Vec::new());
        self.subpaths.extend(SubPath::new(subpath, false));
        self.position = p.into();
        self
    }

    /// Close current subpath
    pub fn close(mut self) -> Self {
        let subpath = std::mem::replace(&mut self.subpath, Vec::new());
        if let Some(seg) = subpath.first() {
            self.position = seg.start();
        }
        self.subpaths.extend(SubPath::new(subpath, true));
        self
    }

    /// Add line from the current position to the specified point
    pub fn line_to(mut self, p: impl Into<Point>) -> Self {
        let line = Line::new(self.position, p);
        self.position = line.end();
        self.subpath.push(line.into());
        self
    }

    /// Add quadratic bezier curve
    pub fn quad_to(mut self, p1: impl Into<Point>, p2: impl Into<Point>) -> Self {
        let quad = Quad::new(self.position, p1, p2);
        self.position = quad.end();
        self.subpath.push(quad.into());
        self
    }

    /// Add smooth quadratic bezier curve
    pub fn quad_smooth_to(self, p2: impl Into<Point>) -> Self {
        let p1 = match self.subpath.last() {
            Some(Segment::Quad(quad)) => quad.smooth(),
            _ => self.position,
        };
        self.quad_to(p1, p2)
    }

    /// Add cubic beizer curve
    pub fn cubic_to(
        mut self,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
        p3: impl Into<Point>,
    ) -> Self {
        let cubic = Cubic::new(self.position, p1, p2, p3);
        self.position = cubic.end();
        self.subpath.push(cubic.into());
        self
    }

    /// Add smooth cubic bezier curve
    pub fn cubic_smooth_to(self, p2: impl Into<Point>, p3: impl Into<Point>) -> Self {
        let p1 = match self.subpath.last() {
            Some(Segment::Cubic(cubic)) => cubic.smooth(),
            _ => self.position,
        };
        self.cubic_to(p1, p2, p3)
    }

    /// Add elliptic arc segment
    pub fn arc_to(
        mut self,
        radii: impl Into<Point>,
        x_axis_rot: Scalar,
        large: bool,
        sweep: bool,
        p: impl Into<Point>,
    ) -> Self {
        let radii: Point = radii.into();
        let p = p.into();
        let arc = ElipArc::new_param(
            self.position,
            p,
            radii.x(),
            radii.y(),
            x_axis_rot,
            large,
            sweep,
        );
        self.subpath.extend(arc.to_cubic().map(Segment::from));
        self.position = p;
        self
    }

    /// Current possition of the builder
    pub fn position(&self) -> Point {
        self.position
    }
}

impl FromStr for Path {
    type Err = PathParseError;

    fn from_str(text: &str) -> Result<Path, Self::Err> {
        let parser = PathParser::new(text.as_ref());
        let builder = parser.parse(PathBuilder::new())?;
        Ok(builder.build())
    }
}

#[derive(Debug)]
pub struct PathParseError {
    message: String,
    offset: usize,
}

impl fmt::Display for PathParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.offset)
    }
}

impl std::error::Error for PathParseError {}

#[derive(Debug)]
pub struct PathParser<'a> {
    // text containing unparsed path
    text: &'a [u8],
    // current offset in the text
    offset: usize,
    // previous command
    prev_cmd: Option<u8>,
    // current position from which next curve will start
    position: Point,
}

impl<'a> PathParser<'a> {
    fn new(text: &'a [u8]) -> PathParser<'a> {
        Self {
            text,
            offset: 0,
            prev_cmd: None,
            position: Point::new(0.0, 0.0),
        }
    }

    fn error<S: Into<String>>(&self, message: S) -> PathParseError {
        PathParseError {
            message: message.into(),
            offset: self.offset,
        }
    }

    fn current(&self) -> Result<u8, PathParseError> {
        match self.text.get(self.offset) {
            Some(byte) => Ok(*byte),
            None => Err(self.error("current: end of input is reached")),
        }
    }

    fn advance(&mut self, count: usize) {
        self.offset += count;
    }

    fn is_eof(&self) -> bool {
        self.offset >= self.text.len()
    }

    fn parse_separators(&mut self) -> Result<(), PathParseError> {
        while !self.is_eof() {
            match self.text[self.offset] {
                b' ' | b'\t' | b'\r' | b'\n' | b',' => {
                    self.offset += 1;
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn parse_digits(&mut self) -> Result<bool, PathParseError> {
        let mut found = false;
        loop {
            match self.current() {
                Ok(b'0'..=b'9') => {
                    self.advance(1);
                    found = true;
                }
                _ => return Ok(found),
            }
        }
    }

    fn parse_sign(&mut self) -> Result<(), PathParseError> {
        match self.current()? {
            b'-' | b'+' => {
                self.advance(1);
            }
            _ => (),
        }
        Ok(())
    }

    fn parse_scalar(&mut self) -> Result<Scalar, PathParseError> {
        self.parse_separators()?;
        let start = self.offset;
        self.parse_sign()?;
        let whole = self.parse_digits()?;
        if !self.is_eof() {
            let fraction = match self.current()? {
                b'.' => {
                    self.advance(1);
                    self.parse_digits()?
                }
                _ => false,
            };
            if !whole && !fraction {
                return Err(self.error("parse_scalar: missing whole and fractional value"));
            }
            match self.current()? {
                b'e' | b'E' => {
                    self.advance(1);
                    self.parse_sign()?;
                    if !self.parse_digits()? {
                        return Err(self.error("parse_scalar: missing exponent value"));
                    }
                }
                _ => (),
            }
        }
        // unwrap is safe here since we have validated content
        let scalar_str = std::str::from_utf8(&self.text[start..self.offset]).unwrap();
        let scalar = Scalar::from_str(scalar_str).unwrap();
        Ok(scalar)
    }

    fn parse_point(&mut self) -> Result<Point, PathParseError> {
        let x = self.parse_scalar()?;
        let y = self.parse_scalar()?;
        let is_relative = match self.prev_cmd {
            Some(cmd) => cmd.is_ascii_lowercase(),
            None => false,
        };
        if is_relative {
            Ok(Point([x, y]) + self.position)
        } else {
            Ok(Point([x, y]))
        }
    }

    fn parse_flag(&mut self) -> Result<bool, PathParseError> {
        self.parse_separators()?;
        match self.current()? {
            b'0' => {
                self.advance(1);
                Ok(false)
            }
            b'1' => {
                self.advance(1);
                Ok(true)
            }
            flag => Err(self.error(format!("parse_flag: invalid flag `{}`", flag))),
        }
    }

    fn parse_cmd(&mut self) -> Result<u8, PathParseError> {
        let cmd = self.current()?;
        match cmd {
            b'M' | b'm' | b'L' | b'l' | b'V' | b'v' | b'H' | b'h' | b'C' | b'c' | b'S' | b's'
            | b'Q' | b'q' | b'T' | b't' | b'A' | b'a' | b'Z' | b'z' => {
                self.advance(1);
                self.prev_cmd = if cmd == b'm' {
                    Some(b'l')
                } else if cmd == b'M' {
                    Some(b'L')
                } else if cmd == b'Z' || cmd == b'z' {
                    None
                } else {
                    Some(cmd)
                };
                Ok(cmd)
            }
            _ => match self.prev_cmd {
                Some(cmd) => Ok(cmd),
                None => {
                    Err(self.error(format!("parse_cmd: command expected `{}`", char::from(cmd))))
                }
            },
        }
    }

    fn parse(mut self, mut builder: PathBuilder) -> Result<PathBuilder, PathParseError> {
        loop {
            self.parse_separators()?;
            if self.is_eof() {
                break;
            }
            self.position = builder.position();
            let cmd = self.parse_cmd()?;
            builder = match cmd {
                b'M' | b'm' => builder.move_to(self.parse_point()?),
                b'L' | b'l' => builder.line_to(self.parse_point()?),
                b'V' | b'v' => {
                    let y = self.parse_scalar()?;
                    let p0 = builder.position();
                    let p1 = if cmd == b'v' {
                        Point::new(p0.x(), p0.y() + y)
                    } else {
                        Point::new(p0.x(), y)
                    };
                    builder.line_to(p1)
                }
                b'H' | b'h' => {
                    let x = self.parse_scalar()?;
                    let p0 = builder.position();
                    let p1 = if cmd == b'h' {
                        Point::new(p0.x() + x, p0.y())
                    } else {
                        Point::new(x, p0.y())
                    };
                    builder.line_to(p1)
                }
                b'Q' | b'q' => builder.quad_to(self.parse_point()?, self.parse_point()?),
                b'T' | b't' => builder.quad_smooth_to(self.parse_point()?),
                b'C' | b'c' => builder.cubic_to(
                    self.parse_point()?,
                    self.parse_point()?,
                    self.parse_point()?,
                ),
                b'S' | b's' => builder.cubic_smooth_to(self.parse_point()?, self.parse_point()?),
                b'A' | b'a' => {
                    let rx = self.parse_scalar()?;
                    let ry = self.parse_scalar()?;
                    let x_axis_rot = self.parse_scalar()?;
                    let large_flag = self.parse_flag()?;
                    let sweep_flag = self.parse_flag()?;
                    let dst = self.parse_point()?;
                    builder.arc_to((rx, ry), x_axis_rot, large_flag, sweep_flag, dst)
                }
                b'Z' | b'z' => builder.close(),
                _ => unreachable!(),
            }
        }
        Ok(builder)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Transform([Scalar; 6]);

impl Default for Transform {
    fn default() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }
}

impl Transform {
    pub fn apply(&self, point: Point) -> Point {
        let Self([m00, m01, m02, m10, m11, m12]) = self;
        let Point([x, y]) = point;
        Point([x * m00 + y * m01 + m02, x * m10 + y * m11 + m12])
    }

    pub fn invert(&self) -> Option<Self> {
        // inv([[M, v], [0, 1]]) = [[inv(M), - inv(M) * v], [0, 1]]
        let Self([m00, m01, m02, m10, m11, m12]) = self;
        let det = m00 * m11 - m10 * m01;
        if det.abs() <= EPSILON {
            return None;
        }
        let o00 = m11 / det;
        let o01 = -m01 / det;
        let o10 = -m10 / det;
        let o11 = m00 / det;
        let o02 = -o00 * m02 - o01 * m12;
        let o12 = -o10 * m02 - o11 * m12;
        Some(Self([o00, o01, o02, o10, o11, o12]))
    }

    pub fn translate(&self, tx: Scalar, ty: Scalar) -> Self {
        self.matmul(Self([1.0, 0.0, tx, 0.0, 1.0, ty]))
    }

    pub fn scale(&self, sx: Scalar, sy: Scalar) -> Self {
        self.matmul(Self([sx, 0.0, 0.0, 0.0, sy, 0.0]))
    }

    pub fn rotate(&self, a: Scalar) -> Self {
        let (sin, cos) = a.sin_cos();
        self.matmul(Self([cos, -sin, 0.0, sin, cos, 0.0]))
    }

    pub fn rotate_around(&self, a: Scalar, p: impl Into<Point>) -> Self {
        let p = p.into();
        self.translate(p.x(), p.y())
            .rotate(a)
            .translate(-p.x(), -p.y())
    }

    pub fn skew(&self, ax: Scalar, ay: Scalar) -> Self {
        self.matmul(Self([1.0, ax.tan(), 0.0, ay.tan(), 1.0, 0.0]))
    }

    pub fn matmul(&self, other: Transform) -> Self {
        let Self([s00, s01, s02, s10, s11, s12]) = self;
        let Self([o00, o01, o02, o10, o11, o12]) = other;

        // s00, s01, s02 | o00, o01, o02
        // s10, s11, s12 | o10, o11, o12
        // 0  , 0  , 1   | 0  , 0  , 1
        Self([
            s00 * o00 + s01 * o10,
            s00 * o01 + s01 * o11,
            s00 * o02 + s01 * o12 + s02,
            s10 * o00 + s11 * o10,
            s10 * o01 + s11 * o11,
            s10 * o02 + s11 * o12 + s12,
        ])
    }

    pub fn fit(src: BBox, dst: BBox, align: Align) -> Transform {
        let scale = (dst.height() / src.height()).min(dst.width() / src.width());
        let base = Transform::default()
            .translate(dst.x(), dst.y())
            .scale(scale, scale)
            .translate(-src.x(), -src.y());
        let align = match align {
            Align::Min => Transform::default(),
            Align::Mid => Transform::default().translate(
                (dst.width() - src.width() * scale) / 2.0,
                (dst.height() - src.height() * scale) / 2.0,
            ),
            Align::Max => Transform::default().translate(
                dst.width() - src.width() * scale,
                dst.height() - src.height() * scale,
            ),
        };
        align * base
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Align {
    Min,
    Mid,
    Max,
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    fn mul(self, other: Transform) -> Self::Output {
        self.matmul(other)
    }
}

/// Bounding box
#[derive(Clone, Copy)]
pub struct BBox {
    min: Point,
    max: Point,
}

impl BBox {
    pub fn new(p0: Point, p1: Point) -> Self {
        let Point([x0, y0]) = p0;
        let Point([x1, y1]) = p1;
        let (x0, x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (y0, y1) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        Self {
            min: Point([x0, y0]),
            max: Point([x1, y1]),
        }
    }

    #[inline]
    pub fn x(&self) -> Scalar {
        self.min.x()
    }

    #[inline]
    pub fn y(&self) -> Scalar {
        self.min.y()
    }

    #[inline]
    pub fn width(&self) -> Scalar {
        self.max.x() - self.min.x()
    }

    #[inline]
    pub fn height(&self) -> Scalar {
        self.max.y() - self.min.y()
    }

    /// Determine if the point is inside of the bounding box
    pub fn contains(&self, point: Point) -> bool {
        let Point([x, y]) = point;
        self.min.x() <= x && x <= self.max.x() && self.min.y() <= y && y <= self.max.y()
    }

    /// Extend bounding box so it would contains provided point
    pub fn extend(&self, point: Point) -> Self {
        let Point([x, y]) = point;
        let Point([x0, y0]) = self.min;
        let Point([x1, y1]) = self.max;
        let (x0, x1) = if x < x0 {
            (x, x1)
        } else if x > x1 {
            (x0, x)
        } else {
            (x0, x1)
        };
        let (y0, y1) = if y < y0 {
            (y, y1)
        } else if y > y1 {
            (y0, y)
        } else {
            (y0, y1)
        };
        Self {
            min: Point([x0, y0]),
            max: Point([x1, y1]),
        }
    }

    /// Create bounding box the spans both bbox-es
    pub fn union(&self, other: BBox) -> Self {
        self.extend(other.min).extend(other.max)
    }

    pub fn union_opt(&self, other: Option<BBox>) -> Self {
        match other {
            Some(other) => self.union(other),
            None => *self,
        }
    }
}

impl fmt::Debug for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BBox ")?;
        scalar_fmt(f, self.x())?;
        write!(f, ", ")?;
        scalar_fmt(f, self.y())?;
        write!(f, ", ")?;
        scalar_fmt(f, self.width())?;
        write!(f, ", ")?;
        scalar_fmt(f, self.height())
    }
}

/// Update provided surface with the signed difference of the line
///
/// Signed difference is a diffrence between adjacent pixels introduced by the line.
fn signed_difference_line(mut surf: impl SurfaceMut<Item = Scalar>, line: Line) {
    // y - is a row
    // x - is a column
    let Line([p0, p1]) = line;

    // handle lines that are intersecting `x == surf.width()`
    // - just throw away part that has x > surf.width for all points
    let width = surf.width() as Scalar - 1.0;
    let line = if p0.x() > width || p1.x() > width {
        if p0.x() > width && p1.x() > width {
            return;
        }
        let t = (p0.x() - width) / (p0.x() - p1.x());
        let mid = Point::new(width, (1.0 - t) * p0.y() + t * p1.y());
        if p0.x() < width {
            Line::new(p0, mid)
        } else {
            Line::new(mid, p1)
        }
    } else {
        line
    };

    // handle lines that are intersecting `x == 0.0`
    // - line is splitted in left (for all points where x < 0.0) and the mid part
    // - left part is converted to a vertical line that spans same y's and x == 0.0
    // - left part is rastterized recursivelly, and mid part rasterized after this
    let line = if p0.x() < 0.0 || p1.x() < 0.0 {
        let (vertical, line) = if p1.x() > 0.0 || p0.x() > 0.0 {
            let t = p0.x() / (p0.x() - p1.x());
            let mid = Point::new(0.0, (1.0 - t) * p0.y() + t * p1.y());
            if p1.x() > 0.0 {
                let p = Point::new(0.0, p0.y());
                (Line::new(p, mid), Line::new(mid, p1))
            } else {
                let p = Point::new(0.0, p1.y());
                (Line::new(mid, p), Line::new(p0, mid))
            }
        } else {
            (
                Line::new((0.0, p0.y()), (0.0, p1.y())),
                Line::new((0.0, 0.0), (0.0, 0.0)),
            )
        };
        // signed difference by the line left of `x == 0.0`
        signed_difference_line(surf.view_mut(.., ..), vertical);
        line
    } else {
        line
    };

    let Line([p0, p1]) = line;
    let shape = surf.shape();
    let data = surf.data_mut();
    let stride = shape.col_stride;

    if (p0.y() - p1.y()).abs() < EPSILON {
        // line does not introduce any signed converage
        return;
    }
    // always iterate from the point with the smallest y coordinate
    let (dir, p0, p1) = if p0.y() < p1.y() {
        (1.0, p0, p1)
    } else {
        (-1.0, p1, p0)
    };
    let dxdy = (p1.x() - p0.x()) / (p1.y() - p0.y());
    // find first point to trace. since we are going to interate over y's
    // we should pick min(y , p0.y) as a starting y point, and adjust x
    // accordingly
    let y = p0.y().max(0.0) as usize;
    let mut x = if p0.y() < 0.0 {
        p0.x() - p0.y() * dxdy
    } else {
        p0.x()
    };
    let mut x_next = x;
    for y in y..min(shape.height, p1.y().ceil().max(0.0) as usize) {
        x = x_next;
        let line_offset = shape.offset(y, 0); // current line offset in the data array
        let dy = ((y + 1) as Scalar).min(p1.y()) - (y as Scalar).max(p0.y());
        // signed y difference
        let d = dir * dy;
        // find next x position
        x_next = x + dxdy * dy;
        // order (x, x_next) from smaller value x0 to bigger x1
        let (x0, x1) = if x < x_next { (x, x_next) } else { (x_next, x) };
        // lower bound of effected x pixels
        let x0_floor = x0.floor().max(0.0);
        let x0i = x0_floor as i32;
        // uppwer bound of effected x pixels
        let x1_ceil = x1.ceil();
        let x1i = x1_ceil as i32;
        if x1i <= x0i + 1 {
            // only goes through one pixel (with the total coverage of `d` spread over two pixels)
            let xmf = 0.5 * (x + x_next) - x0_floor; // effective height
            data[line_offset + (x0i as usize) * stride] += d * (1.0 - xmf);
            data[line_offset + ((x0i + 1) as usize) * stride] += d * xmf;
        } else {
            let s = (x1 - x0).recip();
            let x0f = x0 - x0_floor; // fractional part of x0
            let x1f = x1 - x1_ceil + 1.0; // fractional part of x1
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f); // fractional area of the pixel with smallest x
            let am = 0.5 * s * x1f * x1f; // fractional area of the pixel with largest x
            data[line_offset + (x0i as usize) * stride] += d * a0;
            if x1i == x0i + 2 {
                // only two pixels are covered
                data[line_offset + ((x0i + 1) as usize) * stride] += d * (1.0 - a0 - am);
            } else {
                // second pixel
                let a1 = s * (1.5 - x0f);
                data[line_offset + ((x0i + 1) as usize) * stride] += d * (a1 - a0);
                // (second, last) pixels
                for xi in x0i + 2..x1i - 1 {
                    data[line_offset + (xi as usize) * stride] += d * s;
                }
                // last pixel
                let a2 = a1 + (x1i - x0i - 3) as Scalar * s;
                data[line_offset + ((x1i - 1) as usize) * stride] += d * (1.0 - a2 - am);
            }
            data[line_offset + (x1i as usize) * stride] += d * am
        }
    }
}

pub fn signed_difference_to_mask(mut surf: impl SurfaceMut<Item = Scalar>, fill_rule: FillRule) {
    let shape = surf.shape();
    let data = surf.data_mut();
    match fill_rule {
        FillRule::NonZero => {
            for y in 0..shape.height {
                let mut acc = 0.0;
                for x in 0..shape.width {
                    let offset = shape.offset(y, x);
                    acc += data[offset];

                    let value = acc.abs();
                    data[offset] = if value > 1.0 {
                        1.0
                    } else if value < 1e-6 {
                        0.0
                    } else {
                        value
                    };
                }
            }
        }
        FillRule::EvenOdd => {
            for y in 0..shape.height {
                let mut acc = 0.0;
                for x in 0..shape.width {
                    let offset = shape.offset(y, x);
                    acc += data[offset];

                    data[offset] = ((acc + 1.0).rem_euclid(2.0) - 1.0).abs()
                }
            }
        }
    }
}

fn quadratic_solve(a: Scalar, b: Scalar, c: Scalar) -> impl Iterator<Item = Scalar> {
    let mut result = ArrayIter::<[Option<Scalar>; 2]>::new();
    if a.abs() < EPSILON {
        if b.abs() > EPSILON {
            result.push(-c / b);
        }
        return result;
    }
    let det = b * b - 4.0 * a * c;
    if det.abs() < EPSILON {
        result.push(-b / (2.0 * a));
    } else if det > 0.0 {
        let sq = det.sqrt();
        result.push((-b + sq) / (2.0 * a));
        result.push((-b - sq) / (2.0 * a));
    }
    result
}

fn scalar_fmt(f: &mut fmt::Formatter<'_>, value: Scalar) -> fmt::Result {
    if value.abs() < EPSILON {
        write!(f, "0.0")
    } else if value.abs() > 9999.0 || value.abs() <= 0.0001 {
        write!(f, "{:.3e}", value)
    } else {
        write!(f, "{:.3}", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Surface;

    macro_rules! assert_approx_eq {
        ( $v0:expr, $v1: expr ) => {{
            assert!(($v0 - $v1).abs() < EPSILON, "{} != {}", $v0, $v1);
        }};
        ( $v0:expr, $v1: expr, $e: expr ) => {{
            assert!(($v0 - $v1).abs() < $e, "{} != {}", $v0, $v1);
        }};
    }

    #[test]
    fn test_signed_difference_line() {
        let mut surf = SurfaceOwned::new(2, 5);

        // line convers many columns but just one row
        signed_difference_line(&mut surf, Line::new((0.5, 1.0), (3.5, 0.0)));
        // covered areas per-pixel
        let a0 = (0.5 * (1.0 / 6.0)) / 2.0;
        let a1 = ((1.0 / 6.0) + (3.0 / 6.0)) / 2.0;
        let a2 = ((3.0 / 6.0) + (5.0 / 6.0)) / 2.0;
        assert_approx_eq!(*surf.get(0, 0).unwrap(), -a0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), a0 - a1);
        assert_approx_eq!(*surf.get(0, 2).unwrap(), a1 - a2);
        assert_approx_eq!(*surf.get(0, 3).unwrap(), a0 - a1);
        assert_approx_eq!(*surf.get(0, 4).unwrap(), -a0);
        // total difference
        let a: Scalar = surf.iter().sum();
        assert_approx_eq!(a, -1.0);
        surf.clear();

        // out of bound line (intersects x = 0.0)
        signed_difference_line(&mut surf, Line::new((-1.0, 0.0), (1.0, 1.0)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 3.0 / 4.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 / 4.0);
        surf.clear();

        // multiple rows diag
        signed_difference_line(&mut surf, Line::new((0.0, -0.5), (2.0, 1.5)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 - 2.0 / 8.0);
        assert_approx_eq!(*surf.get(0, 2).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 1).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 2).unwrap(), 0.5 - 1.0 / 8.0);
        surf.clear();

        // multiple rows vertical
        signed_difference_line(&mut surf, Line::new((0.5, 0.5), (0.5, 1.75)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 1.0 / 4.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 / 4.0);
        assert_approx_eq!(*surf.get(1, 0).unwrap(), 3.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 1).unwrap(), 3.0 / 8.0);
        surf.clear();
    }

    #[test]
    fn test_bbox() {
        let cubic = Cubic::new((106.0, 0.0), (0.0, 100.0), (382.0, 216.0), (324.0, 14.0));
        let bbox = cubic.bbox(None);
        assert_approx_eq!(bbox.x(), 87.308, 0.001);
        assert_approx_eq!(bbox.y(), 0.0, 0.001);
        assert_approx_eq!(bbox.width(), 242.724, 0.001);
        assert_approx_eq!(bbox.height(), 125.140, 0.001);

        let quad = Quad::new((30.0, 90.0), (220.0, 200.0), (120.0, 50.0));
        let bbox = quad.bbox(None);
        assert_approx_eq!(bbox.x(), 30.0, 0.001);
        assert_approx_eq!(bbox.y(), 50.0, 0.001);
        assert_approx_eq!(bbox.width(), 124.483, 0.001);
        assert_approx_eq!(bbox.height(), 86.538, 0.001);

        let cubic = Cubic::new((0.0, 0.0), (10.0, -3.0), (-4.0, -3.0), (6.0, 0.0));
        let bbox = cubic.bbox(None);
        assert_approx_eq!(bbox.x(), 0.0);
        assert_approx_eq!(bbox.y(), -2.25);
        assert_approx_eq!(bbox.width(), 6.0);
        assert_approx_eq!(bbox.height(), 2.25);

        let path: Path = SQUIRREL.parse().unwrap();
        let bbox = path.bbox(Transform::default()).unwrap();
        assert_approx_eq!(bbox.x(), 0.25);
        assert_approx_eq!(bbox.y(), 1.0);
        assert_approx_eq!(bbox.width(), 15.75);
        assert_approx_eq!(bbox.height(), 14.0);
    }

    const SQUIRREL: &str = r#"
    M12 1C9.79 1 8 2.31 8 3.92c0 1.94.5 3.03 0 6.08 0-4.5-2.77-6.34-4-6.34.05-.5-.48
    -.66-.48-.66s-.22.11-.3.34c-.27-.31-.56-.27-.56-.27l-.13.58S.7 4.29 .68 6.87c.2.33
    1.53.6 2.47.43.89.05.67.79.47.99C2.78 9.13 2 8 1 8S0 9 1 9s1 1 3 1c-3.09 1.2 0 4 0 4
    H3c-1 0-1 1-1 1h6c3 0 5-1 5-3.47 0-.85-.43-1.79 -1-2.53-1.11-1.46.23-2.68 1-2
    .77.68 3 1 3-2 0-2.21-1.79-4-4-4zM2.5 6 c-.28 0-.5-.22-.5-.5s.22-.5.5-.5.5.22.5.5
    -.22.5-.5.5z
    "#;

    #[test]
    fn test_path_parse() -> Result<(), PathParseError> {
        let path: Path = SQUIRREL.parse()?;
        let reference = Path::builder()
            .move_to((12.0, 1.0))
            .cubic_to((9.79, 1.0), (8.0, 2.31), (8.0, 3.92))
            .cubic_to((8.0, 5.86), (8.5, 6.95), (8.0, 10.0))
            .cubic_to((8.0, 5.5), (5.23, 3.66), (4.0, 3.66))
            .cubic_to((4.05, 3.16), (3.52, 3.0), (3.52, 3.0))
            .cubic_to((3.52, 3.0), (3.3, 3.11), (3.22, 3.34))
            .cubic_to((2.95, 3.03), (2.66, 3.07), (2.66, 3.07))
            .line_to((2.53, 3.65))
            .cubic_to((2.53, 3.65), (0.7, 4.29), (0.68, 6.87))
            .cubic_to((0.88, 7.2), (2.21, 7.47), (3.15, 7.3))
            .cubic_to((4.04, 7.35), (3.82, 8.09), (3.62, 8.29))
            .cubic_to((2.78, 9.13), (2.0, 8.0), (1.0, 8.0))
            .cubic_to((0.0, 8.0), (0.0, 9.0), (1.0, 9.0))
            .cubic_to((2.0, 9.0), (2.0, 10.0), (4.0, 10.0))
            .cubic_to((0.91, 11.2), (4.0, 14.0), (4.0, 14.0))
            .line_to((3.0, 14.0))
            .cubic_to((2.0, 14.0), (2.0, 15.0), (2.0, 15.0))
            .line_to((8.0, 15.0))
            .cubic_to((11.0, 15.0), (13.0, 14.0), (13.0, 11.53))
            .cubic_to((13.0, 10.68), (12.57, 9.74), (12.0, 9.0))
            .cubic_to((10.89, 7.54), (12.23, 6.32), (13.0, 7.0))
            .cubic_to((13.77, 7.68), (16.0, 8.0), (16.0, 5.0))
            .cubic_to((16.0, 2.79), (14.21, 1.0), (12.0, 1.0))
            .close()
            .move_to((2.5, 6.0))
            .cubic_to((2.22, 6.0), (2.0, 5.78), (2.0, 5.5))
            .cubic_to((2.0, 5.22), (2.22, 5.0), (2.5, 5.0))
            .cubic_to((2.78, 5.0), (3.0, 5.22), (3.0, 5.5))
            .cubic_to((3.0, 5.78), (2.78, 6.0), (2.5, 6.0))
            .close()
            .build();
        assert_eq!(format!("{:?}", path), format!("{:?}", reference));

        let path: Path = " M0,0L1-1L1,0ZL0,1 L1,1Z ".parse()?;
        let reference = Path::new(vec![
            SubPath::new(
                vec![
                    Line::new((0.0, 0.0), (1.0, -1.0)).into(),
                    Line::new((1.0, -1.0), (1.0, 0.0)).into(),
                ],
                true,
            )
            .unwrap(),
            SubPath::new(
                vec![
                    Line::new((0.0, 0.0), (0.0, 1.0)).into(),
                    Line::new((0.0, 1.0), (1.0, 1.0)).into(),
                ],
                true,
            )
            .unwrap(),
        ]);
        assert_eq!(format!("{:?}", path), format!("{:?}", reference));
        Ok(())
    }

    #[test]
    fn test_flatten() -> Result<(), PathParseError> {
        let path: Path = SQUIRREL.parse()?;
        let tr = Transform::default()
            .rotate(PI / 3.0)
            .translate(-10.0, -20.0);
        let lines: Vec<_> = path.flatten(tr, FLATNESS, true).collect();
        let mut reference = Vec::new();
        for subpath in path.subpaths() {
            let subpath_lines: Vec<_> = subpath.flatten(tr, FLATNESS, false).collect();
            // line are connected
            for ls in subpath_lines.windows(2) {
                assert!(ls[0].end().is_close_to(ls[1].start()));
            }
            reference.extend(subpath_lines);
        }
        assert_eq!(reference.len(), lines.len());
        for (l0, l1) in lines.iter().zip(reference.iter()) {
            assert!(l0.start().is_close_to(l1.start()));
            assert!(l0.end().is_close_to(l1.end()));
        }
        Ok(())
    }

    #[test]
    fn test_fill_rule() -> Result<(), PathParseError> {
        let tr = Transform::default();
        let path: Path = r#"
            M50,0 21,90 98,35 2,35 79,90z
            M110,0 h90 v90 h-90z
            M130,20 h50 v50 h-50 z
            M210,0  h90 v90 h-90 z
            M230,20 v50 h50 v-50 z
        "#
        .parse()?;
        let y = 50;
        let x0 = 50; // middle of the star
        let x1 = 150; // middle of the first box
        let x2 = 250; // middle of the second box

        let surf = path.rasterize(tr, FillRule::EvenOdd);
        assert_approx_eq!(surf.get(y, x0).unwrap(), 0.0);
        assert_approx_eq!(surf.get(y, x1).unwrap(), 0.0);
        assert_approx_eq!(surf.get(y, x2).unwrap(), 0.0);
        let area = surf.iter().sum::<Scalar>();
        assert_approx_eq!(area, 13130.0, 1.0);

        let surf = path.rasterize(tr, FillRule::NonZero);
        assert_approx_eq!(surf.get(y, x0).unwrap(), 1.0);
        assert_approx_eq!(surf.get(y, x1).unwrap(), 1.0);
        assert_approx_eq!(surf.get(y, x2).unwrap(), 0.0);
        let area = surf.iter().sum::<Scalar>();
        assert_approx_eq!(area, 16492.5, 1.0);

        Ok(())
    }

    #[test]
    fn test_trasform() {
        let tr = Transform::default()
            .translate(1.0, 2.0)
            .rotate(PI / 3.0)
            .skew(2.0, 3.0)
            .scale(3.0, 2.0);
        let inv = tr.invert().unwrap();
        let p0 = Point::new(1.0, 1.0);

        let p1 = tr.apply(p0);
        assert_approx_eq!(p1.x(), -1.04674389, 1e-6);
        assert_approx_eq!(p1.y(), 1.59965634, 1e-6);

        let p2 = inv.apply(p1);
        assert_approx_eq!(p2.x(), 1.0, 1e-6);
        assert_approx_eq!(p2.y(), 1.0, 1e-6);
    }

    #[test]
    fn test_split_at() {
        let cubic = Cubic::new((3.0, 7.0), (2.0, 8.0), (0.0, 3.0), (6.0, 5.0));
        assert_eq!(
            format!("{:?}", cubic.split()),
            format!("{:?}", cubic.split_at(0.5))
        );
    }
}
