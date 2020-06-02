use log::debug;
use std::{
    cmp::min,
    fmt,
    io::Write,
    ops::{Add, Mul, Sub},
    str::FromStr,
    usize,
};

pub mod surface;
use crate::surface::{Surface, SurfaceMut, SurfaceOwned};

type Scalar = f64;

const EPSILON: f64 = std::f64::EPSILON;
const INFINITY: f64 = std::f64::INFINITY;
const NEG_INFINITY: f64 = std::f64::NEG_INFINITY;
const PI: f64 = std::f64::consts::PI;

/// flatness of 0.1px gives good accuracy tradeoff
const FLATNESS: Scalar = 0.1;

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

fn quadratic_solve(a: Scalar, b: Scalar, c: Scalar) -> [Option<Scalar>; 2] {
    let mut result = [None; 2];
    let det = b * b - 4.0 * a * c;
    if det.abs() < EPSILON {
        result[0] = Some(-b / (2.0 * a));
    } else if det > 0.0 {
        let sq = det.sqrt();
        result[0] = Some((-b + sq) / (2.0 * a));
        result[1] = Some((-b - sq) / (2.0 * a));
    }
    result
}

fn scalar_fmt(f: &mut fmt::Formatter<'_>, value: Scalar) -> fmt::Result {
    if value.abs() < EPSILON {
        write!(f, "0.0")
    } else {
        if value.abs() > 9999.0 || value.abs() <= 0.0001 {
            write!(f, "{:.3e}", value)
        } else {
            write!(f, "{:.3}", value)
        }
    }
}

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
    pub fn y(&self) -> Scalar {
        self.0[1]
    }

    pub fn hypot(&self) -> Scalar {
        let Self([x, y]) = self;
        x.hypot(*y)
    }

    pub fn dot(&self, other: &Self) -> Scalar {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        x0 * x1 + y0 * y1
    }

    pub fn cross(&self, other: &Self) -> Scalar {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        x0 * y1 - y0 * x1
    }

    pub fn angle_between(&self, other: &Self) -> Scalar {
        let angle_cos = self.dot(other) / (self.hypot() * other.hypot());
        let angle = clamp(angle_cos, -1.0, 1.0).acos();
        if self.cross(other) < 0.0 {
            -angle
        } else {
            angle
        }
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

#[derive(Clone, Copy, Debug)]
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
        self.matmul(&Self([1.0, 0.0, tx, 0.0, 1.0, ty]))
    }

    pub fn scale(&self, sx: Scalar, sy: Scalar) -> Self {
        self.matmul(&Self([sx, 0.0, 0.0, 0.0, sy, 0.0]))
    }

    pub fn rotate(&self, a: Scalar) -> Self {
        let (sin, cos) = a.sin_cos();
        self.matmul(&Self([cos, -sin, 0.0, sin, cos, 0.0]))
    }

    pub fn rotate_around(&self, a: Scalar, p: Point) -> Self {
        self.translate(p.x(), p.y())
            .rotate(a)
            .translate(-p.x(), -p.y())
    }

    pub fn skew(&self, ax: Scalar, ay: Scalar) -> Self {
        self.matmul(&Self([1.0, ax.tan(), 0.0, ay.tan(), 1.0, 0.0]))
    }

    pub fn matmul(&self, other: &Transform) -> Self {
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
}

pub trait Transformable {
    fn transform(&self, tr: Transform) -> Self;
}

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

    pub fn contains(&self, point: Point) -> bool {
        let Point([x, y]) = point;
        self.min.x() <= x && x <= self.max.x() && self.min.y() <= y && y <= self.max.y()
    }

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

// Line curve
#[derive(Clone, Copy)]
pub struct Line([Point; 2]);

impl fmt::Debug for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Line([p0, p1]) = self;
        write!(f, "Line {:?} {:?}", p0, p1)
    }
}

impl Line {
    pub fn new<P: Into<Point>>(p0: P, p1: P) -> Self {
        Self([p0.into(), p1.into()])
    }

    pub fn flatten(&self, tr: Transform) -> LineFlattenIter {
        LineFlattenIter(Some(self.transform(tr)))
    }

    pub fn from(&self) -> Point {
        self.0[0]
    }

    pub fn to(&self) -> Point {
        self.0[1]
    }

    pub fn at(&self, t: Scalar) -> Point {
        let Self([p0, p1]) = self;
        (1.0 - t) * p0 + t * p1
    }

    pub fn bbox(&self) -> BBox {
        let Self([p0, p1]) = self;
        BBox::new(*p0, *p1)
    }
}

impl Transformable for Line {
    fn transform(&self, tr: Transform) -> Self {
        let Line([p0, p1]) = self;
        Self([tr.apply(*p0), tr.apply(*p1)])
    }
}

pub struct LineFlattenIter(Option<Line>);

impl Iterator for LineFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.take()
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
    pub fn new<P: Into<Point>>(p0: P, p1: P, p2: P) -> Self {
        Self([p0.into(), p1.into(), p2.into()])
    }

    pub fn from(&self) -> Point {
        self.0[0]
    }

    pub fn to(&self) -> Point {
        self.0[2]
    }

    pub fn at(&self, t: Scalar) -> Point {
        let Self([p0, p1, p2]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        t_2 * p0 + 2.0 * t1 * t_1 * p1 + t2 * p2
    }

    pub fn smooth(&self) -> Point {
        let Quad([_p0, p1, p2]) = self;
        2.0 * p2 - *p1
    }
}

impl Transformable for Quad {
    fn transform(&self, tr: Transform) -> Self {
        let Quad([p0, p1, p2]) = self;
        Self([tr.apply(*p0), tr.apply(*p1), tr.apply(*p2)])
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
    pub fn new<P: Into<Point>>(p0: P, p1: P, p2: P, p3: P) -> Self {
        Self([p0.into(), p1.into(), p2.into(), p3.into()])
    }

    pub fn flatten(&self, tr: Transform, flatness: Scalar) -> CubicFlattenIter {
        CubicFlattenIter {
            flatness: 16.0 * flatness * flatness,
            cubics: vec![self.transform(tr)],
        }
    }

    pub fn from(&self) -> Point {
        self.0[0]
    }

    pub fn to(&self) -> Point {
        self.0[3]
    }

    pub fn at(&self, t: Scalar) -> Point {
        let Self([p0, p1, p2, p3]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        let (t3, t_3) = (t2 * t1, t_2 * t_1);
        t_3 * p0 + 3.0 * t1 * t_2 * p1 + 3.0 * t2 * t_1 * p2 + t3 * p3
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

    /// Split cubic curve at `t = 0.5`
    fn split(&self) -> (Self, Self) {
        let Self([p0, p1, p2, p3]) = self;
        let mid = 0.125 * p0 + 0.375 * p1 + 0.375 * p2 + 0.125 * p3;
        let left = Self([
            *p0,
            0.5 * p0 + 0.5 * p1,
            0.25 * p0 + 0.5 * p1 + 0.25 * p2,
            mid,
        ]);
        let right = Self([
            mid,
            0.25 * p1 + 0.5 * p2 + 0.25 * p3,
            0.5 * p2 + 0.5 * p3,
            *p3,
        ]);
        (left, right)
    }

    pub fn bbox(&self) -> BBox {
        let Self([p0, p1, p2, p3]) = self;
        let bbox = BBox::new(*p0, *p3);
        if bbox.contains(*p1) && bbox.contains(*p2) {
            return bbox;
        }

        // Solve for `curve'(t)_x = 0 || curve'(t)_y = 0`
        let Point([a0, a1]) = -3.0 * p0 + 9.0 * p1 - 9.0 * p2 + 3.0 * p3;
        let Point([b0, b1]) = 6.0 * p0 - 12.0 * p1 + 6.0 * p2;
        let Point([c0, c1]) = -3.0 * p0 + 3.0 * p1;
        quadratic_solve(a0, b0, c0)
            .iter()
            .flatten()
            .chain(quadratic_solve(a1, b1, c1).iter().flatten())
            .filter(|t| **t > 0.0 && **t < 1.0)
            .fold(bbox, |bbox, t| bbox.extend(self.at(*t)))
    }
}

impl Transformable for Cubic {
    fn transform(&self, tr: Transform) -> Self {
        let Cubic([p0, p1, p2, p3]) = self;
        Self([tr.apply(*p0), tr.apply(*p1), tr.apply(*p2), tr.apply(*p3)])
    }
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
        let eta = v0.angle_between(&v1);
        //delta angle to be covered when t changes from 0..1
        let eta_delta = v1.angle_between(&v2).rem_euclid(2.0 * PI);
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

    pub fn at(&self, t: Scalar) -> Point {
        let (angle_sin, angle_cos) = (self.eta + t * self.eta_delta).sin_cos();
        let point = Point([self.rx * angle_cos, self.ry * angle_sin]);
        Transform::default().rotate(self.phi).apply(point) + self.center
    }

    pub fn from(&self) -> Point {
        self.at(0.0)
    }

    pub fn to(&self) -> Point {
        self.at(1.0)
    }

    pub fn to_cubic(&self) -> ElipArcCubicIter {
        ElipArcCubicIter::new(*self)
    }

    pub fn flatten(&self, tr: Transform, flatness: Scalar) -> ElipArcFlattenIter {
        ElipArcFlattenIter::new(*self, tr, flatness)
    }
}

/// Approximate arc with a sequnce of cubic bezier curves
///
/// [Drawing an elliptical arc using polylines, quadraticor cubic Bezier curves]
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
    ElipArc(ElipArc),
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Line(line) => line.fmt(f),
            Segment::Quad(quad) => quad.fmt(f),
            Segment::Cubic(cubic) => cubic.fmt(f),
            Segment::ElipArc(arc) => arc.fmt(f),
        }
    }
}

impl Segment {
    /// Split path segment into lines with specified tolerance.
    pub fn flatten(&self, tr: Transform, flatness: Scalar) -> SegmentFlattenIter {
        match self {
            Segment::Line(line) => SegmentFlattenIter::Line(line.flatten(tr)),
            Segment::Quad(quad) => {
                let cubic: Cubic = From::from(*quad);
                SegmentFlattenIter::Cubic(cubic.flatten(tr, flatness))
            }
            Segment::Cubic(cubic) => SegmentFlattenIter::Cubic(cubic.flatten(tr, flatness)),
            Segment::ElipArc(arc) => SegmentFlattenIter::ElipArc(arc.flatten(tr, flatness)),
        }
    }

    pub fn from(&self) -> Point {
        match self {
            Segment::Line(line) => line.from(),
            Segment::Quad(quad) => quad.from(),
            Segment::Cubic(cubic) => cubic.from(),
            Segment::ElipArc(arc) => arc.from(),
        }
    }

    pub fn to(&self) -> Point {
        match self {
            Segment::Line(line) => line.to(),
            Segment::Quad(quad) => quad.to(),
            Segment::Cubic(cubic) => cubic.to(),
            Segment::ElipArc(arc) => arc.to(),
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

impl From<ElipArc> for Segment {
    fn from(arc: ElipArc) -> Self {
        Self::ElipArc(arc)
    }
}

pub enum SegmentFlattenIter {
    Line(LineFlattenIter),
    Cubic(CubicFlattenIter),
    ElipArc(ElipArcFlattenIter),
}

impl Iterator for SegmentFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Line(line) => line.next(),
            Self::Cubic(cubic) => cubic.next(),
            Self::ElipArc(arc) => arc.next(),
        }
    }
}

#[derive(Debug, Clone)]
struct SubPath {
    segments: Vec<Segment>,
    closed: bool,
}

impl SubPath {
    fn flatten<'a>(
        &'a self,
        tr: Transform,
        flatness: Scalar,
        close: bool,
    ) -> impl Iterator<Item = Line> + 'a {
        let last = if self.closed || close {
            self.to()
                .and_then(|p1| self.from().map(|p2| Line([p1, p2]).transform(tr)))
        } else {
            None
        };

        self.segments
            .iter()
            .flat_map(move |segment| segment.flatten(tr, flatness))
            .chain(last)
    }

    fn from(&self) -> Option<Point> {
        self.segments.first().map(Segment::from)
    }

    fn to(&self) -> Option<Point> {
        self.segments.last().map(Segment::to)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Debug, Clone)]
pub struct Path {
    subpaths: Vec<SubPath>,
}

impl Path {
    pub fn flatten_simple<'a>(
        &'a self,
        tr: Transform,
        flatness: Scalar,
        close: bool,
    ) -> impl Iterator<Item = Line> + 'a {
        self.subpaths
            .iter()
            .flat_map(move |subpath| subpath.flatten(tr, flatness, close))
    }

    pub fn flatten(&self, tr: Transform, flatness: Scalar, close: bool) -> PathFlattenIter {
        PathFlattenIter::new(self, tr, flatness, close)
    }

    pub fn rasterize(&self, tr: Transform, fill_rule: FillRule) -> Option<SurfaceOwned<Scalar>> {
        // flatten all curves
        let lines: Vec<Line> = timeit("[flatten]", || self.flatten(tr, FLATNESS, true).collect());
        if lines.is_empty() {
            return None;
        }
        debug!("[lines]: {}", lines.len());

        // determine size of output layer
        let (min_x, min_y, max_x, max_y) = timeit("[size]", || {
            lines.iter().fold(
                (INFINITY, INFINITY, NEG_INFINITY, NEG_INFINITY),
                |(min_x, min_y, max_x, max_y), Line([p0, p1])| {
                    (
                        min_x.min(p0.x().min(p1.x())),
                        min_y.min(p0.y().min(p1.y())),
                        max_x.max(p0.x().max(p1.x())),
                        max_y.max(p0.y().max(p1.y())),
                    )
                },
            )
        });
        // one pixel broder to account for anti-aliasing
        let x = min_x.floor() as i32 - 1;
        let y = min_y.floor() as i32 - 1;
        let width = max_x.ceil() as i32 - x + 1;
        let height = max_y.ceil() as i32 - y + 1;

        if width == 0 || height == 0 {
            return None;
        }

        // calculate signed coverage
        let mut surf: SurfaceOwned<Scalar> = timeit("[alloc]", || {
            SurfaceOwned::new(height as usize, width as usize)
        });
        timeit("[coverage]", || {
            let offset = Transform::default().translate(-x as Scalar, -y as Scalar);
            for line in lines {
                rasterize_line(&mut surf, line.transform(offset));
            }
        });

        // cummulative sum over rows
        timeit("[mask]", || coverage_to_mask(&mut surf, fill_rule));

        Some(surf)
    }
}

pub struct PathFlattenIter {
    stack: Vec<Result<Line, Cubic>>,
    flatness: Scalar,
}

impl PathFlattenIter {
    fn new(path: &Path, tr: Transform, flatness: Scalar, close: bool) -> Self {
        let size: usize = path
            .subpaths
            .iter()
            .map(|subpath| subpath.segments.len())
            .sum();
        let mut stack = Vec::new();
        stack.reserve(size * 2);
        for subpath in path.subpaths.iter().rev() {
            if subpath.closed || close {
                let line = subpath
                    .to()
                    .and_then(|p1| subpath.from().map(|p2| Ok(Line::new(p1, p2).transform(tr))));
                stack.extend(line);
            }
            for segment in subpath.segments.iter().rev() {
                match segment {
                    Segment::Line(line) => stack.push(Ok(line.transform(tr))),
                    Segment::Quad(quad) => stack.push(Err(From::from(quad.transform(tr)))),
                    Segment::Cubic(cubic) => stack.push(Err(cubic.transform(tr))),
                    Segment::ElipArc(arc) => {
                        stack.extend(arc.to_cubic().map(|cubic| Err(cubic.transform(tr))))
                    }
                }
            }
        }

        Self {
            flatness: 16.0 * flatness * flatness,
            stack,
        }
    }
}

impl Iterator for PathFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stack.pop() {
                None => return None,
                Some(Ok(line)) => return Some(line),
                Some(Err(cubic)) if cubic.flatness() < self.flatness => {
                    let Cubic([p0, _p1, _p2, p3]) = cubic;
                    return Some(Line([p0, p3]));
                }
                Some(Err(cubic)) => {
                    let (c0, c1) = cubic.split();
                    self.stack.push(Err(c1));
                    self.stack.push(Err(c0));
                }
            }
        }
    }
}

impl FromStr for Path {
    type Err = PathParseError;

    fn from_str(text: &str) -> Result<Path, Self::Err> {
        let mut parser = PathParser::new(text);
        let mut segments = Vec::new();
        let mut subpaths = Vec::new();
        while !parser.is_eof() {
            match parser.parse_segment()? {
                Ok(segment) => segments.push(segment),
                Err(closed) if !segments.is_empty() => subpaths.push(SubPath {
                    closed,
                    segments: std::mem::replace(&mut segments, Vec::new()),
                }),
                _ => (),
            }
            parser.parse_separators()?;
        }
        if !segments.is_empty() {
            subpaths.push(SubPath {
                closed: false,
                segments: std::mem::replace(&mut segments, Vec::new()),
            })
        }
        Ok(Path { subpaths })
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
    // start of the current subpath
    start: Option<Point>,
    // previous segment
    prev_seg: Option<Segment>,
    // previous command
    prev_cmd: Option<u8>,
}

impl<'a> PathParser<'a> {
    fn new(text: &'a str) -> PathParser<'a> {
        Self {
            text: text.as_ref(),
            offset: 0,
            start: None,
            prev_seg: None,
            prev_cmd: None,
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

    fn position(&self) -> Result<Point, PathParseError> {
        self.prev_seg
            .map(|segment| segment.to())
            .ok_or_else(|| self.error("position: current position is missing"))
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

    fn parse_point(&mut self, is_relative: bool) -> Result<Point, PathParseError> {
        let x = self.parse_scalar()?;
        let y = self.parse_scalar()?;
        if is_relative {
            Ok(Point([x, y]) + self.position()?)
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
                self.prev_cmd = Some(cmd);
                Ok(cmd)
            }
            _ => match self.prev_cmd {
                Some(cmd) => Ok(cmd),
                None => Err(self.error(format!("parse_cmd: command expected `{}`", cmd))),
            },
        }
    }

    fn parse_segment(&mut self) -> Result<Result<Segment, bool>, PathParseError> {
        self.parse_separators()?;
        let cmd = self.parse_cmd()?;
        let seg: Segment = match cmd {
            b'M' | b'm' => {
                let is_relative = cmd == b'm';
                let p = self.parse_point(is_relative && self.prev_seg.is_some())?;
                self.prev_seg = Some(Line([Point([0.0, 0.0]), p]).into());
                self.prev_cmd = Some(if is_relative { b'l' } else { b'L' });
                self.start = Some(p);
                return Ok(Err(false));
            }
            b'L' | b'l' => {
                let is_relative = cmd == b'l';
                let p0 = self.position()?;
                let p1 = self.parse_point(is_relative)?;
                Line([p0, p1]).into()
            }
            b'V' | b'v' => {
                let p0 = self.position()?;
                let y = self.parse_scalar()?;
                let p1 = if cmd == b'v' {
                    Point([p0.x(), p0.y() + y])
                } else {
                    Point([p0.x(), y])
                };
                Line([p0, p1]).into()
            }
            b'H' | b'h' => {
                let p0 = self.position()?;
                let x = self.parse_scalar()?;
                let p1 = if cmd == b'h' {
                    Point([p0.x() + x, p0.y()])
                } else {
                    Point([x, p0.y()])
                };
                Line([p0, p1]).into()
            }
            b'C' | b'c' => {
                let is_relative = cmd == b'c';
                let p0 = self.position()?;
                let p1 = self.parse_point(is_relative)?;
                let p2 = self.parse_point(is_relative)?;
                let p3 = self.parse_point(is_relative)?;
                Cubic([p0, p1, p2, p3]).into()
            }
            b'S' | b's' => {
                let is_relative = cmd == b's';
                let p0 = self.position()?;
                let p1 = match self.prev_seg {
                    Some(Segment::Cubic(cubic)) => cubic.smooth(),
                    _ => p0,
                };
                let p2 = self.parse_point(is_relative)?;
                let p3 = self.parse_point(is_relative)?;
                Cubic([p0, p1, p2, p3]).into()
            }
            b'Q' | b'q' => {
                let is_relative = cmd == b'q';
                let p0 = self.position()?;
                let p1 = self.parse_point(is_relative)?;
                let p2 = self.parse_point(is_relative)?;
                Quad([p0, p1, p2]).into()
            }
            b'T' | b't' => {
                let is_relative = cmd == b't';
                let p0 = self.position()?;
                let p1 = match self.prev_seg {
                    Some(Segment::Quad(quad)) => quad.smooth(),
                    _ => p0,
                };
                let p2 = self.parse_point(is_relative)?;
                Quad([p0, p1, p2]).into()
            }
            b'A' | b'a' => {
                let is_relative = cmd == b'a';
                let src = self.position()?;
                let rx = self.parse_scalar()?;
                let ry = self.parse_scalar()?;
                let x_axis_rot = self.parse_scalar()?;
                let large_flag = self.parse_flag()?;
                let sweep_flag = self.parse_flag()?;
                let dst = self.parse_point(is_relative)?;
                ElipArc::new_param(src, dst, rx, ry, x_axis_rot, large_flag, sweep_flag).into()
            }
            b'Z' | b'z' => {
                if let Some(start) = self.start {
                    self.prev_seg = Some(Line([Point([0.0, 0.0]), start]).into());
                }
                self.prev_cmd = None;
                return Ok(Err(true));
            }
            _ => return Err(self.error(format!("parse_segment: invalid command: {}", cmd))),
        };
        self.prev_seg = Some(seg);
        Ok(Ok(seg))
    }
}

fn linear_to_srgb(value: Scalar) -> Scalar {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/*
fn srgb_to_linear(value: Scalar) -> Scalar {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.5)
    }
}

#[derive(Clone, Copy)]
pub struct RGBA {
    rgba: [Scalar; 4],
}
*/

trait Color {
    fn to_rgb(&self) -> [u8; 3];
    fn to_rgba(&self) -> [u8; 4];
}

impl Color for Scalar {
    fn to_rgb(&self) -> [u8; 3] {
        let color = (linear_to_srgb(1.0 - *self) * 255.0).round() as u8;
        [color; 3]
    }

    fn to_rgba(&self) -> [u8; 4] {
        let color = (linear_to_srgb(1.0 - *self) * 255.0).round() as u8;
        [color; 4]
    }
}

fn rasterize_line(mut surf: impl SurfaceMut<Item = Scalar>, line: Line) {
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
        // calculate coverage by the line left of `x == 0.0`
        rasterize_line(surf.view_mut(.., ..), vertical);
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
    for y in y..min(shape.height, p1.y().ceil() as usize) {
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

pub fn coverage_to_mask(mut surf: impl SurfaceMut<Item = Scalar>, fill_rule: FillRule) {
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

fn surf_to_ppm<S, W>(surf: S, mut w: W) -> Result<(), std::io::Error>
where
    S: Surface,
    S::Item: Color,
    W: Write,
{
    write!(w, "P6 {} {} 255 ", surf.width(), surf.height())?;
    for color in surf.iter() {
        w.write_all(&color.to_rgb())?;
    }
    Ok(())
}

fn surf_to_png<S, W>(surf: S, w: W) -> Result<(), png::EncodingError>
where
    S: Surface,
    S::Item: Color,
    W: Write,
{
    let mut encoder = png::Encoder::new(w, surf.width() as u32, surf.height() as u32);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut stream_writer = writer.stream_writer();
    for color in surf.iter() {
        stream_writer.write_all(&color.to_rgba())?;
    }
    stream_writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_approx_eq {
        ( $v0:expr, $v1: expr ) => {{
            assert!(($v0 - $v1).abs() < EPSILON, "{} != {}", $v0, $v1);
        }};
    }

    #[test]
    fn test_rasterize_line() {
        let mut surf = SurfaceOwned::new(2, 5);

        // line convers many columns but just one row
        rasterize_line(&mut surf, Line::new((0.5, 1.0), (3.5, 0.0)));
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
        rasterize_line(&mut surf, Line::new((-1.0, 0.0), (1.0, 1.0)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 3.0 / 4.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 / 4.0);
        surf.clear();

        // multiple rows diag
        rasterize_line(&mut surf, Line::new((0.0, -0.5), (2.0, 1.5)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 - 2.0 / 8.0);
        assert_approx_eq!(*surf.get(0, 2).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 1).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 2).unwrap(), 0.5 - 1.0 / 8.0);
        surf.clear();

        // multiple rows vertical
        rasterize_line(&mut surf, Line::new((0.5, 0.5), (0.5, 1.75)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 1.0 / 4.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 / 4.0);
        assert_approx_eq!(*surf.get(1, 0).unwrap(), 3.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 1).unwrap(), 3.0 / 8.0);
        surf.clear();
    }

    #[test]
    fn test_cubic_bbox() {
        let curve = Cubic::new((106.0, 0.0), (0.0, 100.0), (382.0, 216.0), (324.0, 14.0));
        let bbox = curve.bbox();
        assert!((bbox.x() - 87.308).abs() < 0.001);
        assert!(bbox.y().abs() < 0.001);
        assert!((bbox.width() - 242.724).abs() < 0.001);
        assert!((bbox.height() - 125.140).abs() < 0.001);
    }
}

// ------------------------------------------------------------------------------
// Playground
// ------------------------------------------------------------------------------
pub type Error = Box<dyn std::error::Error>;

pub fn path_load<P: AsRef<std::path::Path>>(path: P) -> Result<Path, Error> {
    use std::{fs::File, io::Read};
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(timeit("[parse]", || Path::from_str(&contents))?)
}

pub const SQUIRREL: &str = "M12 1C9.79 1 8 2.31 8 3.92c0 1.94.5 3.03 0 6.08 0-4.5-2.77-6.34-4-6.34.05-.5-.48-.66-.48-.66s-.22.11-.3.34c-.27-.31-.56-.27-.56-.27l-.13.58S.7 4.29 .68 6.87c.2.33 1.53.6 2.47.43.89.05.67.79.47.99C2.78 9.13 2 8 1 8S0 9 1 9s1 1 3 1c-3.09 1.2 0 4 0 4H3c-1 0-1 1-1 1h6c3 0 5-1 5-3.47 0-.85-.43-1.79 -1-2.53-1.11-1.46.23-2.68 1-2 .77.68 3 1 3-2 0-2.21-1.79-4-4-4zM2.5 6 c-.28 0-.5-.22-.5-.5s.22-.5.5-.5.5.22.5.5-.22.5-.5.5z";

pub const VERIFIED: &str = "M7.67 14.72H8.38L10.1 13H12.5L13 12.5V10.08L14.74 8.36004V7.65004L13.03 5.93004V3.49004L12.53 3.00004H10.1L8.38 1.29004H7.67L6 3.00004H3.53L3 3.50004V5.93004L1.31 7.65004V8.36004L3 10.08V12.5L3.53 13H6L7.67 14.72ZM6.16 12H4V9.87004L3.88 9.52004L2.37 8.00004L3.85 6.49004L4 6.14004V4.00004H6.16L6.52 3.86004L8 2.35004L9.54 3.86004L9.89 4.00004H12V6.14004L12.17 6.49004L13.69 8.00004L12.14 9.52004L12 9.87004V12H9.89L9.51 12.15L8 13.66L6.52 12.14L6.16 12ZM6.73004 10.4799H7.44004L11.21 6.71L10.5 6L7.09004 9.41991L5.71 8.03984L5 8.74984L6.73004 10.4799Z";

pub const NOMOVE: &str = "M50,100 0,50 25,25Z L100,50 75,25Z";
pub const STAR: &str = "M50,0 21,90 98,35 2,35 79,90z M110,0 h90 v90 h-90 z M130,20 h50 v50 h-50 zM210,0  h90 v90 h-90 z M230,20 v50 h50 v-50 z";
pub const ARCS: &str = "M600,350 l 50,-25a80,60 -30 1,1 50,-25 l 50,-25a25,50 -30 0,1 50,-25 l 50,-25a25,75 -30 0,1 50,-25 l 50,-25a25,100 -30 0,1 50,-25 l 50,-25";
pub const TEST: &str = "M-20.0,-10.0 L-10.0,-20.0 L20.0,10.0 L10.0,20.0 Z";

pub fn timeit<F: FnOnce() -> R, R>(msg: &str, f: F) -> R {
    let start = std::time::Instant::now();
    let result = f();
    debug!("{} {:?}", msg, start.elapsed());
    result
}

fn main() -> Result<(), Error> {
    env_logger::init();

    // let path = Path::from_str(SQUIRREL)?;
    // let tr = Transform::default().scale(12.0, 12.0);

    let path = path_load("material-big.path")?;
    let tr = Transform::default();

    let mask = timeit("[rasterize]", || path.rasterize(tr, FillRule::EvenOdd)).unwrap();

    // let path = Path::from_str(VERIFIED)?;
    // let tr = Transform::default()
    //     .scale(12.0, 12.0)
    //     .rotate_around(0.523598, Point::new(-8.0, -8.0))
    //     .translate(-1.0, -1.0);
    // let mut mask = SurfaceOwned::new(300, 300);
    // for line in path.flatten(tr, FLATNESS, true) {
    //     rasterize_line(&mut mask, line);
    // }
    // coverage_to_mask(&mut mask, FillRule::EvenOdd);

    println!("{:?}", mask.shape());
    if false {
        let mut image = std::io::BufWriter::new(std::fs::File::create("rasterize.png")?);
        timeit("[save:png]", || surf_to_png(&mask, &mut image))?;
    } else {
        let mut image = std::io::BufWriter::new(std::fs::File::create("rasterize.ppm")?);
        timeit("[save:ppm]", || surf_to_ppm(&mask, &mut image))?;
    }

    Ok(())
}
