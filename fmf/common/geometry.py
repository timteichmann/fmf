# FreeMolecularFlow - common - geometry
# (c) 2024 Tim Teichmann

import numpy as np

EPSILON = 1e-6

def cell_bounds_to_centers(xmin, xmax, nx):
    dx = (xmax - xmin)/nx
    xc = np.linspace(xmin + dx/2.0, xmax - dx/2.0, nx)
    return xc

def centers_to_grid(xc, yc):
    xgrid = []
    ygrid = []
    for y in yc:
        x_xgrid = []
        x_ygrid = []
        for x in xc:
            x_xgrid.append(x)
            x_ygrid.append(y)
        xgrid.append(x_xgrid)
        ygrid.append(x_ygrid)
    xgrid = np.array(xgrid)
    ygrid = np.array(ygrid)
    return (xgrid, ygrid)

def mirror_grid(grid, sign=False):
    grid = np.array(grid)
    flipped = np.flip(grid, 0)
    if sign:
        flipped = -flipped
    grid = np.concatenate((flipped, grid))
    return grid

class Ray2d:
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype="d")
        self.p2 = np.array(p2, dtype="d")
        # connection vector
        self.v = self.p2 - self.p1
        self.v /= np.sqrt(self.v[0]**2 + self.v[1]**2)
        # normal vector
        self.n = np.array((-(self.p2[1]-self.p1[1]), self.p2[0]-self.p1[0]))
        self.n /= np.sqrt(self.n[0]**2 + self.n[1]**2)

        # rays are infinite
        self.l = None

    def intersect(self, other):
        # check if self intersects other
        # check if parallel
        if (self.v[0]*other.v[1] - self.v[1]*other.v[0]) == 0.0:
            return False
        # find intersection point p
        # p = self.p1 + u*self.v
        # p = other.p1 + v*line.v
        # solve for u, v e.g. using sympy
        u = (other.v[0]*(self.p1[1] - other.p1[1]) - other.v[1]*(self.p1[0] - other.p1[0]))/(self.v[0]*other.v[1] - self.v[1]*other.v[0])
        v = (self.v[0]*(self.p1[1] - other.p1[1]) - self.v[1]*(self.p1[0] - other.p1[0]))/(self.v[0]*other.v[1] - self.v[1]*other.v[0])

        p = np.array([self.p1[0] + u*self.v[0], self.p1[1] + u*self.v[1]])

        # if one or both are finite check if point lies on the objects
        if not self.l is None:
            if u-EPSILON < 0.0 or u+EPSILON > self.l:
                return False
        if not other.l is None:
            if v-EPSILON < 0.0 or v+EPSILON > other.l:
                return False
        return p

    def in_view(self, p):
        # check if p lies on positive side of segment (as defined by direction
        # of normal vector)
        # p = u*v + w*n
        # w > 0 -> in view, else not
        w = (p[0]*self.v[1] - p[1]*self.v[0] - self.p1[0]*self.v[1] + self.p1[1]*self.v[0])/(self.n[0]*self.v[1] - self.n[1]*self.v[0])
        if w >= 0:
            return True
        return False

class Segment2d(Ray2d):
    def __init__(self, p1, p2):
        super().__init__(p1, p2)
        # segments have a finite length
        self.l = np.sqrt((self.p2[0]-self.p1[0])**2 + (self.p2[1] - self.p1[1])**2)

        # inclination angle
        if self.v[0] == 0.0:
            self.alpha = np.radians(90)
        else:
            self.alpha = np.arctan2(self.v[1], self.v[0])
        self.sinalpha = np.sin(self.alpha)
        self.cosalpha = np.cos(self.alpha)

        # (virtual) point on x-axis (y = 0)
        # but integration currently expects this
        self.c = self.p1 - self.p1[1]*self.v/self.v[1]

        # local integration range
        self.w = np.array((self.point_to_w(self.p1), self.point_to_w(self.p2)))

    def w_to_x(self, w):
        return self.c[0] + self.cosalpha * w

    def w_to_y(self, w):
        return self.sinalpha * w

    def point_to_w(self, p):
        p = np.array(p, dtype="d")
        # w = (p[0] - self.c[0])/self.cosalpha
        # w = p[1] / self.sinalpha
        w = p[1] / self.sinalpha
        return w

    def point_on_me(self, p):
        # check if p lies on me
        d1 = p - self.p1
        d2 = self.p2 - p
        if np.abs(np.sqrt(d1[0]**2 + d1[1]**2) + np.sqrt(d2[0]**2 + d2[1]**2) - self.l) <= EPSILON:
            return True
        return False

class Shape2d():
    def __init__(self, segments):
        self.segments = segments
        # check if closed, and normals facing out and collect points
        p2_ = self.segments[-1].p2
        orientation = 0
        self.points = []
        for s in self.segments:
            if not np.all(s.p1 == p2_):
                raise ValueError("Shape2d is not closed")
            orientation += (s.p2[0] - s.p1[0])*(s.p2[1] + s.p1[1])
            p2_ = s.p2
            self.points.append(s.p1)
        if orientation < 0:
            raise ValueError("Shape2d is not clockwise")

        # bounding box
        self.bb = None
        for s in self.segments:
            if self.bb is None:
                self.bb = np.array([
                    (min(s.p1[0],s.p2[0]), min(s.p1[1],s.p2[1])),
                    (max(s.p1[0],s.p2[0]), max(s.p1[1],s.p2[1]))
                ])
            else:
                self.bb = np.array([
                    (min(s.p1[0],s.p2[0],self.bb[0][0]), min(s.p1[1],s.p2[1],self.bb[0][1])),
                    (max(s.p1[0],s.p2[0],self.bb[1][0]), max(s.p1[1],s.p2[1],self.bb[1][1]))
                ])

    def intersect_number(self, other):
        # check if line intersects any of my segments
        n = 0
        for s in self.segments:
            inter = s.intersect(other)
            if not inter is False:
                n += 1
        return n

    def on_circumference(self, p):
        p = np.array(p, dtype="d")
        # check if the point p lies on the circumference of me
        for s in self.segments:
            if s.point_on_me(p):
                return True
        return False

    def inside(self, p):
        p = np.array(p, dtype="d")
        # first check if point is on circumference (not inside!)
        if self.on_circumference(p):
            return False
        # check if point p is inside me
        # Segment from p to point outside bounding box
        r = Segment2d(p,self.bb[0]-np.array([1,1]))
        if self.intersect_number(r) % 2 == 1:
            return True
        return False

    def point_segment_los(self, p, s):
        p = np.array(p, dtype="d")
        los1 = Segment2d(p, s.p1)
        los2 = Segment2d(p, s.p2)
        i1 = self.intersect_number(los1)
        i2 = self.intersect_number(los2)
        if i1 == 0 and i2 == 0:
            # free los
            return (los1.p2, los2.p2)
        elif i1 > 0 and i2 > 0:
            # completely hidden los
            return (None, None)
        elif i1 > 0:
            # los1 partially hidden
            # draw rays through all my lines and p, correct los will have no
            # intersections with any other point
            for pi in self.points:
                ri = Ray2d(p, pi)
                pinter = s.intersect(ri)
                if pinter is False:
                    continue
                si = Segment2d(p, pinter)
                if self.intersect_number(si) == 0:
                    return (pinter, los2.p2)
            return (None, None) # why is this necessary?
        else:
            # los2 partially hidden
            # draw rays through all my lines and p, correct los will have no
            # intersections with any other point
            for pi in self.points:
                ri = Ray2d(p, pi)
                pinter = s.intersect(ri)
                if pinter is False:
                    continue
                si = Segment2d(p, pinter)
                if self.intersect_number(si) == 0:
                    return (los1.p2, pinter)
            return (None, None) # why is this necessary?
