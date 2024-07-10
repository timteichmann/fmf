# FreeMolecularFlow
# (c) 2024 Tim Teichmann

# The underlying mathematical derivation is presented in these works by Cai et al.:
#   [1] C. Cai and I. D. Boyd, “Theoretical and Numerical Study of Free
#   Molecular-Flow Problems,” Journal of Spacecraft and Rockets, vol. 44,
#   no. 3, pp. 619–624, May 2007, doi: 10.2514/1.25893.
#   [2] C. Cai and I. D. Boyd, “Collisionless Gas Expanding into Vacuum,”
#   Journal of Spacecraft and Rockets, vol. 44, no. 6, pp. 1326–1330,
#   Dec. 2007, doi: 10.2514/1.32173.
#   [3] K. Khasawneh, H. Liu, and C. Cai, “Highly rarefied two-dimensional
#   jet impingement on a flat plate,” Physics of Fluids, vol. 22, no. 11,
#   p. 117101, Nov. 2010, doi: 10.1063/1.3490409.
#   [4] L. Wang and C. Cai, “Gaseous plume flows in space propulsion,”
#   Chinese Journal of Aeronautics, vol. 26, no. 3, pp. 522–528, Jun. 2013,
#   doi: 10.1016/j.cja.2013.04.049.
#   [5] C. Cai and X. He, “Detailed flowfield and surface properties for high
#   Knudsen number planar jet impingement at an inclined flat plate,” Physics
#   of Fluids, vol. 28, no. 5, p. 056103, May 2016, doi: 10.1063/1.4948365.
# The derivation, application and implementation for arbitrary convex shapes
# with an arbitrary number of surfaces was established by myself based on the
# implementation for inclined surfaces by Cai and He.

import os
import copy
import ctypes
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy import LowLevelCallable

# local imports
from fmf.common import constants
from fmf.common import geometry

class FreeMolecularFlow:
    def __init__(self, d, molmass, nrho0, vel0, temp0):
        # diameter of the slit/orifice
        # [m]
        self.d = d
        # molecular mass of the gas
        # [kg/mol] (but provided in g/mol)
        self.molmass = molmass*1e-3
        # number density at the slit/orifice
        # [1/m^3]
        self.nrho0 = nrho0
        # bulk velocity at the slit/orifice
        # [m/s]
        self.vel0 = vel0
        # gas temperature at the slit/orifice
        # [K]
        self.temp0 = temp0

        # specific gas constant
        # [J/kg/K]
        self.R = constants.GAS_CONSTANT/self.molmass

        # most probable speed
        # [m/s]
        self.velth = np.sqrt(2.0*self.R*self.temp0)

        # dimensionless velocity at the slit/orifice
        # [1]
        self.s0 = self.vel0/self.velth # = sqrt(beta)*vel0

        # collected field values over grid
        self.nrho_ = None
        self.velx_ = None
        self.vely_ = None
        self.temp_ = None
        self.press_ = None

    def clone(self):
        return copy.deepcopy(self)

    def build_lib_func(self, lib_name, constants):
        base_path = os.path.dirname(os.path.realpath(__file__))
        lib = ctypes.CDLL(os.path.abspath(base_path + "/libs/" + lib_name))
        lib.f.restype = ctypes.c_double
        if constants:
            lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
            ptr_to_buffer=(ctypes.c_double*len(constants))(*constants)
            user_data = ctypes.cast(ptr_to_buffer, ctypes.c_void_p)
            lib_func = LowLevelCallable(lib.f, user_data)
        else:
            lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))
            lib_func = LowLevelCallable(lib.f)
        return lib_func

    def integrate_(self, integrand, t1, t2, points = None):
        if points is None:
            return quad(integrand, t1, t2, epsabs = 0)[0]
        else:
            return quad(integrand, t1, t2, points = points, epsabs = 0)[0]

    def field_xy_(self, f, x, y):
        findx = np.isclose(self.xgrid, x)
        findy = np.isclose(self.ygrid, y)
        find = np.logical_and(findx, findy)
        i, j = np.where(find)
        i = i[0]
        j = j[0]
        return f[i][j]

    def field_(self, f):
        values = []
        for y in self.yc:
            x_values = []
            for x in self.xc:
                x_values.append(f(x, y))
            values.append(x_values)
        return np.array(values)

    def nrho_field(self):
        if self.nrho_ is None:
            # set and return the field
            self.nrho_ = self.field_(self.calc_nrho)
        return self.nrho_

    def velx_field(self):
        if self.velx_ is None:
            # set and return the field
            self.velx_ = self.field_(self.calc_velx)
        return self.velx_

    def vely_field(self):
        if self.vely_ is None:
            # set and return the field
            self.vely_ = self.field_(self.calc_vely)
        return self.vely_

    def temp_field(self):
        if self.temp_ is None:
            # set and return the field
            self.temp_ = self.field_(self.calc_temp)
        return self.temp_

    def press_field(self):
        if self.press_ is None:
            # set and return the field
            self.press_ = self.field_(self.calc_press)
        return self.press_

    def calc_fields(self):
        # calculate all fields in the correct order
        self.nrho_field()
        self.velx_field()
        self.vely_field()
        self.temp_field()
        self.press_field()

    def clear_fields(self):
        self.nrho_ = None
        self.velx_ = None
        self.vely_ = None
        self.temp_ = None
        self.press_ = None

class FreeMolecularFlow2d(FreeMolecularFlow):
    def __init__(self, d, molmass, nrho0, vel0, temp0, xmin, xmax, nx, ymin, ymax, ny):
        super().__init__(d, molmass, nrho0, vel0, temp0)
        # global coordinate system:
        self.xc = geometry.cell_bounds_to_centers(xmin, xmax, nx)
        self.yc = geometry.cell_bounds_to_centers(ymin, ymax, ny)

        # xc, yc grids with equal layout to fields
        self.xgrid, self.ygrid = geometry.centers_to_grid(self.xc, self.yc)

        # local coordinate system can be transformed (translated, rotated)
        self.translation_ = None
        self.rotation_ = None

        # integrand libraries (compiled C for speedup):
        self.integrand_nrho = self.build_lib_func("2d_nrho.so", [self.s0])
        self.integrand_velx1 = self.build_lib_func("2d_velx1.so", [self.s0])
        self.integrand_velx2 = self.build_lib_func("2d_velx2.so", [self.s0])
        self.integrand_temp = self.build_lib_func("2d_temp.so", [self.s0])

    def global_to_local_xy_(self, x, y):
        # default, no manipulation, local identical to global
        xloc = x
        yloc = y
        # first rotate, then translate
        if not self.translation_ is None:
            # translate
            xloc = xloc - self.translation_[0]
            yloc = yloc - self.translation_[1]
        if not self.rotation_ is None:
            # rotate
            xloc_ = xloc
            yloc_ = yloc
            a = self.rotation_
            xloc = xloc_*np.cos(a) + yloc_*np.sin(a)
            yloc = - xloc_*np.sin(a) + yloc_*np.cos(a)

        return (xloc, yloc)

    def local_to_global_xy_(self, x, y):
        # default, no manipulation, global identical to local
        xglob = x
        yglob = y
        # first rotate, then translate
        if not self.rotation_ is None:
            # rotate
            xglob_ = xglob
            yglob_ = yglob
            a = -self.rotation_
            xglob = xglob_*np.cos(a) + yglob_*np.sin(a)
            yglob = - xglob_*np.sin(a) + yglob_*np.cos(a)
        if not self.translation_ is None:
            # translate
            xglob = xglob + self.translation_[0]
            yglob = yglob + self.translation_[1]

        return (xglob, yglob)

    def translate(self, d):
        # clear fields
        recalculate = False
        if not self.nrho_ is None:
            recalculate = True
        self.clear_fields()
        self.translation_ = np.array(d)
        if recalculate:
            self.calc_fields()

    def rotate(self, a):
        # clear fields
        recalculate = False
        if not self.nrho_ is None:
            recalculate = True
        self.clear_fields()
        self.rotation_ = a
        if recalculate:
            self.calc_fields()

    def slit(self, p1, p2):
        # clear fields
        recalculate = False
        if not self.nrho_ is None:
            recalculate = True
        p1 = np.array(p1)
        p2 = np.array(p2)
        v = p2 - p1
        d = np.sqrt(v[0]**2 + v[1]**2)
        # set new slit width
        self.d = d
        # calculate necessary rotation and translation
        # 1. find center
        c = p1 + 0.5*v
        # move points in local center
        self.translation_ = c
        # find the rotation angle
        p1_ = p1 - c
        x1 = p1_[0]
        y1 = p1_[1]
        # angle
        # rotate by 90° clockwise
        p1_[0] = x1*np.cos(-np.pi/2) + y1*np.sin(-np.pi/2)
        p1_[1] = -x1*np.sin(-np.pi/2) + y1*np.cos(-np.pi/2)
        # find angle to axis
        a = np.arctan2(p1_[1],p1_[0])
        self.rotation_ = a

        if recalculate:
            self.calc_fields()

    def theta_(self, x, y, sgn):
        dx = x
        dy = y - sgn*0.5*self.d
        return np.arctan(dy/dx)

    def theta1_(self, x, y):
        return self.theta_(x, y, 1.0)

    def theta2_(self, x, y):
        return self.theta_(x, y, -1.0)

    def calc_nrho(self, x, y):
        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0
        t1 = self.theta1_(x, y)
        t2 = self.theta2_(x, y)
        '''
        integrand = lambda t : (
              np.exp(-(self.s0*np.sin(t))**2)
            * np.cos(t)
            * erf(self.s0*np.cos(t))
        )
        '''
        # C library
        integrand = self.integrand_nrho
        return self.nrho0*(
              np.exp(-self.s0**2)/(2.0*np.pi)*(t2-t1)
            + 0.25 * (erf(self.s0*np.sin(t2)) - np.sign(t1)*erf(self.s0*np.sin(np.abs(t1))))
            + self.s0/(2.0*constants.SQRT_PI)*self.integrate_(integrand, t1, t2)
        )

    def calc_velx(self, x, y, nrhoxy = None):
        if nrhoxy is None:
            n = self.field_xy_(self.nrho_field(), x, y)/self.nrho0
        else:
            n = nrhoxy/self.nrho0

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0
        t1 = self.theta1_(x, y)
        t2 = self.theta2_(x, y)
        '''
        integrand1 = lambda t : (
              constants.SQRT_PI/2.0*np.exp((self.s0*np.cos(t))**2)*np.cos(t)
            * (1.0 + erf(self.s0*np.cos(t)))
        )
        '''
        # C library
        integrand1 = self.integrand_velx1
        '''
        integrand2 = lambda t : (
              (np.cos(t))**3*(1.0 + erf(self.s0*np.cos(t)))*np.exp((self.s0*np.cos(t))**2)
        )
        '''
        # C library
        integrand2 = self.integrand_velx2
        return self.velth*np.exp(-self.s0**2)/(2.0*n*np.pi)*(
              self.integrate_(integrand1, t1, t2)
            + self.s0*(t2 - t1)/2.0
            + self.s0*(np.sin(2.0*t2) - np.sin(2.0*t1))/4.0
            + self.s0**2*constants.SQRT_PI*self.integrate_(integrand2, t1, t2)
        )

    def calc_vely(self, x, y, nrhoxy = None):
        if nrhoxy is None:
            n = self.field_xy_(self.nrho_field(), x, y)/self.nrho0
        else:
            n = nrhoxy/self.nrho0

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0
        t1 = self.theta1_(x, y)
        t2 = self.theta2_(x, y)
        return self.velth/(4.0*constants.SQRT_PI*n)*(
              np.exp(-(self.s0*np.sin(t1))**2)*np.cos(t1)
            * (1.0 + erf(self.s0*np.cos(t1)))
            - np.exp(-(self.s0*np.sin(t2))**2)*np.cos(t2)
            * (1.0 + erf(self.s0*np.cos(t2)))
        )

    def calc_temp(self, x, y, nrhoxy = None, velxxy = None, velyxy = None):
        if nrhoxy is None:
            n = self.field_xy_(self.nrho_field(), x, y)/self.nrho0
        else:
            n = nrhoxy/self.nrho0
        if velxxy is None:
            velx = self.field_xy_(self.velx_field(), x, y)
        else:
            velx = velxxy
        if velyxy is None:
            vely = self.field_xy_(self.vely_field(), x, y)
        else:
            vely = velyxy

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0
        t1 = self.theta1_(x, y)
        t2 = self.theta2_(x, y)
        '''
        integrand = lambda t : (
              (2.0 + (self.s0*np.cos(t))**2)*np.cos(t)
            * np.exp((self.s0*np.cos(t))**2)
            * (1.0 + erf(self.s0*np.cos(t)))
        )
        '''
        # C library
        integrand = self.integrand_temp
        return -(velx**2 + vely**2)/(3.0*self.R) + np.exp(-self.s0**2)*self.temp0/(6.0*n*np.pi)*(
              (3.0 + self.s0**2)*(t2-t1)
            + 0.5*self.s0**2*(np.sin(2.0*t2) - np.sin(2.0*t1))
            + 2.0*constants.SQRT_PI*self.s0*self.integrate_(integrand, t1, t2)
        )

    def calc_press(self, x, y, nrhoxy = None, tempxy = None):
        if nrhoxy is None:
            nrho = self.field_xy_(self.nrho_field(), x, y)
        else:
            nrho = nrhoxy
        if tempxy is None:
            temp = self.field_xy_(self.temp_field(), x, y)
        else:
            temp = tempxy

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x < 0:
            return 0.0
        # ideal gas law:
        #   p = nrho*kB*T
        return nrho*constants.BOLTZ*temp

class FreeMolecularFlow2dNormalImpingement(FreeMolecularFlow2d):
    def __init__(self, d, molmass, nrho0, vel0, temp0, xmin, xmax, nx, ymin, ymax, ny, tempw, xw, ywmin, ywmax):
        super().__init__(d, molmass, nrho0, vel0, temp0, xmin, xmax, nx, ymin, ymax, ny)
        # wall properties:
        self.tempw = tempw

        # wall position & dimensions
        # in local coordinate system!
        self.xw = xw
        self.yw = np.array([ywmin, ywmax])

        # collected field values over grid
        self.nrho2_ = None
        self.velx2_ = None
        self.vely2_ = None
        self.temp2_ = None
        self.press2_ = None

    def calc_nrho2(self, x, y):
        nrho1 = self.field_xy_(self.nrho_field(), x, y)

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0

        def integrand(yw):
            # xw, yw are local, but functions below expect global
            xw_, yw_ = self.local_to_global_xy_(self.xw, yw)
            nrho1w = self.calc_nrho(xw_, yw_)
            velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
            nrhow = np.sqrt(2.0*np.pi/self.tempw/self.R)*nrho1w*velx1w
            return nrhow/((y-yw)**2 + (x-self.xw)**2)

        return (
              nrho1
            - (x - self.xw)/2.0/np.pi
            * self.integrate_(integrand, self.yw[0], self.yw[1])
        )

    def nrho2_field(self):
        if self.nrho2_ is None:
            # set and return the field
            self.nrho2_ = self.field_(self.calc_nrho2)
        return self.nrho2_

    def calc_velx2(self, x, y):
        nrho1 = self.field_xy_(self.nrho_field(), x, y)
        velx1 = self.field_xy_(self.velx_field(), x, y)
        nrho2 = self.field_xy_(self.nrho2_field(), x, y)

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0

        def integrand(yw):
            # xw, yw are local, but functions below expect global
            xw_, yw_ = self.local_to_global_xy_(self.xw, yw)
            nrho1w = self.calc_nrho(xw_, yw_)
            velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
            nrhow = np.sqrt(2.0*np.pi/self.tempw/self.R)*nrho1w*velx1w
            return nrhow/(((y-yw)**2 + (x-self.xw)**2)**(3/2))

        return 1.0/nrho2*(
              nrho1*velx1
            - (x - self.xw)**2
            * np.sqrt(self.R*self.tempw/8.0/np.pi)
            * self.integrate_(integrand, self.yw[0], self.yw[1])
        )

    def velx2_field(self):
        if self.velx2_ is None:
            # set and return the field
            self.velx2_ = self.field_(self.calc_velx2)
        return self.velx2_

    def calc_vely2(self, x, y):
        nrho1 = self.field_xy_(self.nrho_field(), x, y)
        vely1 = self.field_xy_(self.vely_field(), x, y)
        nrho2 = self.field_xy_(self.nrho2_field(), x, y)

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0

        def integrand(yw):
            # xw, yw are local, but functions below expect global
            xw_, yw_ = self.local_to_global_xy_(self.xw, yw)
            nrho1w = self.calc_nrho(xw_, yw_)
            velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
            nrhow = np.sqrt(2.0*np.pi/self.tempw/self.R)*nrho1w*velx1w
            return nrhow*(y-yw)/(((y-yw)**2 + (x-self.xw)**2)**(3/2))

        return 1.0/nrho2*(
              nrho1*vely1
            - (x - self.xw)
            * np.sqrt(self.R*self.tempw/8.0/np.pi)
            * self.integrate_(integrand, self.yw[0], self.yw[1])
        )

    def vely2_field(self):
        if self.vely2_ is None:
            # set and return the field
            self.vely2_ = self.field_(self.calc_vely2)
        return self.vely2_

    def calc_nrho2_wall(self, y):
        # xw is local, but functions below expect global
        xw_, _ = self.local_to_global_xy_(self.xw, 0)
        nrho1w = self.calc_nrho(xw_, y)
        velx1w = self.calc_velx(xw_, y, nrhoxy = nrho1w)
        nrhow = np.sqrt(np.pi/2.0/self.R/self.tempw)*nrho1w*velx1w
        return nrho1w + nrhow

    def calc_vely2_wall(self, y):
        # xw is local, but functions below expect global
        xw_, _ = self.local_to_global_xy_(self.xw, 0)
        nrho1w = self.calc_nrho(xw_, y)
        vely1w = self.calc_vely(xw_, y, nrhoxy = nrho1w)
        velx1w = self.calc_velx(xw_, y, nrhoxy = nrho1w)
        betaw = 1.0/(2.0*self.R*self.tempw)
        return vely1w/(1.0+np.sqrt(np.pi*betaw)*velx1w)

    def calc_temp2_near_wall(self, y):
        # xw is local, but functions below expect global
        xw_, _ = self.local_to_global_xy_(self.xw, 0)
        nrho1w = self.calc_nrho(xw_, y)
        velx1w = self.calc_velx(xw_, y, nrhoxy = nrho1w)
        vely1w = self.calc_vely(xw_, y, nrhoxy = nrho1w)
        temp1w = self.calc_temp(xw_, y, nrhoxy = nrho1w, velxxy = velx1w, velyxy = vely1w)
        nrho2w = self.calc_nrho2_wall(y)
        vely2w = self.calc_vely2_wall(y)
        nrhow = np.sqrt(2.0*np.pi/self.tempw/self.R)*nrho1w*velx1w
        return (
            - vely2w**2/3.0/self.R
            + nrhow/2.0/nrho2w*self.tempw
            + nrho1w/nrho2w*(temp1w + (velx1w**2 + vely1w**2)/3.0/self.R)
        )

    def calc_press2_near_wall(self, y):
        nrho2w = self.calc_nrho2_wall(y)
        temp2nw = self.calc_temp2_near_wall(y)
        return nrho2w*constants.BOLTZ*temp2nw

    def calc_temp2_wall(self, y):
        # xw is local, but functions below expect global
        xw_, _ = self.local_to_global_xy_(self.xw, 0)
        nrho1w = self.calc_nrho(xw_, y)
        velx1w = self.calc_velx(xw_, y, nrhoxy = nrho1w)
        nrho2w = self.calc_nrho2_wall(y)
        nrhow = np.sqrt(2.0*np.pi/self.tempw/self.R)*nrho1w*velx1w

        # global to local coordinates
        x = self.xw
        _, y = self.global_to_local_xy_(0, y)
        if x <= 0:
            return 0.0
        t1 = self.theta1_(x, y)
        t2 = self.theta2_(x, y)
        integrand = lambda t : (
              (3.0 + 2.0*(self.s0*np.cos(t))**2)*np.cos(t)
            * np.exp((self.s0*np.cos(t))**2)
            * (1.0 + erf(self.s0*np.cos(t)))
        )
        return nrhow*self.tempw/2.0/nrho2w + np.exp(-self.s0**2)*self.temp0/(2.0*np.pi)*self.nrho0/nrho2w*(
              (2.0 + self.s0**2)*(t2-t1)
            + 0.5*self.s0**2*(np.sin(2.0*t2) - np.sin(2.0*t1))
            + constants.SQRT_PI*self.s0*self.integrate_(integrand, t1, t2)
        )

    def calc_press2_wall(self, y):
        nrho2w = self.calc_nrho2_wall(y)
        temp2nw = self.calc_temp2_wall(y)
        return nrho2w*constants.BOLTZ*temp2nw

class FreeMolecularFlow2dImpingement(FreeMolecularFlow2d):
    def __init__(self, d, molmass, nrho0, vel0, temp0, xmin, xmax, nx, ymin, ymax, ny, tempw, xw1, yw1, xw2, yw2):
        super().__init__(d, molmass, nrho0, vel0, temp0, xmin, xmax, nx, ymin, ymax, ny)
        # wall properties:
        self.tempw = tempw

        # wall position & dimensions
        # in local coordinate system!
        self.pw1 = np.array((xw1, yw1))
        self.pw2 = np.array((xw2, yw2))

        # connection vector
        self.vw = self.pw2 - self.pw1
        # normal vector
        self.nw = np.array((-(self.pw2[1]-self.pw1[1]), self.pw2[0]-self.pw1[0]))
        # normalize
        self.nw /= np.sqrt(self.nw[0]**2 + self.nw[1]**2)
        # length
        self.lw = np.sqrt(self.vw[0]**2 + self.vw[1]**2)
        # (virtual) point on wall on slit centerline
        self.cw = self.pw1 - self.pw1[1]*self.vw/self.vw[1]
        self.cwx = self.cw[0]
        # width to both sides of (virtual) center point (can also be both on
        # one side)
        self.vw1 = self.pw1 - self.cw
        self.vw2 = self.pw2 - self.cw
        self.ww = np.array((np.sign(self.pw1[1])*np.sqrt(self.vw1[0]**2 + self.vw1[1]**2), np.sign(self.pw2[1])*np.sqrt(self.vw2[0]**2 + self.vw2[1]**2)))
        # inclination angle of the wall compared to slit centerline
        if self.vw[0] == 0.0:
            self.alpha = np.radians(90)
        else:
            self.alpha = np.arctan(self.vw[1]/self.vw[0])
        self.sinalpha = np.sin(self.alpha)
        self.cosalpha = np.cos(self.alpha)

        # collected field values over grid
        self.nrho2_ = None
        self.velx2_ = None
        self.vely2_ = None
        self.temp2_ = None
        self.press2_ = None

    def x_from_yw_(self, yw):
        return self.cwx + self.cosalpha * yw

    def y_from_yw_(self, yw):
        return self.sinalpha * yw

    def y_on_wall_(self, x):
        # x is local
        y = (self.pw1 + (x - self.pw1[0]) * self.vw/self.vw[0])[1]
        if x >= min(self.pw1[0], self.pw2[0]) and x <= max(self.pw1[0], self.pw2[0]):
            return (y, True)
        else:
            # still return virtual point for domain
            return (y, False)

    def in_domain_(self, x, y):
        # x, y are local
        # this does not include points that lie behind the wall (which could
        # potentially be reached by the flow from the slit though)
        yw = self.y_on_wall_(x)[0]
        if self.nw[1] >= 0:
            if y >= yw:
                return True
        else:
            if y <= yw:
                return True
        return False

    def calc_nrho2(self, x, y):
        nrho1 = self.field_xy_(self.nrho_field(), x, y)

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0
        elif not self.in_domain_(x, y):
            return 0.0

        def integrand(yw):
            # yw is inclined and local, functions expect global
            x_ = self.x_from_yw_(yw)
            y_ = self.y_from_yw_(yw)
            xw_, yw_ = self.local_to_global_xy_(x_, y_)
            nrho1w = self.calc_nrho(xw_, yw_)
            velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
            vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
            nrhow = (
                  np.sqrt(2.0*np.pi/self.tempw/self.R)
                * (nrho1w*velx1w*self.sinalpha - nrho1w*vely1w*self.cosalpha)
            )
            return (
                  nrhow
                / ((y-yw*self.sinalpha)**2 + (x-self.cwx-yw*self.cosalpha)**2)
                #/ (y**2 + (x-self.cwx)**2 + yw**2 - 2.0*y*yw*self.sinalpha - 2.0*(x-self.cwx)*yw*self.cosalpha)
            )

        return (
              nrho1
            + (y*self.cosalpha - (x - self.cwx)*self.sinalpha)/2.0/np.pi
            * self.integrate_(integrand, self.ww[0], self.ww[1])
        )

    def nrho2_field(self):
        if self.nrho2_ is None:
            # set and return the field
            self.nrho2_ = self.field_(self.calc_nrho2)
        return self.nrho2_

    def calc_velx2(self, x, y):
        nrho1 = self.field_xy_(self.nrho_field(), x, y)
        velx1 = self.field_xy_(self.velx_field(), x, y)
        nrho2 = self.field_xy_(self.nrho2_field(), x, y)

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0
        elif not self.in_domain_(x, y):
            return 0.0

        def integrand(yw):
            # yw is inclined and local, functions expect global
            x_ = self.x_from_yw_(yw)
            y_ = self.y_from_yw_(yw)
            xw_, yw_ = self.local_to_global_xy_(x_, y_)
            nrho1w = self.calc_nrho(xw_, yw_)
            velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
            vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
            nrhow = (
                  np.sqrt(2.0*np.pi/self.tempw/self.R)
                * (nrho1w*velx1w*self.sinalpha - nrho1w*vely1w*self.cosalpha)
            )
            return (
                  nrhow*(x - self.cwx - yw*self.cosalpha)
                / (((y-yw*self.sinalpha)**2 + (x-self.cwx-yw*self.cosalpha)**2)**(3/2))
            )

        return 1.0/nrho2*(
              nrho1*velx1
            + (y*self.cosalpha - (x - self.cwx)*self.sinalpha)
            * np.sqrt(self.R*self.tempw/8.0/np.pi)
            * self.integrate_(integrand, self.ww[0], self.ww[1])
        )

    def velx2_field(self):
        if self.velx2_ is None:
            # set and return the field
            self.velx2_ = self.field_(self.calc_velx2)
        return self.velx2_

    def calc_vely2(self, x, y):
        nrho1 = self.field_xy_(self.nrho_field(), x, y)
        vely1 = self.field_xy_(self.vely_field(), x, y)
        nrho2 = self.field_xy_(self.nrho2_field(), x, y)

        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        if x <= 0:
            return 0.0
        elif not self.in_domain_(x, y):
            return 0.0

        def integrand(yw):
            # yw is inclined and local, functions expect global
            x_ = self.x_from_yw_(yw)
            y_ = self.y_from_yw_(yw)
            xw_, yw_ = self.local_to_global_xy_(x_, y_)
            nrho1w = self.calc_nrho(xw_, yw_)
            velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
            vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
            nrhow = (
                  np.sqrt(2.0*np.pi/self.tempw/self.R)
                * (nrho1w*velx1w*self.sinalpha - nrho1w*vely1w*self.cosalpha)
            )
            return (
                  nrhow*(y - yw*self.sinalpha)
                / (((y-yw*self.sinalpha)**2 + (x-self.cwx-yw*self.cosalpha)**2)**(3/2))
            )

        return 1.0/nrho2*(
              nrho1*vely1
            + (y*self.cosalpha - (x - self.cwx)*self.sinalpha)
            * np.sqrt(self.R*self.tempw/8.0/np.pi)
            * self.integrate_(integrand, self.ww[0], self.ww[1])
        )

    def vely2_field(self):
        if self.vely2_ is None:
            # set and return the field
            self.vely2_ = self.field_(self.calc_vely2)
        return self.vely2_

    def calc_nrho2_wall(self, yw):
        # yw is inclined and local, functions expect global
        x_ = self.x_from_yw_(yw)
        y_ = self.y_from_yw_(yw)
        xw_, yw_ = self.local_to_global_xy_(x_, y_)
        nrho1w = self.calc_nrho(xw_, yw_)
        velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
        vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
        nrhow = (
              np.sqrt(2.0*np.pi/self.tempw/self.R)
            * (nrho1w*velx1w*self.sinalpha - nrho1w*vely1w*self.cosalpha)
        )
        return nrho1w + nrhow

    def calc_vely2_wall(self, yw):
        # yw is inclined and local, functions expect global
        x_ = self.x_from_yw_(yw)
        y_ = self.y_from_yw_(yw)
        xw_, yw_ = self.local_to_global_xy_(x_, y_)
        nrho1w = self.calc_nrho(xw_, yw_)
        velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
        vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
        nrhow = (
              np.sqrt(2.0*np.pi/self.tempw/self.R)
            * (nrho1w*velx1w*self.sinalpha - nrho1w*vely1w*self.cosalpha)
        )
        return (
              (nrho1w*velx1w*self.cosalpha + nrho1w*vely1w*self.sinalpha)
            / (nrhow + nrho1w)
        )

class FreeMolecularFlow2dConvexImpingement(FreeMolecularFlow2d):
    def __init__(self, d, molmass, nrho0, vel0, temp0, xmin, xmax, nx, ymin, ymax, ny, shape, tempw):
        super().__init__(d, molmass, nrho0, vel0, temp0, xmin, xmax, nx, ymin, ymax, ny)
        # wall properties:
        self.tempw = tempw

        # 2d convex shape with fully diffuse interaction at tempw
        self.global_shape = shape
        self.shape = shape
        for s in self.shape.segments:
            # iterate over w to x, y
            steps = np.linspace(s.w[0], s.w[1], 100)
            for step in steps:
                w = step
                x_ = s.w_to_x(w)
                y_ = s.w_to_y(w)
                xw_, yw_ = self.local_to_global_xy_(x_, y_)

        # local slit points
        self.ps1 = (0,-d/2)
        self.ps2 = (0,d/2)
        # and slit Segment2d
        self.slit = geometry.Segment2d(self.ps1, self.ps2)

        # collected field values over grid
        self.nrho2_ = None
        self.velx2_ = None
        self.vely2_ = None
        self.temp2_ = None
        self.press2_ = None

    def los(self, x, y):
        # x, y are global
        x, y = self.global_to_local_xy_(x, y)
        p = np.array([x, y], dtype="d")

        # poistion and integration range per point

        # outside of domain?
        if x <= 0:
            return (None, None)

        # inside the shape?
        if self.shape.inside(p):
            return (None, None)

        # LOS to point
        ps1, ps2 = self.shape.point_segment_los(p, self.slit)

        # completely blocked LOS (e.g. behind shape)?
        if ps1 is None and ps2 is None:
            return (None, None)

        # note: psi[0] should always be zero (slit)
        t1, t2 = (np.arctan((p[1] - ps2[1])/(p[0] - ps2[0])), np.arctan((p[1] - ps1[1])/(p[0] - ps1[0])))
        return (t1, t2)

    def calc_nrho(self, x, y):
        # override super behaviour with line of sight method
        t1, t2 = self.los(x, y)
        if t1 is None and t2 is None:
            return 0.0

        # C library
        integrand = self.integrand_nrho
        return self.nrho0*(
              np.exp(-self.s0**2)/(2.0*np.pi)*(t2-t1)
            + 0.25 * (erf(self.s0*np.sin(t2)) - np.sign(t1)*erf(self.s0*np.sin(np.abs(t1))))
            + self.s0/(2.0*constants.SQRT_PI)*self.integrate_(integrand, t1, t2)
        )

    def calc_velx(self, x, y, nrhoxy = None):
        # override super behaviour with line of sight method
        if nrhoxy is None:
            n = self.field_xy_(self.nrho_field(), x, y)/self.nrho0
        else:
            n = nrhoxy/self.nrho0

        t1, t2 = self.los(x, y)
        if t1 is None and t2 is None:
            return 0.0

        # C library
        integrand1 = self.integrand_velx1
        # C library
        integrand2 = self.integrand_velx2
        return self.velth*np.exp(-self.s0**2)/(2.0*n*np.pi)*(
              self.integrate_(integrand1, t1, t2)
            + self.s0*(t2 - t1)/2.0
            + self.s0*(np.sin(2.0*t2) - np.sin(2.0*t1))/4.0
            + self.s0**2*constants.SQRT_PI*self.integrate_(integrand2, t1, t2)
        )

    def calc_vely(self, x, y, nrhoxy = None):
        if nrhoxy is None:
            n = self.field_xy_(self.nrho_field(), x, y)/self.nrho0
        else:
            n = nrhoxy/self.nrho0

        t1, t2 = self.los(x, y)
        if t1 is None and t2 is None:
            return 0.0
        return self.velth/(4.0*constants.SQRT_PI*n)*(
              np.exp(-(self.s0*np.sin(t1))**2)*np.cos(t1)
            * (1.0 + erf(self.s0*np.cos(t1)))
            - np.exp(-(self.s0*np.sin(t2))**2)*np.cos(t2)
            * (1.0 + erf(self.s0*np.cos(t2)))
        )

    def calc_nrho2(self, x, y):
        # calls the overriden calc_nrho method above!
        nrho1 = self.field_xy_(self.nrho_field(), x, y)
        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        p = np.array([x, y], dtype="d")

        nrho = nrho1
        # inside the shape?
        if self.shape.inside(p):
            return nrho

        # loop over all segments of the shape
        for s in self.shape.segments:
            # check if p is in view of s
            if not s.in_view(p):
                continue

            w1, w2 = s.w
            cosalpha = s.cosalpha
            sinalpha = s.sinalpha
            cx = s.c[0]

            def integrand(w):
                # w is on inclined segment and local, functions expect global
                x_ = s.w_to_x(w)
                y_ = s.w_to_y(w)
                xw_, yw_ = self.local_to_global_xy_(x_, y_)

                nrho1w = self.calc_nrho(xw_, yw_)

                if nrho1w == 0.0:
                    return 0.0
                velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
                vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
                nrhow = (
                      np.sqrt(2.0*np.pi/self.tempw/self.R)
                    * (nrho1w*velx1w*sinalpha - nrho1w*vely1w*cosalpha)
                )
                return (
                      nrhow
                    /((y-y_)**2 + (x-x_)**2)
                )

            nrho += (y*cosalpha - (x - cx)*sinalpha)/2.0/np.pi*self.integrate_(integrand, w1, w2)
        return nrho

    def nrho2_field(self):
        if self.nrho2_ is None:
            # set and return the field
            self.nrho2_ = self.field_(self.calc_nrho2)
        return self.nrho2_

    def calc_velx2(self, x, y):
        # calls the overriden calc_nrho method above!
        nrho1 = self.field_xy_(self.nrho_field(), x, y)
        velx1 = self.field_xy_(self.velx_field(), x, y)
        nrho2 = self.field_xy_(self.nrho2_field(), x, y)
        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        p = np.array([x, y], dtype="d")

        # inside the shape?
        if self.shape.inside(p):
            return velx1

        if nrho2 == 0.0:
            return velx1

        nrhovelx = nrho1*velx1

        # loop over all segments of the shape
        for s in self.shape.segments:
            # check if p is in view of s
            if not s.in_view(p):
                continue

            w1, w2 = s.w
            cosalpha = s.cosalpha
            sinalpha = s.sinalpha
            cx = s.c[0]

            def integrand(w):
                # w is on inclined segment and local, functions expect global
                x_ = s.w_to_x(w)
                y_ = s.w_to_y(w)
                xw_, yw_ = self.local_to_global_xy_(x_, y_)

                nrho1w = self.calc_nrho(xw_, yw_)
                if nrho1w == 0.0:
                    return 0.0
                velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
                vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
                nrhow = (
                      np.sqrt(2.0*np.pi/self.tempw/self.R)
                    * (nrho1w*velx1w*sinalpha - nrho1w*vely1w*cosalpha)
                )
                return (
                      nrhow*(x - cx - w*cosalpha)
                    / (((y-y_)**2 + (x-x_)**2)**(3/2))
                )

            nrhovelx += ((y*cosalpha - (x - cx)*sinalpha)
                * np.sqrt(self.R*self.tempw/8.0/np.pi)
                * self.integrate_(integrand, w1, w2)
            )
        return nrhovelx/nrho2

    def velx2_field(self):
        if self.velx2_ is None:
            # set and return the field
            self.velx2_ = self.field_(self.calc_velx2)
        return self.velx2_

    def calc_vely2(self, x, y):
        # calls the overriden calc_nrho method above!
        nrho1 = self.field_xy_(self.nrho_field(), x, y)
        vely1 = self.field_xy_(self.vely_field(), x, y)
        nrho2 = self.field_xy_(self.nrho2_field(), x, y)
        # global to local coordinates
        x, y = self.global_to_local_xy_(x, y)
        p = np.array([x, y], dtype="d")

        # inside the shape?
        if self.shape.inside(p):
            return vely1

        if nrho2 == 0.0:
            return vely1

        nrhovely = nrho1*vely1

        # loop over all segments of the shape
        for s in self.shape.segments:
            # check if p is in view of s
            if not s.in_view(p):
                continue

            w1, w2 = s.w
            cosalpha = s.cosalpha
            sinalpha = s.sinalpha
            cx = s.c[0]

            def integrand(w):
                # w is on inclined segment and local, functions expect global
                x_ = s.w_to_x(w)
                y_ = s.w_to_y(w)
                xw_, yw_ = self.local_to_global_xy_(x_, y_)

                nrho1w = self.calc_nrho(xw_, yw_)
                if nrho1w == 0.0:
                    return 0.0
                velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
                vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
                nrhow = (
                      np.sqrt(2.0*np.pi/self.tempw/self.R)
                    * (nrho1w*velx1w*sinalpha - nrho1w*vely1w*cosalpha)
                )
                return (
                      nrhow*(y - w*sinalpha)
                    / (((y-y_)**2 + (x-x_)**2)**(3/2))
                )

            nrhovely += ((y*cosalpha - (x - cx)*sinalpha)
                * np.sqrt(self.R*self.tempw/8.0/np.pi)
                * self.integrate_(integrand, w1, w2)
            )
        return nrhovely/nrho2

    def vely2_field(self):
        if self.vely2_ is None:
            # set and return the field
            self.vely2_ = self.field_(self.calc_vely2)
        return self.vely2_

    def calc_nrho2_wall(self, s, w):
        # w is on inclined segment and local, functions expect global
        x_ = s.w_to_x(w)
        y_ = s.w_to_y(w)
        xw_, yw_ = self.local_to_global_xy_(x_, y_)

        nrho1w = self.calc_nrho(xw_, yw_)
        if nrho1w == 0.0:
            return 0.0
        velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
        vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
        nrhow = (
              np.sqrt(2.0*np.pi/self.tempw/self.R)
            * (nrho1w*velx1w*s.sinalpha - nrho1w*vely1w*s.cosalpha)
        )
        return nrho1w + nrhow

    def calc_vely2_wall(self, s, w):
        # w is on inclined segment and local, functions expect global
        x_ = s.w_to_x(w)
        y_ = s.w_to_y(w)
        xw_, yw_ = self.local_to_global_xy_(x_, y_)

        nrho1w = self.calc_nrho(xw_, yw_)
        if nrho1w == 0.0:
            return 0.0
        velx1w = self.calc_velx(xw_, yw_, nrhoxy = nrho1w)
        vely1w = self.calc_vely(xw_, yw_, nrhoxy = nrho1w)
        nrhow = (
              np.sqrt(2.0*np.pi/self.tempw/self.R)
            * (nrho1w*velx1w*s.sinalpha - nrho1w*vely1w*s.cosalpha)
        )
        return (
              (nrho1w*velx1w*s.cosalpha + nrho1w*vely1w*s.sinalpha)
            / (nrhow + nrho1w)
        )

class SuperpositionFreeMolecularFlow2d(FreeMolecularFlow):
    def __init__(self, flows):
        self.flows = flows

        # global coordinate system:
        self.xc = flows[0].xc
        self.yc = flows[0].yc

        # xc, yc grids with equal layout to fields
        self.xgrid, self.ygrid = geometry.centers_to_grid(self.xc, self.yc)

        # fields:
        self.nrho_ = None
        self.velx_ = None
        self.vely_ = None
        self.temp_ = None
        self.press_ = None
        self.calc_fields()

    def calc_fields(self):
        self.nrho_ = self.flows[0].nrho_field()
        self.velx_ = self.flows[0].nrho_field()*self.flows[0].velx_field()
        self.vely_ = self.flows[0].nrho_field()*self.flows[0].vely_field()
        for flow in self.flows[1:]:
            self.nrho_ += flow.nrho_field()
            self.velx_ += flow.nrho_field()*flow.velx_field()
            self.vely_ += flow.nrho_*flow.vely_field()
        self.velx_ /= self.nrho_
        self.vely_ /= self.nrho_

        # press has to consider bulk vel
        mfac = 1.0/3.0*self.flows[0].molmass/constants.AVOGADRO
        self.press_ = self.flows[0].press_field() + mfac*self.flows[0].nrho_field()*(
              (self.flows[0].velx_field() - self.velx_)**2
            + (self.flows[0].vely_field() - self.vely_)**2
        )
        for flow in self.flows[1:]:
            self.press_ += flow.press_field() + mfac*flow.nrho_field()*(
                  (flow.velx_field() - self.velx_)**2
                + (flow.vely_field() - self.vely_)**2
            )
        # temp follows from ideal gas law
        self.temp_ = self.press_/self.nrho_/constants.BOLTZ
