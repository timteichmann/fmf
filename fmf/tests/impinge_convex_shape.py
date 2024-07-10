# FreeMolecularFlow - tests - impinge_convex_shape
# (c) 2024 Tim Teichmann

# Test case for my implementation of jet impingement on arbitrary two
# dimensional convex shapes with an arbitrary number of surfaces. The results
# have been compared with collisionless DSMC results and match nearly perfectly.

import numpy as np
import matplotlib.pyplot as plt

from fmf.fmf import FreeMolecularFlow2dConvexImpingement
from fmf.common import constants
from fmf.common import plot
from fmf.common.geometry import Ray2d, Segment2d, Shape2d

H = 0.5 # m
T0 = 200 # K
M = 40 # g/mol (reference)
R = constants.GAS_CONSTANT/M/1e-3 # J/kg/K
vth = np.sqrt(5./3.*R*T0) # monatomic, Ma=1
v0 = 2.0*np.sqrt(2.0*R*T0)
nrho0 = 1 # 1/m^3 (reference)

diamond = Shape2d([Segment2d((2,0),(3,1)), Segment2d((3,1),(4,0)), Segment2d((4,0),(3,-1)), Segment2d((3,-1),(2,0))])

flow = FreeMolecularFlow2dConvexImpingement(2*H, M, nrho0, v0, T0, 0.0, 6, 100, -3, 3, 100, diamond, 1.5*T0)

plot.scalar_field(flow.xc, flow.yc, flow.nrho2_field(), np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]), r"$\frac{\rho_N}{\rho_{N,0}}$")
plot.scalar_field(flow.xc, flow.yc, flow.velx2_field()/np.sqrt(2.0*R*T0), np.array([-1.4, -0.8, -0.4, -0.2, 0, 0.2, 0.4, 0.8, 1.4]), r"$v_x/\sqrt[2]{2RT_0}$")
plot.scalar_field(flow.xc, flow.yc, flow.vely2_field()/np.sqrt(2.0*R*T0), np.array([-0.8, -0.6, -0.4, -0.2, 0, 0.2 , 0.4, 0.6, 0.8]), r"$v_x/\sqrt[2]{2RT_0}$")
