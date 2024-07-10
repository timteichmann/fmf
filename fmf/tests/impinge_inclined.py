# FreeMolecularFlow - tests - impinge_inclined
# (c) 2024 Tim Teichmann

# Validation following
#   C. Cai and X. He, “Detailed flowfield and surface properties for high
#   Knudsen number planar jet impingement at an inclined flat plate,”
#   Physics of Fluids, vol. 28, no. 5, p. 056103, May 2016,
#   doi: 10.1063/1.4948365.

# From my calculations I believe that the DSMC and analytical line assignment
# in their figures 10, 12 and 14 are switched. Additionally, I think that the
# plate was assumed to have larger dimensions. This is why the L/W ratio is
# chosen double in my following calculations. With the specified L/W = 1.0 the
# plates corners would be inside the shown domain.

import numpy as np
import matplotlib.pyplot as plt

from fmf.fmf import FreeMolecularFlow2dImpingement
from fmf.common import constants
from fmf.common import plot

H = 0.5 # m
T0 = 200 # K
M = 40 # g/mol (reference)
R = constants.GAS_CONSTANT/M/1e-3 # J/kg/K
vth = np.sqrt(5./3.*R*T0) # monatomic, Ma=1
v0 = 2.0*np.sqrt(2.0*R*T0)
nrho0 = 1 # 1/m^3 (reference)

flow = FreeMolecularFlow2dImpingement(2*H, M, nrho0, v0, T0, 0.0, 8, 100, -4, 4, 100, 1.5*T0, 0, -np.tan(np.radians(60))*4, 8, np.tan(np.radians(60))*4)

ys = np.linspace(flow.ww[0], flow.ww[1], 101)
nrho2w = []
slip = []

print("wall conditions:")
for yw in ys:
    nrho2wi = flow.calc_nrho2_wall(yw)
    nrho2w.append(nrho2wi)
    print("    -> nrho wall = " +  str(nrho2wi))
    slipi = flow.calc_vely2_wall(yw)/np.sqrt(2.0*R*T0)
    slip.append(slipi)
    print("    -> norm slip vel = " + str(slipi))

plot.xy(ys, slip, xlabel=r"$y/(2H)$", ylabel=r"$u_\mathrm{slip}/\sqrt[2]{2RT_0}$")

plot.scalar_field(flow.xc, flow.yc, flow.nrho2_field(), np.array([0.2, 0.4, 0.6, 0.8, 1.0]), r"$\frac{\rho_N}{\rho_{N,0}}$")
plot.scalar_field(flow.xc, flow.yc, flow.velx2_field()/np.sqrt(2.0*R*T0), np.array([-1, -0.6, -0.4, -0.2, 0, 0.4, 0.8, 1.4]), r"$v_x/\sqrt[2]{2RT_0}$")
plot.scalar_field(flow.xc, flow.yc, flow.vely2_field()/np.sqrt(2.0*R*T0), np.array([-0.6, -0.4, -0.2, 0, 0.2 , 0.4, 0.6, 0.8]), r"$v_y/\sqrt[2]{2RT_0}$")
