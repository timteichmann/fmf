# FreeMolecularFlow - tests - superposition
# (c) 2024 Tim Teichmann

# As outlined in 
#   L. Wang and C. Cai, “Gaseous plume flows in space propulsion,”
#   Chinese Journal of Aeronautics, vol. 26, no. 3, pp. 522–528, Jun. 2013,
#   doi: 10.1016/j.cja.2013.04.049.
# it is possible to calculate more complex cases (they show the example of two
# planar slit flows) as a superposition of multiple "simple" slit calculations.
# This is done here by splitting the slit of the "slit" test case in two
# smaller slits.

import numpy as np
import matplotlib.pyplot as plt

from fmf.fmf import FreeMolecularFlow2d
from fmf.fmf import SuperpositionFreeMolecularFlow2d
from fmf.common import constants
from fmf.common import plot

H = 0.05 # m # halfed
T0 = 300 # K
M = 40 # g/mol (reference)
R = constants.GAS_CONSTANT/M/1e-3 # J/kg/K
v0 = np.sqrt(5./3.*R*T0) # monatomic, Ma=1
nrho0 = 1 # 1/m^3 (reference)

flow1 = FreeMolecularFlow2d(2*H, M, nrho0, v0, T0, 0.0, 2.0, 200, 0.0, 1.0, 100)
flow1.translate([0.0, +H])
flow1.calc_fields()
flow2 = FreeMolecularFlow2d(2*H, M, nrho0, v0, T0, 0.0, 2.0, 200, 0.0, 1.0, 100)
flow2.translate([0.0, -H])
flow2.calc_fields()
flow = SuperpositionFreeMolecularFlow2d([flow1,flow2])
plot.scalar_field(flow.xc, flow.yc, flow.nrho_field(), [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], r"$\frac{\rho_N}{\rho_{N,0}}$")
plot.scalar_field(flow.xc, flow.yc, flow.velx_field(), np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.35])*np.sqrt(2.0*constants.GAS_CONSTANT/M/1e-3*T0), r"$v_x$")
plot.scalar_field(flow.xc, flow.yc, flow.vely_field(), np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])*np.sqrt(2.0*constants.GAS_CONSTANT/M/1e-3*T0), r"$v_y$")
