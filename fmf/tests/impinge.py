# FreeMolecularFlow - tests - impinge
# (c) 2024 Tim Teichmann

# Validation following
#   K. Al Khasawneh, C. Cai, and H. Liu, “Rarefied Compressible Two-Dimensional
#   Jet Plume Impingement on a Flat Plate,” in 49th AIAA Aerospace Sciences
#   Meeting including the New Horizons Forum and Aerospace Exposition, American
#   Institute of Aeronautics and Astronautics, 2011. doi: 10.2514/6.2011-761.

import numpy as np
import matplotlib.pyplot as plt

from fmf.fmf import FreeMolecularFlow2dNormalImpingement
from fmf.common import constants
from fmf.common import plot

H = 0.1 # m
T0 = 273 # K
M = 40 # g/mol (reference)
R = constants.GAS_CONSTANT/M/1e-3 # J/kg/K
vth = np.sqrt(5./3.*R*T0) # monatomic, Ma=1
v0 = 2.0*np.sqrt(2.0*R*T0)
nrho0 = 1 # 1/m^3 (reference)

flow = FreeMolecularFlow2dNormalImpingement(2*H, M, nrho0, v0, T0, 0.0, 2.0, 100, 0.0, 2.0, 100, T0, 2.0, -2.0, 2.0)

ys = np.linspace(0.0, 2.0, 21)
yd = []
nrho2w = []
temp2w = []
temp2nw = []
press2w = []
press2nw = []
slip = []

print("wall conditions:")
for y in ys:
    ydi = y/(2*H)
    yd.append(ydi)
    print("  y/D = " + str(ydi))
    nrho2wi = flow.calc_nrho2_wall(y)
    nrho2w.append(nrho2wi)
    print("    -> nrho wall = " +  str(nrho2wi))
    temp2wi = flow.calc_temp2_wall(y)/T0
    temp2w.append(temp2wi)
    print("    -> norm temp wall = " +  str(temp2wi))
    temp2nwi = flow.calc_temp2_near_wall(y)/T0
    temp2nw.append(temp2wi)
    print("    -> norm near temp wall = " +  str(temp2nwi))
    press2wi = flow.calc_press2_wall(y)/(nrho0*flow.molmass/constants.AVOGADRO*v0**2/2.0)
    press2w.append(press2wi)
    print("    -> norm press wall = " +  str(press2wi))
    press2nwi = flow.calc_press2_near_wall(y)/(nrho0*flow.molmass/constants.AVOGADRO*v0**2/2.0)
    press2nw.append(press2nwi)
    print("    -> norm press near wall = " +  str(press2nwi))
    slipi = flow.calc_vely2_wall(y)/np.sqrt(2.0*flow.R*flow.tempw)
    slip.append(slipi)
    print("    -> norm slip vel = " + str(slipi))

plot.xy(yd, press2w, xlabel=r"$y/D$", ylabel=r"$p_\mathrm{w}/\left(0.5 \rho_{N,0} v_{x,0} \right)$")

plot.xy(yd, slip, xlabel=r"$y/D$", ylabel=r"$u_\mathrm{slip}/\sqrt[2]{2RT_0}$")

plot.scalar_field(flow.xc, flow.yc, flow.nrho2_field(), np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), r"$\frac{\rho_N}{\rho_{N,0}}$")
plot.scalar_field(flow.xc, flow.yc, flow.velx2_field()/np.sqrt(2.0*R*T0), np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]), r"$v_x/\sqrt[2]{2RT_0}$")
plot.scalar_field(flow.xc, flow.yc, flow.vely2_field()/np.sqrt(2.0*R*T0), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), r"$v_y/\sqrt[2]{2RT_0}$")
