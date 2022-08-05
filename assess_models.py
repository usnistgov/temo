import pandas
import matplotlib.pyplot as plt
import numpy as np

import teqp 

from temo.analyze import plotting

# rp = plotting.ResultsParser('run9d248019-1437-11ed-bfe3-ccd9ac334590.tar.xz')
rp = plotting.ResultsParser('runb4cd0eae-146f-11ed-8dcf-ccd9ac334590.tar.xz')
last_stepfile = rp.get_stepfiles(rp.get_lowest_cost_uid())[-1]
mutant, basemodel = plotting.build_mutant(['AMMONIA', 'WATER'], path=teqp.get_datapath(), s=last_stepfile['model'])
modelNH3 = teqp.build_multifluid_model(['AMMONIA'], teqp.get_datapath())
ancNH3 = modelNH3.build_ancillaries()
modelH2O = teqp.build_multifluid_model(['WATER'], teqp.get_datapath())
ancH2O = modelNH3.build_ancillaries()

# Critical curves
cr = plotting.calc_critical_curves(model=mutant, basemodel=basemodel, ipure=0, show=False)
cr['z_0 / mole frac.'] = cr['rho0 / mol/m^3']/(cr['rho0 / mol/m^3']+cr['rho1 / mol/m^3'])
# df.info()
plt.plot(cr['T / K'], cr['p / Pa']/1e6)
plt.gca().set(xlabel='$T$ / K', ylabel='$p$ / MPa')
df = pandas.read_csv('NH3H2O/upstream/crit.csv')
plt.plot(df['T / K'], df['p / Pa']/1e6, 'o')
plt.close()

# VLE
df = pandas.read_csv('NH3H2O/upstream/NH3H2Odata.csv')
df = df[df['type'] == 'PTXY']
df = df[(df['T / K'] > 10+273.15) & (df['T / K'] < 40+273.15)]
for bibkey, gp in df.groupby('bibkey'):
    plt.scatter(gp['x_1 / mole frac.'], gp['p / Pa']/1e6, marker='o', c=gp['T / K'], vmin = 273, vmax=700, label=bibkey)
    sc = plt.scatter(gp['y_1 / mole frac.'], gp['p / Pa']/1e6, marker='^', c=gp['T / K'], vmin = 273, vmax=700, label=bibkey)

for T, gp in df.groupby('T / K'):
    for pure, anc in zip([modelH2O, modelNH3],[ancH2O, ancNH3]):
        rhoL, rhoV = teqp.pure_VLE_T(pure, T, anc.rhoL(T), anc.rhoV(T), 10)
        rhovecL = np.array([rhoL, 0])
        rhovecV = np.array([rhoV, 0])
        i = plotting.isotherm(mutant, T, rhovecL, rhovecV)
        plt.plot(i['x0 / mole frac.'], i['pL / Pa']/1e6, color = sc.to_rgba(T))
        plt.plot(i['y0 / mole frac.'], i['pL / Pa']/1e6, color = sc.to_rgba(T), dashes=[2,2])

plt.plot(cr['z_0 / mole frac.'], cr['p / Pa']/1e6, c='k')
plt.yscale('log')
cb = plt.colorbar()
cb.set_label('$T$ / K')
plt.gca().set(xlabel='$x_1,y_1$ / mole frac.', ylabel='$p$ / MPa')
plt.show()