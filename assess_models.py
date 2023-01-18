from multiprocessing.sharedctypes import Value
import pickle
from dataclasses import dataclass

import pandas, os
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize 

import teqp 

from temo.fit.cost_contributions import calc_errVLE_x, calc_errrho, calc_errrho_devp, calc_errSOS, calc_err_critisoT
from temo.fit.data_loaders import load_PVT, load_PVT_P, load_VLE, load_SOS
from temo.analyze import plotting

import os 
os.environ['RPPREFIX'] = os.getenv('HOME') + '/REFPROP10'

# for i, result in enumerate(glob.glob('results/run*.tar.xz')):
#     load_args = ['NH3H2O']
#     load_kwargs = dict(
#         identifier='FLD', 
#         identifiers=['AMMONIA','WATER'], 
#         molar_masses=[0.01703052, 0.018015268], 
#         apply_skip=True,
#         verbosity=0
#     )
#     rp = plotting.ResultsParser(result)
#     last_stepfile = rp.get_stepfiles(rp.get_lowest_cost_uid())[-1]
#     mutant, basemodel = plotting.build_mutant(['AMMONIA', 'WATER'], path=teqp.get_datapath(), s=last_stepfile['model'])

#     df = load_SOS(*load_args, **load_kwargs)
#     df['werr / %'] = calc_errSOS(model=mutant, df=df)
#     print(i, result, np.mean(np.abs(df['werr / %'])), 'werr%')
#     del df
#     del mutant
# quit()

def monitor_run(*, path):
    from temo.analyze.plotting import plot_critical_locus_history, plot_cost_history
    rp = plotting.ResultsParser(path); uid = rp.get_all_uid(path)[0]
    stepfiles = rp.get_stepfiles(uid=uid)
    mutant, basemodel = plotting.build_mutant(['AMMONIA', 'WATER'], path=teqp.get_datapath(), s=stepfiles[-1]['model'])
    print(calc_err_critisoT(model=mutant, df=pandas.read_csv('NH3H2O/crit.csv')))
    plot_cost_history(basemodel, stepfiles=stepfiles)
    plot_critical_locus_history(basemodel, stepfiles=stepfiles, dfcr=pandas.read_csv('NH3H2O/CRIT.csv'), ylim=(10, 30))

# rp = plotting.ResultsParser('results/run6d6315dc-1f37-11ed-9c0f-509a4c56f794.tar.xz')
# rp = plotting.ResultsParser('results/runc7e1966b-20d3-11ed-b7f9-509a4c56f794.tar.xz')
# rp = plotting.ResultsParser('results/run57220aa3-21b9-11ed-a042-509a4c56f794.tar.xz')
rp = plotting.ResultsParser('results/runc68d391e-2222-11ed-b5ca-509a4c56f794.tar.xz')  # A pretty decent model
# rp = plotting.ResultsParser('results/runc7470097-240a-11ed-80b1-509a4c56f794.tar.xz')
# rp = plotting.ResultsParser('results/run64617172-234a-11ed-8160-509a4c56f794.tar.xz')
# rp = plotting.ResultsParser('results/run8f838561-28d3-11ed-ae4c-509a4c56f794.tar.xz')

# rp = plotting.ResultsParser('run15705f85-2ee7-11ed-873b-ccd9ac334590')
# rp = plotting.ResultsParser('rundcd8a64a-2ee8-11ed-b071-ccd9ac334590') 
# rp = plotting.ResultsParser('run18b0feef-2eef-11ed-87a2-ccd9ac334590') # just betas and gammas, B12
# rp = plotting.ResultsParser('run8c7accdb-2eef-11ed-905b-ccd9ac334590') # just betas and gammas, B12
# rp = plotting.ResultsParser('run56f07c48-2ef2-11ed-8c01-ccd9ac334590') # Try to match Tillner-Roth reducing functions

# rp = plotting.ResultsParser('run3b4d1b74-2efb-11ed-8f55-ccd9ac334590')

rp.to_csv(prefix='aaaa')

# monitor_run(path=rp.path)

# uid = rp.get_all_uid(rp.path)[0]
uid = rp.get_lowest_cost_uid()
# uid = '642610b1-23d1-11ed-8634-509a4c56f794'
# uid = '0fc45562-24d4-11ed-8293-548d5aaf8853'
last_stepfile = rp.get_stepfiles(uid)[-1]

pandas.DataFrame(rp.get_stepfiles(uid)).to_csv('steps.csv', index=False)

mutant, basemodel = plotting.build_mutant(['AMMONIA', 'WATER'], path=teqp.get_datapath(), s=last_stepfile['model'])
modelNH3 = teqp.build_multifluid_model(['AMMONIA'], teqp.get_datapath())
ancNH3 = modelNH3.build_ancillaries()
modelH2O = teqp.build_multifluid_model(['WATER'], teqp.get_datapath())
ancH2O = modelH2O.build_ancillaries()
# print(mutant.get_meta())

import CoolProp.CoolProp as CP
# print([CP.PropsSI('molemass', f) for f in ['AMMONIA', 'WATER']])

load_args = ['NH3H2O']
load_kwargs = dict(
    identifier='FLD', 
    identifiers=['AMMONIA','WATER'], 
    molar_masses=[0.01703052, 0.018015268], 
    apply_skip=False,
    verbosity=0
)

@dataclass
class BCD:
    B_2: float
    B_12: float
    B_3: float
    B_4: float

def BCD_Harms(*, T, xNH3):
    T_0 = 500
    Vm0 = 65e-6 # m^3/mol (from 65 cm^3/mol)
    tau = T/T_0
    b = [0, -4.0821148, -3.5986282, -0.20374031, -0.015102575, -0.058261346, -0.0042050400, 549.16567, 660.36029, -8.5136369]
    alpha = [0, -5.9140587, -5.9735552, -2.4132251]
    c = [0, 0.63493889, 13.622925, -0.23149486, 0.61859968, -5.0935767, -2.3460636, -0.011492282, 8.0018597]
    d = [0, 8.0018597, -161.57381, 172.01315, -93.733212, -0.18540881]
    assert(len(b) == 10)
    assert(len(c) == 9)
    assert(len(d) == 6)
    
    from numpy import exp
    B11 = (b[1]*tau**(-4.0) + b[4]*tau**(-33/4) + b[7]*exp(alpha[1]*tau))*Vm0
    B12 = (b[2]*tau**(-4.0) + b[5]*tau**(-33/4) + b[8]*exp(alpha[2]*tau))*Vm0
    B22 = (b[3]*tau**(-4.0) + b[6]*tau**(-33/4) + b[9]*exp(alpha[3]*tau))*Vm0
    B = (1-xNH3)**2*B11 + 2*xNH3*(1-xNH3)*B12 + xNH3**2*B22

    C111 = (c[1]*tau**(-15/4) + c[5]*tau**(-47/4))*Vm0**2
    C112 = (c[2]*tau**(-15/4) + c[6]*tau**(-47/4))*Vm0**2
    C122 = (c[3]*tau**(-15/4) + c[7]*tau**(-47/4))*Vm0**2
    C222 = (c[4]*tau**(-15/4) + c[8]*tau**(-47/4))*Vm0**2
    C = (1-xNH3)**3*C111 + 3*xNH3*(1-xNH3)**2*C112 + 3*xNH3**2*(1-xNH3)*C122 + xNH3**3*C222

    D1111 = (d[1]*tau**(-21/4))*Vm0**3
    D1112 = (d[2]*tau**(-7))*Vm0**3
    D1122 = (d[3]*tau**(-15))*Vm0**3
    D1222 = (d[4]*tau**(-41/4))*Vm0**3
    D2222 = (d[5]*tau**(-21/4))*Vm0**3
    D = (1-xNH3)**4*D1111 + 4*xNH3*(1-xNH3)**3*D1112 + 6*xNH3**2*(1-xNH3)**2*D1122 + 4*xNH3**3*(1-xNH3)*D1222 + xNH3**4*D2222

    return BCD(B_2=B, B_3=C, B_12=B12, B_4=D)

def get_phi12_TillnerRoth_AW(model, T, molefrac):
    """ Need this special function because teqp doesn't allow molefrac of ammonia of zero """
    ammonia = teqp.build_multifluid_model(['jTR.json'], teqp.get_datapath())
    water = teqp.build_multifluid_model(['Water'], teqp.get_datapath())
    phi = model.get_B2vir(T, np.array(molefrac))- T*model.get_dmBnvirdTm(2, 1, T, np.array(molefrac)) # Overall B2 for mixture
    # print(phi)
    z = np.array([1.0])
    phi20 = ammonia.get_B2vir(T, z) - T*ammonia.get_dmBnvirdTm(2, 1, T, z) # Pure first component with index 0
    phi21 = water.get_B2vir(T, z) - T*water.get_dmBnvirdTm(2, 1, T, z)# Pure second component with index 1
    z0 = molefrac[0]
    phi12 = (phi - z0*z0*phi20 - (1-z0)*(1-z0)*phi21)/(2*z0*(1-z0))
    return phi12

def get_phi12_AW(model, T, molefrac):
    ammonia = teqp.build_multifluid_model(['Ammonia'], teqp.get_datapath())
    water = teqp.build_multifluid_model(['Water'], teqp.get_datapath())
    phi = model.get_B2vir(T, np.array(molefrac)) - T*model.get_dmBnvirdTm(2, 1, T, np.array(molefrac)) # Overall phi for mixture
    # print(phi)
    z = np.array([1.0])
    phi20 = ammonia.get_B2vir(T, z) - T*ammonia.get_dmBnvirdTm(2, 1, T, z) # Pure first component with index 0
    phi21 = water.get_B2vir(T, z) - T*water.get_dmBnvirdTm(2, 1, T, z)# Pure second component with index 1
    z0 = molefrac[0]
    phi12 = (phi - z0*z0*phi20 - (1-z0)*(1-z0)*phi21)/(2*z0*(1-z0))
    return phi12
    
def get_B12_TillnerRoth_AW(model, T, molefrac):
    """ Need this special function because teqp doesn't allow molefrac of ammonia of zero """
    ammonia = teqp.build_multifluid_model(['jTR.json'], teqp.get_datapath())
    water = teqp.build_multifluid_model(['Water'], teqp.get_datapath())
    B2 = model.get_B2vir(T, np.array(molefrac)) # Overall B2 for mixture
    z = np.array([1.0])
    B20 = ammonia.get_B2vir(T, z) # Pure first component with index 0
    B21 = water.get_B2vir(T, z) # Pure second component with index 1
    z0 = molefrac[0]
    B12 = (B2 - z0*z0*B20 - (1-z0)*(1-z0)*B21)/(2*z0*(1-z0))
    return B12

# from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
# root = r'C:\Program Files (x86)\REFPROP91'
# RP = REFPROPFunctionLibrary(root)
# RP.SETPATHdll(root)
# print(get_B12_TillnerRoth_AW(teqp.AmmoniaWaterTillnerRoth(), T=298, molefrac=[0.7, 0.3]))
# print(RP.SETUPdll(2,'AMMONIA.FLD|WATER.FLD','HMX.BNC','DEF'))
# print(RP.RPVersion())
# print(RP.B12dll(T=298, z=[0.7, 0.3])/1e3)
# quit()

def get_B12_Wormald(T):
    """ B_{12} from correlation of Wormald for ammonia + water, in m^3/mol"""
    return (38 - 43059/T - 1.993*np.exp(1900.0/T))/1e6

def get_B_2_Wormald_AW(T, molefrac):
    """ Wormald for B_12, reference EOS for Ammonia and Water """
    ammonia = teqp.build_multifluid_model(['Ammonia'], teqp.get_datapath())
    water = teqp.build_multifluid_model(['Water'], teqp.get_datapath())
    
    z = np.array([1.0])
    B20 = ammonia.get_B2vir(T, z) # Pure ammonia
    B21 = water.get_B2vir(T, z) # Pure water
    z0 = molefrac[0]
    B_12 = get_B12_Wormald(T)
    return z0*z0*B20 + (1-z0)*(1-z0)*B21 + 2*z0*(1-z0)*B_12

def recalc_Wormald_phi12():
    ammonia = teqp.build_multifluid_model(['Ammonia'], teqp.get_datapath())
    water = teqp.build_multifluid_model(['Water'], teqp.get_datapath())
    z = np.array([1.0])

    d = pandas.read_csv('Wormald_JCT_2001.csv')
    # d.info()
    y = 0.5
    p0 = 101325 # Pa
    def get_phi12(row):
        phiterm = row['HEm(po) / J/mol']/(p0*(y*(1-y)))
        T = row['T / K']
        phi11 = ammonia.get_B2vir(T, z) - T*ammonia.get_dmBnvirdTm(2, 1, T, z)
        phi22 = water.get_B2vir(T, z) - T*water.get_dmBnvirdTm(2, 1, T, z)
        phi12 = (phiterm+phi11+phi22)/2
        return phi12*1e6 # cm^3/mol
    d['phi12(recalc) / cm^3/mol'] = d.apply(get_phi12, axis=1)
    d.to_csv('recalc_Wormald.csv', index=False, encoding='utf_8_sig')

recalc_Wormald_phi12()

def check_virials(*, author):
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    root = os.getenv('RPPREFIX')
    RP = REFPROPFunctionLibrary(root)
    df = load_PVT(*load_args, **load_kwargs)
    if author == 'Harms-Watzenberg':
        df = df[df['method'] == 'Burnett']
    else:
        df = df[df['bibkey'] == 'Ellerwald-THESIS-1981']
    model = teqp.AmmoniaWaterTillnerRoth()
    
    o = []
    for z_1, gp in df.groupby('z_1 / mole frac.'):
        for T, ggp in gp.groupby('T / K'):
            BCD = BCD_Harms(T=T, xNH3=z_1)
            Z = ggp['p / Pa']/(ggp['T / K']*8.31446161815324*ggp['rho / mol/m^3'])
            Z1rho = (Z-1)/ggp['rho / mol/m^3']
            plt.plot(ggp['rho / mol/m^3']/1e3, Z1rho*1e3, '.')

            Zcorr = Z - (1-z_1)**2*70e-6*ggp['rho / mol/m^3']
            plt.plot(ggp['rho / mol/m^3']/1e3, (Zcorr-1)/ggp['rho / mol/m^3']*1e3, '.')

            plt.plot(0, model.get_B2vir(T, np.array([z_1, 1-z_1]))*1e3, '*', label='Tillner-Roth')

            B11 = RP.REFPROPdll('AMMONIA','TD&','Bvir', RP.MOLAR_BASE_SI,0,0, T, 0, [1.0]).Output[0]
            B22 = RP.REFPROPdll('WATER','TD&','Bvir', RP.MOLAR_BASE_SI,0,0, T, 0, [1.0]).Output[0]
            B12 = get_B12_Wormald(T)
            # print(B11, B22, B12)
            plt.plot(0, (z_1**2*B11 + (1-z_1)**2*B22 + 2*z_1*(1-z_1)*B12)*1e3, '*', label='Wormald B12')
            
            r = RP.REFPROPdll('AMMONIA;WATER','QT','D', RP.MOLAR_BASE_SI,0,0, 1,T, [z_1,1-z_1])
            plt.axvline(r.Output[0]/1e3)

            x = np.linspace(1e-10, r.Output[0])
            y = ((BCD.B_2 + BCD.B_3*x + BCD.B_4*x**2)*1e3)
            plt.plot(x/1e3, y, label='Harms correlation')

            w = np.ones_like(Z1rho)
            w[np.array(ggp['rho / mol/m^3']) < 20] = 0
            c = np.polyfit(ggp['rho / mol/m^3'], Z1rho, 2, w=w)
            yy = (np.polyval(c, ggp['rho / mol/m^3']) - 0.0000/ggp['rho / mol/m^3'])*1e3
            # print(c)
            B_2 = B = c[-1]
            B_3 = C = c[-2]
            B_4 = D = c[-3]
            B_12 = (B - z_1**2*BCD_Harms(T=T,xNH3=1).B_2 - (1-z_1)**2*BCD_Harms(T=T,xNH3=0).B_2)/(2*z_1*(1-z_1))
            # print(T, z_1, B_12, C, BCD_Harms(T=T, xNH3=z_1).B_3)

            o.append({'T / K': T, 'B_12': B_12})
            
            yy = (np.polyval(c, x) - 0.0000/x)*1e3
            plt.plot(x/1e3, yy, label='Harms data refit')

            plt.title(rf'$T$: {T} K; $z_1$: {z_1} mole frac.')
            plt.gca().set(xlabel=r'$\rho$ / mol/dm$^3$', ylabel=r'$(Z-1)/\rho$ / dm$^3$/mol')
            plt.legend(loc='best')
            if T < 4.10:
                plt.show()
            else:
                plt.close()

            # plt.plot(ggp['rho / mol/m^3'], (Zcorr/Z-1)*100)
            # plt.title(rf'$T$: {T} K; $z_1$: {z_1} mole frac.')
            # plt.show()

            R_TR = 8.314471
            # print(BCD_Harms(T=350, xNH3=0.5), BCD_Harms(T=550, xNH3=0.25))
            p_virial = ggp['rho / mol/m^3']*R_TR*T*(1+BCD.B_2*ggp['rho / mol/m^3'] + BCD.B_3*ggp['rho / mol/m^3']**2 + BCD.B_4*ggp['rho / mol/m^3']**3)
            plt.plot(ggp['rho / mol/m^3']/1e3, (ggp['p / Pa']/p_virial-1)*1000, 'o')
            plt.title(rf'$T$: {T} K; $z_1$: {z_1} mole frac.')
            plt.ylim(-4, 4)
            plt.close()
    
    ### B12
    xx = np.linspace(250, 800)
    for z_1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        plt.plot(xx, [get_B12_TillnerRoth_AW(model, T=x, molefrac=[z_1,1-z_1])*1e6 for x in xx], label=f'Tillner-Roth ({z_1} mole frac.)', dashes=[2,2])
    # for z_1 in [0.01, 0.1, 0.4, 0.5, 0.7, 0.9]:
    #     plt.plot(xx, [mutant.get_B12vir(T=x, molefrac=np.array([z_1, 1-z_1]))*1e6 for x in xx], label=f'mutant ({z_1} mole frac)', dashes=[3,1,1,1])
    plt.plot(xx, [BCD_Harms(T=x, xNH3=0.5).B_12*1e6 for x in xx], label='Harms-Watzenberg correlation')
    plt.plot(xx, get_B12_Wormald(xx)*1e6, label='Wormald correlation')

    df = pandas.DataFrame(o)
    plt.plot(df['T / K'], df['B_12']*1e6, 'o')
    d = pandas.read_csv('Wormald_JCT_2001.csv')
    # d.info()
    plt.plot(d['T / K'], d['B12 / cm3/mol'], 'o')
    plt.ylim(-600, 0)
    plt.gca().set(xlabel='$T$ / K', ylabel=r'$B_{12}$ / cm$^3$/mol')
    plt.legend(loc='best')
    plt.savefig('NH3H2O_B12_vals.pdf')
    plt.show()

    ### phi12
    xx = np.linspace(250, 550)
    h = 1e-100
    plt.plot(xx, [(BCD_Harms(T=x, xNH3=0.5).B_12 - x*BCD_Harms(T=x+1j*h, xNH3=0.5).B_12.imag/h)*1e6 for x in xx], label='Harms-Watzenberg correlation')
    d = pandas.read_csv('Wormald_JCT_2001.csv')
    # d.info()
    plt.plot(d['T / K'], d['φ12 / cm3/mol'], 'o')
    
    # for z_1 in [0.01, 0.1, 0.5, 0.7, 0.9]:
    #     plt.plot(xx, [get_phi12_TillnerRoth_AW(model, T=x, molefrac=[z_1,1-z_1])*1e6 for x in xx], label=f'Tillner-Roth ({z_1} mole frac.)', dashes=[2,2])
    for z_1 in [0.01, 0.1, 0.4, 0.5, 0.7, 0.9]:
        plt.plot(xx, [get_phi12_AW(mutant, T=x, molefrac=np.array([z_1, 1-z_1]))*1e6 for x in xx], label=f'mutant ({z_1} mole frac)', dashes=[3,1,1,1])
    plt.plot(xx, (get_B12_Wormald(xx)-xx*get_B12_Wormald(xx+1j*h).imag/h)*1e6, label='Wormald')

    plt.ylim(-2300, 0)
    plt.xlim(300, 500)
    plt.gca().set(xlabel='$T$ / K', ylabel=r'$\phi_{12}$ / cm$^3$/mol')
    plt.legend(loc='best')
    plt.savefig('NH3H2O_phi12_vals.pdf')
    plt.show()

def check_gas_PVT_just_B2(*, source, mutant=None):
    """ Deviations from density calculated from a B_2-truncated virial expansion """
    df = load_PVT(*load_args, **load_kwargs)
    df = df[(df['bibkey'] == 'Ellerwald-THESIS-1981') | (df['method'] == 'Burnett')]
    
    def get_rho_B(row):
        R = 8.314462618
        T = row['T / K']
        z_1 = row['z_1 / mole frac.']

        if source == 'Wormald':
            B_2 = get_B_2_Wormald_AW(T, molefrac=[z_1,1-z_1])
        elif source == 'perfect gas':
            B_2 = 0.0
        elif source == 'Harms-Watzenberg':
            BCD = BCD_Harms(T=T, xNH3=z_1)
            B_2 = BCD.B_2
        elif source == 'mutant':
            B_2 = mutant.get_B2vir(T, np.array([z_1, 1-z_1]))
            # print((B_2-get_B_2_Wormald_AW(T, molefrac=[z_1,1-z_1]))*1e6)
        elif source == 'Tillner-Roth':
            B_2 = mutant.get_B2vir(T, np.array([z_1, 1-z_1]))
            print((B_2-get_B_2_Wormald_AW(T, molefrac=[z_1,1-z_1]))*1e6)
        else:
            raise KeyError(source)
        
        def objective(rho):
            return rho*R*T*(1+B_2*rho)-row['p / Pa']
        rho_B = scipy.optimize.newton(objective, row['rho / mol/m^3'])
        # r = objective(rho_B)
        return (1-rho_B/row['rho / mol/m^3'])*100
    df['rhoerr / %'] = df.apply(get_rho_B, axis=1)

    fig, (axT, axD, axx, axP) = plt.subplots(1,4,figsize=(12, 4),sharey=True)
    print(f'source: {source}')
    for bibkey, gp in df.groupby('bibkey'):
        axT.scatter(gp['T / K'], gp['rhoerr / %'], marker='o', label=bibkey)
        axD.scatter(gp['rho / mol/m^3']/1e3, gp['rhoerr / %'], marker='o', label=bibkey)
        axx.scatter(gp['z_1 / mole frac.'], gp['rhoerr / %'], marker='o', label=bibkey)
        axP.scatter(gp['p / Pa']/1e6, gp['rhoerr / %'], marker='o', label=bibkey)
        print(bibkey, np.mean(np.abs(gp['rhoerr / %'])), 'rhoerr%')
    axT.set(xlabel=r'$T$ / K',ylabel=r'$\Delta\rho$ / %')
    axD.set(xlabel=r'$\rho$ / mol/dm$^3$')
    axx.set(xlabel=r'$z_1$ / mole frac.')
    axP.set(xlabel=r'$p$ / MPa')
    for ax in axT, axD, axx, axP:
        ax.axhline(0, zorder=-100)
        for y in [-0.75, 0.75]:
            ax.axhline(y, dashes=[2,2], color='k')
    plt.legend(loc='best')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'PVT_deviations_gas_just_B2_{source}.pdf')
    plt.show()


# for z_1 in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
#     fig, (ax1, ax2) = plt.subplots(1,2)
#     for model, ax in [(mutant, ax1),(teqp.AmmoniaWaterTillnerRoth(),ax2)]:
#         plotting.plot_criticality(
#             model=model, 
#             Tlim=(300, 700), TN=200, 
#             rholim=(1,40000), rhoN=200, 
#             z_1=z_1, ax=ax, show=False
#         )
#     plt.show()

# for T in [551.1]: # point from Sassen
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     for model, ax in [(mutant, ax1), (teqp.AmmoniaWaterTillnerRoth(), ax2)]:
#         plotting.plot_criticality_constT(
#             model=model, 
#             rholim=(1,40000), rhoN=200,
#             T=T, ax=ax, show=False
#         )
#     plt.show()

# Load the REFPROP 10.0 model from Bell and Lemmon
# mutant = teqp.build_multifluid_model(['AMMONIA', 'WATER'], r'C:\Users\ihb\Code\REFPROP-interop\teqp')

# Load the hardcoded Tillner-Roth model for ammonia+water
# mutant = teqp.AmmoniaWaterTillnerRoth()









def check_lowtemp_ammonia_virials():
    modelTR = teqp.build_multifluid_model(['jTR.json'], teqp.get_datapath())
    modelHui = teqp.build_multifluid_model(['Ammonia'], teqp.get_datapath())
    Tvec = np.linspace(150, 300)
    z = np.array([1.0])
    yTR, yHui = [], []
    for T in Tvec:
        yTR.append(modelTR.get_B2vir(T, z)*1e6)
        yHui.append(modelHui.get_B2vir(T, z)*1e6)
    plt.plot(Tvec, np.array(yTR) - np.array(yHui))
    plt.gca().set(xlabel='T / K', ylabel=r'$B_{2,{\rm Tillner-Roth}} - B_{2,{\rm Gao}}$ / cm$^3$/mol')
    plt.show()

def failure_of_corresponding_states(model):
    basemodels = [teqp.build_multifluid_model([n],teqp.get_datapath()) for n in ['AMMONIA','WATER']]
    # model = teqp.AmmoniaWaterTillnerRoth()
    def f(T, x_1):
        z = np.array([x_1, 1-x_1])
        rhored = model.get_rhor(z)
        Tred = model.get_Tr(z)
        rhocvec = 1/model.get_vcvec()
        Tcvec = model.get_Tcvec()
        
        i = 0
        yB01_B = rhored*z[i]**2*basemodels[i].get_B2vir(T, np.array([1.0]))
        yB01_CS = rhocvec[i]*z[i]*basemodels[i].get_B2vir(Tcvec[i]/Tred*T, np.array([1.0]))
        # print(i, Tcvec[i]/Tred*T, 'K actual temp of evaluation')
        plt.plot(Tred/T, yB01_B-yB01_CS, 'x')
        # print(rhored*z[i]**2, rhocvec[i]*z[i])

    for x_1 in [0.5]:
        T = np.geomspace(250, 10000, 100)
        z = np.array([x_1, 1-x_1])
        Tr = model.get_Tr(z)
        y = [f(T=T_, x_1=x_1) for T_ in T]
        # plt.plot(Tr/T, y, label=x_1)
    # plt.show()

    plt.gca().set(xlabel=r'$T_{\rm red}(x)/T$', ylabel=r'$x_1x_2\lim_{\delta\to 0}\left(\partial \alpha^{\rm r}_{ij}/\partial \delta \right)_{\tau}$')
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('backed_out_alpharij_deriv.pdf')
    plt.show()


def back_calculate_alpharij_B12(model, *, source):
    basemodels = [teqp.build_multifluid_model([n],teqp.get_datapath()) for n in ['AMMONIA','WATER']]
    def f(T, x_1):
        z = np.array([x_1, 1-x_1])
        rhored = model.get_rhor(z)
        Tred = model.get_Tr(z)
        rhocvec = 1/model.get_vcvec()
        Tcvec = model.get_Tcvec()
        
        Bo1 = basemodels[0].get_B2vir(T, np.array([1.0]))
        Bo2 = basemodels[1].get_B2vir(T, np.array([1.0]))
        if source == 'Wormald':
            B12 = get_B12_Wormald(T=T)
        elif source == 'Harms-Watzenberg':
            B12 = BCD_Harms(T=T, xNH3=x_1).B_12
        else:
            raise ValueError(source)
        
        B_2 = x_1**2*Bo1 + (1-x_1)**2*Bo2 + 2*x_1*(1-x_1)*B12 # m^3/mol
        
        num = rhored*B_2
        
        for i in range(2):
            T_i = Tcvec[i]/Tred*T
            # print(i, T, T_i)
            num -= z[i]*basemodels[i].get_B2vir(T_i, np.array([1.0]))*rhocvec[i]
        return num

    for x_1 in [0.000001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999999]:
        T = np.geomspace(250, 10000, 100)
        z = np.array([x_1, 1-x_1])
        Tr = model.get_Tr(z)
        y = [f(T=T_, x_1=x_1) for T_ in T]
        plt.plot(Tr/T, y, label=x_1)

        # aw = teqp.AmmoniaWaterTillnerRoth()
        # tau = np.geomspace(0.3, 2, 100)
        # y = []
        # for tau_ in tau:
        #     gamma = 0.5248379
        #     #### These are all zero, which makes sense, since there are no terms with density exponent of 1
        #     # delta = 1e-6
        #     # yp = aw.alphar_departure(tau=tau_, delta=delta, xNH3=x_1)
        #     # ym = aw.alphar_departure(tau=tau_, delta=1.0, xNH3=x_1)
        #     # y.append((yp-ym)/(delta)/(x_1*(1-x_1**gamma)))
        #     y.append(aw.dalphar_departure_ddelta(tau_, 0.0, x_1)/(x_1*(1-x_1**gamma)))
        # # plt.plot(tau, y, dashes=[2,2])

    plt.gca().set(xlabel=r'$T_{\rm red}(x)/T$', ylabel=r'$x_1x_2\lim_{\delta\to 0}\left(\partial \alpha^{\rm r}_{ij}/\partial \delta \right)_{\tau}$')
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('backed_out_alpharij_deriv.pdf')
    plt.show()

def plot_all_critical_loci(models, aliases):

    T0 = basemodel.get_Tcvec()[1]
    rhovec0 = np.array([0, 1.0/basemodel.get_vcvec()[1]])
    rho = rhovec0.sum()
    z = rhovec0/rho
    R = mutant.get_R(z)
    dpdT_homo = R*rho*(1+mutant.get_Ar01(T0, rho, z) - mutant.get_Ar11(T0, rho, z))
    dpdrhoi = R*T0 + np.dot(rhovec0, mutant.build_Psir_Hessian_autodiff(T0, rhovec0)) # T and other concentrations held constant
    dpdT_crit = float(dpdT_homo + np.dot(dpdrhoi, mutant.get_drhovec_dT_crit(T0, rhovec0)))
    print(dpdT_crit/1e6)

    for model, alias in zip(models, aliases):
        cr = plotting.calc_critical_curves(model=model, basemodel=basemodel, ipure=0, integration_order=5)
        cr['z_0 / mole frac.'] = cr['rho0 / mol/m^3']/(cr['rho0 / mol/m^3']+cr['rho1 / mol/m^3'])
        plt.plot(cr['z_0 / mole frac.'], cr['p / Pa']/1e6, label=alias)
    df = pandas.read_csv('NH3H2O/CRIT.csv')
    plt.plot(df['z_1 / mole frac.'], df['p / Pa']/1e6, 'o')
    df = pandas.read_csv('NH3H2O/upstream/RainwaterLG.csv')
    plt.plot(df['x_NH3'], df['pc_MPa'], 'o')
    plt.gca().set(xlabel='$z_1$ / mole frac.', ylabel='$p$ / MPa')
    plt.legend(loc='best')
    plt.close()

    for model, alias in zip(models, aliases):
        cr = plotting.calc_critical_curves(model=model, basemodel=basemodel, ipure=0, integration_order=5)
        plt.plot(cr['T / K'], cr['p / Pa']/1e6, label=alias)
    x = np.linspace(T0-50, T0)
    plt.plot(x, 22.05 - (T0-x)*dpdT_crit/1e6)
    df = pandas.read_csv('NH3H2O/CRIT.csv')
    plt.plot(df['T / K'], df['p / Pa']/1e6, 'o')
    df = pandas.read_csv('NH3H2O/upstream/RainwaterLG.csv')
    plt.plot(df['Tc_K'], df['pc_MPa'], 'o')
    plt.gca().set(xlabel='$T$ / K', ylabel='$p$ / MPa')
    plt.legend(loc='best')
    plt.close()

# models = [modelNH3, modelH2O]
# ancs = [ancNH3, ancH2O]
# for T in [551.1]:
#     for k in [1]:
#         pure = models[k]; anc = ancs[k]
#         rhoL, rhoV = pure.pure_VLE_T(T, anc.rhoL(T), anc.rhoV(T), 10)
#         rhovecL = np.array([0.0, 0]); rhovecL[k] = rhoL
#         rhovecV = np.array([0.0, 0]); rhovecV[k] = rhoV
#         plotting.plot_criticality_constT(
#             model=mutant,
#             rholim=(1,40000), rhoN=200,
#             T=T, ax=None, show=False
#         )
#         i = plotting.isotherm(mutant, T, rhovecL, rhovecV)
#         # i.info()
#         plt.plot(np.array(i['rhoL / mol/m^3'].tolist()).sum(axis=1), i['xL_0 / mole frac.'])
#         plt.plot(np.array(i['rhoV / mol/m^3'].tolist()).sum(axis=1), i['xV_0 / mole frac.'])
# plt.close()
# # quit()

models = [mutant, teqp.build_multifluid_model(['AMMONIA', 'WATER'], r'../REFPROP-interop/teqp'), teqp.AmmoniaWaterTillnerRoth()]
aliases = ['TW', 'RP10.0', 'Tillner-Roth']

# check_lowtemp_ammonia_virials()
# check_gas_PVT_just_B2(source='perfect gas')
# check_gas_PVT_just_B2(source='Harms-Watzenberg')
# check_gas_PVT_just_B2(source='Wormald')
# check_gas_PVT_just_B2(source='mutant', mutant=mutant)
# check_gas_PVT_just_B2(source='Tillner-Roth', mutant=teqp.AmmoniaWaterTillnerRoth())
check_virials(author='Harms-Watzenberg')
# plotting.plot_all_reducing_functions(models=models, aliases=aliases, yvar='v')
# failure_of_corresponding_states(mutant)
# back_calculate_alpharij_B12(mutant, source='Wormald')
# plotting.plot_all_dilute_neff(0.0, models=models, aliases=aliases)
# plotting.plot_all_dilute_neff(0.5, models=models, aliases=aliases)
# plotting.plot_all_dilute_neff(1.0, models=models, aliases=aliases)
# plot_all_critical_loci(models=models, aliases=aliases)

def badisotherms():
    T = 200
    rhor = CP.PropsSI('rhomolar_reducing', 'WATER')
    rho = np.geomspace(1e-10, CP.PropsSI('rhomolar_critical', 'WATER')*5, 10000)
    p = CP.PropsSI('P','T|phase_liquid',T,'Dmolar', rho, 'WATER')

    plt.plot(rho/rhor, p)
    plt.yscale('symlog')
    plt.show()
# badisotherms()

def cache_isotherms(meta, Tset, fname):
    isotherms = {}
    for T in Tset:
        k = 1; pure = modelH2O; anc = ancH2O
        try:
            rhoL, rhoV = pure.pure_VLE_T(T, anc.rhoL(T), anc.rhoV(T), 10)
        except BaseException as be:
            print(be)
            continue
        rhovecL = np.array([0.0, 0]); rhovecL[k] = rhoL
        rhovecV = np.array([0.0, 0]); rhovecV[k] = rhoV
        # print(T, rhovecL, rhovecV)
        i, j = plotting.isotherm(mutant, T, rhovecL, rhovecV, also_json=True)
        isotherms[T] = j
        print(T, len(j))
    del T
    p = dict(meta=meta, isotherms=isotherms)
    with open(fname, 'wb') as fp:
        fp.write(pickle.dumps(p))

df = load_VLE(*load_args, **(load_kwargs|{"apply_skip":False}))

def calc_VLE_deviations(df):
    meta = mutant.get_meta() if hasattr(mutant, 'get_meta') else 'Tillner-Roth(probably)'
    VLE_cache = '_vle_cache.pkl'
    Tset = sorted(set([round(T_, 3) for T_ in df['T / K']]))

    # Cache if needed
    def cache_isotherms_if_needed():
        if os.path.exists(VLE_cache):
            p = pickle.load(open(VLE_cache, 'rb'))
            if p['meta'] == meta and Tset == list(p['isotherms'].keys()):
                print('VLE isotherm cache hit')
                return
        print('VLE isotherm cache miss, building')
        cache_isotherms(meta, Tset, fname=VLE_cache)
    cache_isotherms_if_needed()
        
    p = pickle.load(open(VLE_cache, 'rb'))
    meta = p['meta']; isotherms = p['isotherms']
    print(meta)
    del p

    def add_p_from_comp(row):

        Tnom = round(row['T / K'], 3)
        i = pandas.DataFrame(isotherms[Tnom])
        # i.info()
        # print(i.head(3))
        
        rhomat = np.array(i['rhoL / mol/m^3'].tolist())
        rhoL1, rhoL2 = rhomat[:,0], rhomat[:,1]
        rhomat = np.array(i['rhoV / mol/m^3'].tolist())
        rhoV1, rhoV2 = rhomat[:,0], rhomat[:,1]
        
        # If has a liquid, interpolate for liquid concentrations
        p_bub = None
        x_1_bub = None

        x_1 = row['x_1 / mole frac.'] # from the measurement
        p = row['p / Pa'] # from the measurement
        if np.isfinite(x_1):
            x1vec = np.array(i['xL_0 / mole frac.'])
            try:
                z = np.array([x_1, 1-x_1])
                rhoL_1_0 = scipy.interpolate.interp1d(x1vec, rhoL1)(x_1)
                rhoL_2_0 = scipy.interpolate.interp1d(x1vec, rhoL2)(x_1)
                rhovecL_init = np.array([rhoL_1_0, rhoL_2_0])
                rhoL = np.sum(rhovecL_init)
                p_bub_init = rhoL*mutant.get_R(z)*Tnom*(1+mutant.get_Ar01(Tnom, rhoL, z))
                assert(abs(p_bub_init/row['p / Pa']-1) < 1)

                rhoV_1_0 = scipy.interpolate.interp1d(x1vec, rhoV1)(x_1)
                rhoV_2_0 = scipy.interpolate.interp1d(x1vec, rhoV2)(x_1)
                rhovecV_init = np.array([rhoV_1_0, rhoV_2_0])
                zdew = rhovecV_init/rhovecV_init.sum()
                p_dew_init = rhovecV_init.sum()*mutant.get_R(zdew)*Tnom*(1+mutant.get_Ar01(Tnom, rhovecV_init.sum(), zdew))
                assert(abs(p_dew_init/row['p / Pa']-1) < 1)

                # Do the calculation for p as a function of T and composition
                code, rhovecL, rhovecV = mutant.mix_VLE_Tx(Tnom, rhovecL_init, rhovecV_init, z, 1e-10, 1e-10, 1e-10, 1e-10, 10)
                p_bub = rhovecL.sum()*mutant.get_R(z)*Tnom*(1+mutant.get_Ar01(Tnom, rhovecL.sum(), z))

            except BaseException as be:
                print(Tnom, row['bibkey'], x_1, be)

            pvec = np.array(i['pL / Pa'])
            lnpvec = np.log(pvec)
            lnp = np.log(p)
            try:
                
                rhoL_1_0 = scipy.interpolate.interp1d(lnpvec, rhoL1)(lnp)
                rhoL_2_0 = scipy.interpolate.interp1d(lnpvec, rhoL2)(lnp)
                rhovecL_init = np.array([rhoL_1_0, rhoL_2_0])
                rhoL = np.sum(rhovecL_init)
                z = rhovecL_init/rhoL
                p_bub_init = rhoL*mutant.get_R(z)*Tnom*(1+mutant.get_Ar01(Tnom, rhoL, z))
                assert(abs(p_bub_init/row['p / Pa']-1) < 1)

                rhoV_1_0 = scipy.interpolate.interp1d(lnpvec, rhoV1)(lnp)
                rhoV_2_0 = scipy.interpolate.interp1d(lnpvec, rhoV2)(lnp)
                rhovecV_init = np.array([rhoV_1_0, rhoV_2_0])
                zdew = rhovecV_init/rhovecV_init.sum()
                p_dew_init = rhovecV_init.sum()*mutant.get_R(zdew)*Tnom*(1+mutant.get_Ar01(Tnom, rhovecV_init.sum(), zdew))
                assert(abs(p_dew_init/row['p / Pa']-1) < 1)

                # Do the calculation for p as a function of T and composition
                code, rhovecL, rhovecV = mutant.mix_VLE_Tx(Tnom, rhovecL_init, rhovecV_init, z, 1e-10, 1e-10, 1e-10, 1e-10, 10)

                # Do the calculation
                def p_bub_(z_1, rhovecL, rhovecV):
                    z = np.array([z_1, 1-z_1])
                    code, rhovecL, rhovecV = mutant.mix_VLE_Tx(Tnom, rhovecL, rhovecV, z, 1e-10, 1e-10, 1e-10, 1e-10, 10)
                    p_bub = rhovecL.sum()*mutant.get_R(z)*Tnom*(1+mutant.get_Ar01(Tnom, rhovecL.sum(), z))
                    return p_bub
                def objective(z_1, rhovecL, rhovecV):
                    return p_bub_(z_1, rhovecL, rhovecV)/p-1
                res = scipy.optimize.newton(objective, x_1, args=(rhovecL, rhovecV))
                x_1_bub_bis = res

                r = mutant.mix_VLE_Tp(Tnom, p, rhovecL_init, rhovecV_init)
                rhovecL_ = r.rhovecL;
                x_1_bub = rhovecL_[0]/sum(rhovecL_)
                if abs(x_1_bub - x_1_bub_bis) > 1e-6:
                    print(x_1_bub - x_1_bub_bis, rhovecL_init, rhovecL_)

            except BaseException as be:
                print('x(p) failed:', Tnom, x_1, be)
            
        # If has a vapor composition
        p_dew = None
        y_1_dew = None
        y_1 = row['y_1 / mole frac.']
        if np.isfinite(y_1):
            y1vec = i['xV_0 / mole frac.']
            try:
                rhoL_1_0 = scipy.interpolate.interp1d(y1vec, rhoL1)(y_1)
                rhoL_2_0 = scipy.interpolate.interp1d(y1vec, rhoL2)(y_1)
                rhoV_1_0 = scipy.interpolate.interp1d(y1vec, rhoV1)(y_1)
                rhoV_2_0 = scipy.interpolate.interp1d(y1vec, rhoV2)(y_1)
                z = np.array([y_1, 1-y_1])
                rhovecA = np.array([rhoV_1_0, rhoV_2_0])
                rhovecB = np.array([rhoL_1_0, rhoL_2_0])
                code, rhovecA, rhovecB = mutant.mix_VLE_Tx(Tnom, rhovecA, rhovecB, z, 1e-10, 1e-10, 1e-10, 1e-10, 10)
                p_dew = rhovecA.sum()*mutant.get_R(z)*Tnom*(1+mutant.get_Ar01(Tnom, rhovecA.sum(), z))
            except BaseException as be:
                print(Tnom, row['bibkey'], y_1, len(i), be)

        return p_bub, p_dew, x_1_bub, y_1_dew
        
    df[['pbub / Pa', 'pdew / Pa', 'x_1(calc) / mole frac.', 'y_1(calc) / mole frac.']] = df.apply(add_p_from_comp, axis=1, result_type='expand')
    return df

# df = calc_VLE_deviations(df)
# df = df[~df['bibkey'].isin(['Rizvi-JCED-1987'])]

# df['perr / %'] = 100*(1 - df['pbub / Pa']/df['p / Pa'])
# for bib, gp in df.groupby(['bibkey']):
#     plt.plot(gp['T / K'], gp['perr / %'], marker='o', lw=0, label=bib)
# plt.axhline(0, zorder=-100, color='k')
# plt.xlim(180, 630)
# plt.axvline(CP.PropsSI('Tcrit','Ammonia'))
# plt.legend()
# plt.show()

# df['x_1 err'] = df['x_1 / mole frac.'] - df['x_1(calc) / mole frac.']
# df['x_1 err2'] = calc_errVLE_x(mutant, df)

# for bib, gp in df.groupby(['bibkey']):
#     plt.plot(gp['T / K'], gp['x_1 err'], marker='o', lw=0, label=bib)
#     print(bib, np.mean(np.abs(gp['x_1 err'])))

# print('overall x_1 average abs diff:', np.mean(np.abs(df['x_1 err'])))
# plt.axhline(0, zorder=-100, color='k')
# plt.xlim(180, 630)
# # plt.yscale('symlog', linthresh=0.04)
# plt.axvline(CP.PropsSI('Tcrit', 'Ammonia'))
# linthresh = 0.04
# for y in [linthresh, -linthresh]:
#     plt.gca().axhline(y)
# plt.legend()
# plt.show()

# df['perr / %'] = 100*(1 - df['pdew / Pa']/df['p / Pa'])
# for bib, gp in df.groupby(['bibkey']):
#     plt.plot(gp['T / K'], gp['perr / %'], marker='o', lw=0, label=bib)
# plt.axhline(0, zorder=-100, color='k')
# plt.legend()
# plt.show()

# df.to_csv('VLE_with_devs.csv', index=False)

# Critical curves
cr = plotting.calc_critical_curves(model=mutant, basemodel=basemodel, ipure=0, integration_order=5)
cr['z_0 / mole frac.'] = cr['rho0 / mol/m^3']/(cr['rho0 / mol/m^3']+cr['rho1 / mol/m^3'])
# plt.plot(cr['T / K'], cr['p / Pa']/1e6)
# plt.gca().set(xlabel='$T$ / K', ylabel='$p$ / MPa')
# df = pandas.read_csv('NH3H2O/upstream/crit.csv')
# plt.plot(df['T / K'], df['p / Pa']/1e6, 'o')
# plt.show()

df = load_PVT(*load_args, **load_kwargs)
df = df[df['z_1 / mole frac.'] > 0]
df['rhoerr / %'] = calc_errrho(model=mutant, df=df, iterate=True)
print(dir(mutant))
df['T_red / K'] = df.apply(lambda row: mutant.get_Tr(np.array([row['z_1 / mole frac.'], 1-row['z_1 / mole frac.']])), axis=1)
df['rho_red / mol/m^3'] = df.apply(lambda row: mutant.get_rhor(np.array([row['z_1 / mole frac.'], 1-row['z_1 / mole frac.']])), axis=1)
df.to_csv('rhoerr.csv', index=False)

# df = df[df.bibkey.isin(['Sakabe-JCT-2008','Munakata-JCT-2002','Kondo-JCT-2002'])]
# df = df[df.bibkey.isin(['HarmsWatzenberg-VDI-1995'])]
# df = df[df['rho / mol/m^3'] < 2000]
# df = df[pandas.isnull(df.skip)]
fig, (axT, axD, axx, axP) = plt.subplots(1,4,figsize=(12, 4),sharey=True)
for bibkey, gp in df.groupby('bibkey'):
    axT.scatter(gp['T / K'], gp['rhoerr / %'], marker='o', label=bibkey)
    axD.scatter(gp['rho / mol/m^3']/1e3, gp['rhoerr / %'], marker='o', label=bibkey)
    axx.scatter(gp['z_1 / mole frac.'], gp['rhoerr / %'], marker='o', label=bibkey)
    axP.scatter(gp['p / Pa']/1e6, gp['rhoerr / %'], marker='o', label=bibkey)
    print(bibkey, np.mean(np.abs(gp['rhoerr / %'])), 'rhoerr%')
axT.set(xlabel=r'$T$ / K',ylabel=r'$\Delta\rho$ / %')
axD.set(xlabel=r'$\rho$ / mol/dm$^3$')
axx.set(xlabel=r'$z_1$ / mole frac.')
axP.set(xlabel=r'$p$ / MPa')
for ax in axT, axD, axx, axP:
    ax.axhline(0, zorder=-100)
    for y in [-0.75, 0.75]:
        ax.axhline(y,dashes=[2,2],color='k')
for vc in basemodel.get_vcvec():
    axD.axvline(1/vc/1e3, dashes=[2,2])
for Tc in basemodel.get_Tcvec():
    axT.axvline(Tc, dashes=[2,2])
plt.legend(loc='best')
plt.tight_layout(pad=0.2)
plt.savefig('PVT_deviations.pdf')
plt.show()

fig, (axT, axD) = plt.subplots(1, 2, figsize=(6, 3),sharey=True)
for bibkey, gp in df.groupby('bibkey'):
    axT.scatter(gp['T / K']/gp['T_red / K'], gp['rhoerr / %'], marker='o', label=bibkey)
    axD.scatter(gp['rho / mol/m^3']/gp['rho_red / mol/m^3'], gp['rhoerr / %'], marker='o', label=bibkey)
axT.set(xlabel=r'$T/T_{\rm red}(x)$', ylabel=r'$\Delta\rho$ / %')
axD.set(xlabel=r'$\rho/\rho_{\rm red}(x)$')
for ax in axT, axD:
    ax.axhline(0, zorder=-100)
    for y in [-0.75, 0.75]:
        ax.axhline(y,dashes=[2,2],color='k')
axT.legend(loc='best', fontsize=4)
plt.tight_layout(pad=0.2)
plt.savefig('PVT_deviations_reduced.pdf')
plt.close()
del df

df = load_PVT(*load_args, **load_kwargs)
df = df[df.bibkey.isin(['Hnedkovsky-JCT-1996'])]
df['rhoerr / %'] = calc_errrho(model=mutant, df=df, iterate=True)
df.to_csv('badHnedkovsky_for_investigation.csv')
fig, (axD, axT) = plt.subplots(1,2,figsize=(10, 5), sharey=True)
for bibkey, gp in df.groupby('bibkey'):
    axD.scatter(gp['rho / mol/m^3']/1e3, gp['rhoerr / %'], marker='o', label=bibkey)
    axT.scatter(gp['T / K']-CP.PropsSI('Tcrit','Water'), gp['rhoerr / %'], marker='o', label=bibkey)
    print(bibkey, np.mean(np.abs(gp['rhoerr / %'])),'rhoerr%')
axD.set(xlabel=r'$\rho$ / mol/dm$^3$', ylabel=r'$\Delta \rho$ / %')
axT.set(xlabel=r'$T-T_{\rm crit, H_2O}$ / K')
for ax in axD, axT:
    ax.axhline(0, zorder=-100)
for vc in basemodel.get_vcvec():
    axD.axvline(1/vc/1e3, dashes=[2,2])
plt.legend(loc='best')
plt.savefig('PVT_Hnedkovsky_deviations.pdf')
plt.close()
del df

T = 298.15 # K
R = 8.31446261815324
rhovec = np.geomspace(1, 77000, 1000)
for z_1 in [0.0, 0.01, 0.03, 0.05]:
    p = [rho*R*T*(1+mutant.get_Ar01(T, rho, np.array([z_1, 1-z_1]))) for rho in rhovec]
    plt.plot(rhovec, p)
rhoL, p = [CP.PropsSI(k,'T',T,'Q',0,'Water') for k in ['Dmolar','P']]
rhoV, p = [CP.PropsSI(k,'T',T,'Q',1,'Water') for k in ['Dmolar','P']]
plt.plot(rhoL, p, '*')
plt.plot(rhoV, p, '*')
plt.axhline(p, dashes=[2,2], label='sat')
plt.legend(loc='best')
plt.gca().set(xlabel=r'$\rho$ / mol/m$^3$', ylabel='$p$ / Pa')
plt.title(r'$\rho$ are two-phase, enormous pressure deviations')
plt.yscale('symlog', linthresh=10)
plt.close()

df = load_PVT_P(*load_args, **load_kwargs)
df = df[df.weight > 0]
df['perr / %'] = calc_errrho_devp(model=mutant, df=df)
fig, (axD, axP) = plt.subplots(1,2,figsize=(10, 5), sharey=True)
for bibkey, gp in df.groupby('bibkey'):
    axD.scatter(gp['rho / mol/m^3']/1e3, gp['perr / %'], marker='o', label=bibkey)
    axP.scatter(gp['p / Pa']/1e6, gp['perr / %'], marker='o', label=bibkey)
    print(bibkey, np.mean(np.abs(gp['perr / %'])), 'perr%')
axD.set(xlabel=r'$\rho$ / mol/dm$^3$', ylabel=r'$\Delta p$ / %')
axP.set(xlabel=r'$p$ / MPa')
for ax in axD, axP:
    ax.axhline(0, zorder=-100)
for vc in basemodel.get_vcvec():
    axD.axvline(1/vc/1e3, dashes=[2,2])
plt.legend(loc='best')
plt.savefig('PVT_P_deviations.pdf')
plt.close()
del df

# df = load_PVT(*load_args, **load_kwargs)
# df['rhoerr / %'] = calc_errrho(model=mutant, df=df)
# df = df[df['method'] == 'Burnett']
# df['Z'] = df['p / Pa']/(df['rho / mol/m^3']*8.31446261815324*df['T / K'])
# for bibkey, gp in df.groupby('bibkey'):
#     plt.scatter(gp['rho / mol/m^3'], (gp['Z']-1)/gp['rho / mol/m^3'], marker='o', label=bibkey, c=gp['T / K'])
# plt.legend(loc='best')
# plt.show()

df = load_SOS(*load_args, **load_kwargs)
for model, alias in zip(models, aliases):
    df['werr / %'] = calc_errSOS(model=model, df=df)
    fig, (axT, axp, axx) = plt.subplots(1,3,figsize=(10,5), sharey=True)
    for bibkey, gp in df.groupby('bibkey'):
        # if bibkey == 'UW': continue
        axT.plot(gp['T / K'], gp['werr / %'], 'o', label=bibkey)
        axp.plot(gp['p / Pa']/1e6, gp['werr / %'], 'o', label=bibkey)
        axx.plot(gp['z_1 / mole frac.'], gp['werr / %'], 'o', label=bibkey)
        print(bibkey, np.mean(np.abs(gp['werr / %'])), 'werr%')
    axT.set(ylabel='$\Delta w$ / %', xlabel='$T$ / K')
    axp.set(ylabel='$\Delta w$ / %', xlabel='$p$ / MPa')
    axx.set(ylabel='$\Delta w$ / %', xlabel='$z_1$ / mole frac.')
    plt.legend(loc='best')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'SOS_deviations_{alias}.pdf')
    plt.savefig(f'SOS_deviations_{alias}.png', dpi=300)
    plt.close()
del df

# VLE
df = load_VLE(*load_args, **(load_kwargs|{"apply_skip":False}))
print('available', set(df['bibkey']))
df = df[df['bibkey'].isin(['Smolen-JCED-1991', 'Sassen-JCED-1990'])]
# df = df[(df['T / K'] > 10+273.15) & (df['T / K'] < 40+273.15)]
# df = df[(df['T / K'] < 400)]
for bibkey, gp in df.groupby('bibkey'):
    plt.scatter(gp['x_1 / mole frac.'], gp['p / Pa']/1e6, marker='o', c=gp['T / K'], vmin = 273, vmax=700, label=bibkey)
    sc = plt.scatter(gp['y_1 / mole frac.'], gp['p / Pa']/1e6, marker='^', c=gp['T / K'], vmin = 273, vmax=700, label=bibkey)
models = [modelNH3, modelH2O]
ancs = [ancNH3, ancH2O]
for T in list(set(df['T / K'])) + [450, 500, 551.1, 575, 599.67]:
    for k in [0, 1]:
        pure = models[k]; anc = ancs[k]
        try:
            rhoL, rhoV = pure.pure_VLE_T(T, anc.rhoL(T), anc.rhoV(T), 10)
        except:
            continue
        
        rhovecL = np.array([0.0, 0]); rhovecL[k] = rhoL
        rhovecV = np.array([0.0, 0]); rhovecV[k] = rhoV
        # z = rhovecL/np.sum(rhovecL)
        # z[k] = 1-1e-6
        # z[1-k] = 1e-6
        # print(T, rhovecL, rhovecV, z, 1e-10, 1e-10, 1e-10, 1e-10, 10)
        # rhovecL, rhovecV = pure.mix_VLE_Tx(T, rhovecL, rhovecV, z, 1e-10, 1e-10, 1e-10, 1e-10, 10)
        print(rhovecL, rhovecV)
        i = plotting.isotherm(mutant, T, rhovecL, rhovecV)
        print(len(i))
        # if abs(T-551.1) < 1: print(rhovecL, rhovecV)
        plt.plot(i['xL_0 / mole frac.'], i['pL / Pa']/1e6, color = sc.to_rgba(T))
        plt.plot(i['xV_0 / mole frac.'], i['pL / Pa']/1e6, color = sc.to_rgba(T), dashes=[2,2])

plt.legend(loc='best')
plt.plot(cr['z_0 / mole frac.'], cr['p / Pa']/1e6, c='k')
df = pandas.read_csv('NH3H2O/upstream/crit.csv')
plt.plot(df['z_1 / mole frac.'], df['p / Pa']/1e6, '*')

plt.yscale('log')
cb = plt.colorbar()
cb.set_label('$T$ / K')
plt.xlim(0, 1)
plt.gca().set(xlabel='$x_1,y_1$ / mole frac.', ylabel='$p$ / MPa')
plt.savefig('VLE.pdf')
plt.close()