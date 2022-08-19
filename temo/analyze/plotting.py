from typing import List
import tarfile, zipfile, json, glob

import pandas 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import teqp

class ResultsParser:

    def __init__(self, path):
        """
        path: path to a folder or a .7z or .zip archive of a folder
        """
        self.path = path
        self.dfresults = self.assess_from_path(path)

    def assess_from_path(self, path):
        if path.endswith('.tar.xz'):
            with tarfile.open(path, mode='r:xz') as tar:
                results = []
                for info in tar.getmembers():
                    filename = info.name
                    if 'step' not in filename and '.json' in filename:
                        results.append(json.load(tar.extractfile(info)))

        elif path.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                results = []
                for filename in z.namelist():
                    if 'step' not in filename and '.json' in filename:
                        with z.open(filename) as myfile:
                            results.append(json.load(myfile))

        else:
            paths = [f for f in glob.glob(path+'/*.json') if 'step' not in f]
            results = [json.load(open(f)) for f in paths]

        return pandas.DataFrame(results)

    def to_csv(self, *, prefix):
        self.dfresults.to_csv(prefix+'.csv', index=False)
    
    def get_all_uid(self, path):
        return [f.replace('_step0.json','') for f in glob.glob(path+'/*step0.json')]

    def get_lowest_cost_uid(self, *, prefilter=None):
        """ 
        prefilter: a function taking the dataframe, returning a mask array (for instance to throw out some solutions)

        returns the uid 
        """
        if prefilter:
            df = self.dfresults[prefilter(self.dfresult)]
        else:
            df = self.dfresults
        imin = np.argmin(np.array(df['cost']))
        return df.iloc[imin]['uid']

    def get_stepfiles(self, uid):
        
        if self.path.endswith('.zip'):
            raise ValueError("zipfiles are not supported (don't compress as well as LZMA)'")
        elif self.path.endswith('.tar.xz'):
            with tarfile.open(self.path, mode='r:xz') as tar:
                stepfiles = []
                for info in tar.getmembers():
                    filename = info.name
                    if 'step' in filename and '.json' in filename and uid in filename:
                        try:
                            stepfiles.append(json.load(tar.extractfile(info)))
                        except json.decoder.JSONDecodeError:
                            print(f'Unable to parse {filename}')
                stepfiles.sort(key=lambda x: -x['cost'])

        else:
            def sort_paths(paths):
                return list(zip(*sorted([(int(step.split('.')[0].split('step')[1]), step) for step in paths])))[1]
            paths = [f for f in glob.glob(self.path+'/*.json') if 'step' in f and uid in f]
            paths = sort_paths(paths)
            stepfiles = [json.load(open(f)) for f in paths]
        return stepfiles

def build_mutant(teqp_names : List[str], path : str, s : dict, *, flags=None):
    if flags is None:
        flags = {'estimate': 'Lorentz-Berthelot'}
    basemodel = teqp.build_multifluid_model(teqp_names, path, path+'/dev/mixtures/mixture_binary_pairs.json', flags)
    return teqp.build_multifluid_mutant(basemodel, s), basemodel

def calc_critical_curves(*, model, basemodel, ipure, integration_order, polish_reltol_T = 100, polish_reltol_rho=100):
    Tcvec = basemodel.get_Tcvec()
    vcvec = basemodel.get_vcvec()
    opt = {"alternative_pure_index": ipure, "alternative_length": 2}
    [T0, rho0] = model.solve_pure_critical(Tcvec[ipure], 1.0/vcvec[ipure], opt)
    rhovec0 = np.array([0.0, 0])
    rhovec0[ipure] = rho0

    opt = teqp.TCABOptions()
    opt.polish = True
    opt.integration_order = integration_order
    # opt.rel_err = 1e-7
    opt.init_dt = 100
    opt.max_dt = 1000
    opt.polish_reltol_T = polish_reltol_T
    opt.polish_reltol_rho = polish_reltol_rho
    df = pandas.DataFrame(model.trace_critical_arclength_binary(T0, rhovec0, '', opt))
    return df

def plot_criticality(*, model, Tlim, rholim, z_1, TN=100, rhoN=100, ax=None, show=True):
    z = np.array([z_1, 1-z_1])
    Tvec = np.linspace(*Tlim, TN)
    rhovec = np.geomspace(*rholim, rhoN)
    TT, DD = np.meshgrid(Tvec, rhovec)
    Nrow, Ncol = TT.shape
    C1 = np.zeros_like(TT)
    C2 = np.zeros_like(TT)
    for i in range(Nrow):
        for j in range(Ncol):
            C1[i,j], C2[i,j] = model.get_criticality_conditions(TT[i,j], DD[i,j]*z)
    if ax is None:
        ax = plt.gca()
    ax.contour(DD, TT, C1, levels=[0], colors='k')
    ax.contour(DD, TT, C2, levels=[0], colors='grey', linestyles=['dashed'])
    ax.set(xlabel=r'$\rho$ / mol/m$^3$', ylabel='$T$ / K')
    ax.set_title(f'$z_1$: {z_1:0.5f} mole frac.')
    if show:
        plt.show()

def plot_criticality_constT(*, T, model, zlim=(0,1), rholim, zN=100, rhoN=100, ax=None, show=True):
    zvec = np.linspace(*zlim, zN)
    rhovec = np.geomspace(*rholim, rhoN)
    ZZ, DD = np.meshgrid(zvec, rhovec)
    Nrow, Ncol = ZZ.shape
    C1 = np.zeros_like(ZZ)
    C2 = np.zeros_like(ZZ)
    for i in range(Nrow):
        for j in range(Ncol):
            z_1 = ZZ[i,j]
            z = np.array([z_1, 1-z_1])
            C1[i,j], C2[i,j] = model.get_criticality_conditions(T, DD[i,j]*z)
    if ax is None:
        ax = plt.gca()
    ax.contour(DD, ZZ, C1, levels=[0], colors='k')
    ax.contour(DD, ZZ, C2, levels=[0], colors='grey', linestyles=['dashed'])
    ax.set(xlabel=r'$\rho$ / mol/m$^3$', ylabel='$z_1$ / mole frac.')
    ax.set_title(f'$T$: {T:0.5f} K')
    if show:
        plt.show()

def isotherm(model, T, rhovecL, rhovecV, also_json=False, crit_threshold=5e-8) -> pandas.DataFrame:
    opt = teqp.TVLEOptions(); opt.polish=True; opt.integration_order=5; opt.calc_criticality = True
    opt.terminate_unstable = True
    o = model.trace_VLE_isotherm_binary(T, rhovecL, rhovecV, opt)
    df = pandas.DataFrame(o)
    def calcz0(row, key):
        z = np.array(row[key])
        z0 = z[0]/z.sum()
        return z0
    df['x0 / mole frac.'] = df.apply(calcz0, axis=1, key='rhoL / mol/m^3')
    df['y0 / mole frac.'] = df.apply(calcz0, axis=1, key='rhoV / mol/m^3')
    df['too_critical'] = df.apply(lambda row: (abs(row['crit. conditions L'][0]) < crit_threshold), axis=1)
    first_too_critical = np.argmax(df['too_critical'])
    df = df.iloc[0:(first_too_critical if first_too_critical else len(df))]
    if also_json:
        return df, o 
    else:
        return df

def plot_px_history(*, root, uid, stepfiles, override=None):
    previous_cost = 1e99
    fname = ('' if not override else override) + f'histpx{uid[0:6]}.pdf'
    with PdfPages(fname) as PDF:
        for N, stepfile in enumerate(stepfiles):
            N += 1

            cost = stepfile['cost']
            if previous_cost > 1e98 or cost < previous_cost:
                print(cost)

                fluids = ('Neon','Hydrogen')
                s = stepfile['model']
                if override:
                    s = override

                model, base, short = build_mutant(fluids, teqp.get_datapath(), s)
                fig, ax = plt.subplots(1, 1)
                
                for T in [24.6, 26.0, 26.3, 27.15, 28.12, 29.0, 31.51, 33.73, 34.6, 37.6, 39.6, 42.49]:
                    df = isotherm(model, fluids, 0, T)
                    line, = ax.plot(df['x_1 / mole frac.'], df['pL / Pa']/1e6, lw=1, label=T)
                    ax.plot(df['y0 / mole frac.'], df['pL / Pa']/1e6, dashes=[2,2], color=line.get_color(), lw=1)

                    df = pandas.read_csv('VLE.csv')
                    df = df[np.abs(df['T / K']-T) < 1e-1]
                    # df.info()
                    for Authorname, gp in df.groupby('Authorname'):
                        # print(T, Authorname)
                        ax.plot(gp['x0 / mole frac.'], gp['p / MPa'], marker='o', color=line.get_color(), lw=0, ms=1)
                        ax.plot(gp['y0 / mole frac.'], gp['p / MPa'], marker='d', color=line.get_color(), lw=0, ms=1)

                plt.gca().set(xlabel=r'$x_{{\rm Ne}}$ / mol/mol', ylabel='$p$ / MPa', ylim=(0,3))
                plt.legend(loc='best')
                plt.title(f'{N} | C$\$$: {cost:0.4f}')
                PDF.savefig(plt.gcf())
                plt.close()
                previous_cost = cost

def plot_cost_history(basemodel, *, stepfiles, override=None):
    iter_history = []; cost_history = []
    for N, stepfile in enumerate(stepfiles):
        iter_history.append(N+1)
        cost_history.append(stepfile['cost'])
    plt.plot(iter_history, cost_history)
    plt.xscale('log')
    plt.show()

def plot_critical_locus_history(basemodel, *, stepfiles, override=None, dfcr=None, ylim=None):
    previous_cost = 1e99
    fname = ('' if not override else override) + f'histcrit.pdf'
    with PdfPages(fname) as PDF:
        for N, stepfile in enumerate(stepfiles):
            N += 1
            cost = stepfile['cost']
            if previous_cost > 1e98 or cost < previous_cost:
                print(cost)
                mutant = teqp.build_multifluid_mutant(basemodel, stepfile['model'])
                cr = calc_critical_curves(model=mutant, basemodel=basemodel, ipure=0, integration_order=1)
                print(len(cr))
                fig, ax = plt.subplots(1, 1)
                cr['z_0 / mole frac.'] = cr['rho0 / mol/m^3']/(cr['rho0 / mol/m^3']+cr['rho1 / mol/m^3'])
                ax.plot(cr['z_0 / mole frac.'], cr['p / Pa']/1e6)

                if dfcr is not None:
                    ax.plot(dfcr['z_1 / mole frac.'], dfcr['p / Pa']/1e6, 'o')

                plt.gca().set(xlabel=r'$x_{1}$ / mole frac.', ylabel='$p$ / MPa')
                # plt.legend(loc='best')
                plt.title(f'{N} | C$\$$: {cost:0.4f}')
                if ylim:
                    plt.ylim(*ylim)
                PDF.savefig(plt.gcf())
                plt.close()
                previous_cost = cost