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
        def sort_paths(paths):
            return list(zip(*sorted([(int(step.split('.')[0].split('step')[1]), step) for step in paths])))[1]

        if self.path.endswith('.zip'):
            raise ValueError("zipfiles are not supported (don't compress as well as LZMA)'")
        elif self.path.endswith('.tar.xz'):
            with tarfile.open(self.path, mode='r:xz') as tar:
                stepfiles = []
                for info in tar.getmembers():
                    filename = info.name
                    if 'step' in filename and '.json' in filename and uid in filename:
                        stepfiles.append(json.load(tar.extractfile(info)))
        else:
            paths = [f for f in glob.glob(self.path+'/*.json') if 'step' in f and uid in f]
            paths = sort_paths(paths)
            stepfiles = [json.load(open(f)) for f in paths]
        return stepfiles

def build_mutant(teqp_names : List[str], path : str, s : dict, *, flags=None):
    if flags is None:
        flags = {'estimate': 'Lorentz-Berthelot'}
    basemodel = teqp.build_multifluid_model(teqp_names, path, path+'/dev/mixtures/mixture_binary_pairs.json', flags)
    return teqp.build_multifluid_mutant(basemodel, s), basemodel

def calc_critical_curves(*, model, basemodel, ipure, show=False):
    Tcvec = basemodel.get_Tcvec()
    vcvec = basemodel.get_vcvec()
    rhovec = np.array([0.0]*2)
    rhovec[ipure] = 1/vcvec[ipure]
    opt = teqp.TCABOptions()
    opt.polish = True
    df = pandas.DataFrame(teqp.trace_critical_arclength_binary(model, Tcvec[ipure], rhovec, '', opt))
    return df

def isotherm(model, T, rhovecL, rhovecV, also_json=False):
    opt = teqp.TVLEOptions(); opt.polish=True; opt.integration_order=5
    o = teqp.trace_VLE_isotherm_binary(model, T, rhovecL, rhovecV, opt)
    df = pandas.DataFrame(o)
    def calcz0(row, key):
        z = np.array(row[key])
        z0 = z[0]/z.sum()
        return z0
    df['x0 / mole frac.'] = df.apply(calcz0, axis=1, key='rhoL / mol/m^3')
    df['y0 / mole frac.'] = df.apply(calcz0, axis=1, key='rhoV / mol/m^3')
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