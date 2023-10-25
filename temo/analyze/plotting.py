from typing import List, Dict
import tarfile, zipfile, json, glob, os

import pandas 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import teqp

class ResultsParser:

    def __init__(self, path):
        """
        path: path to a folder or a .tar.xz or .zip archive of a folder
        """
        self.path = path
        self.jsonresults = self.assess_from_path(path)
        self.dfresults = pandas.DataFrame(self.jsonresults)

    def get_result(self, uid: str):
        """
        Get a particular run result, given by its uid
        """
        for result in self.jsonresults:
            if result['uid'] == uid:
                return result
        raise ValueError

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
            if not os.path.isdir(path):
                raise ValueError("this is not a valid path: " + path)

            paths = [f for f in glob.glob(path+'/*.json') if 'step' not in f]
            results = [json.load(open(f)) for f in paths]

        return results

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
        """
        Args:
            uid: The unique identifier for the run
        """
        
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
        
    def get_fitdata_df(self, key, **kwargs):
        """ Return a selected DataFrame from the ``fitdataroot`` folder in the archive 
        Args:
            key: The search string that should be in the filename to be pulled from the ``fitdataroot`` folder in the archive

        Usage: provide 'SOS' for key to obtain the DataFrame for SOS.csv file, for instance

        Good options for key are: 'VLE','SOS','PVT', etc.
        """
        if self.path.endswith('.tar.xz'):
            with tarfile.open(self.path, mode='r:xz') as tar:
                for info in tar.getmembers():
                    filename = info.name
                    if 'fitdataroot' in filename and key in filename:
                        return pandas.read_csv(tar.extractfile(info), **kwargs)
        else:
            raise ValueError("zipfiles are not supported (don't compress as well as LZMA)'")
    
class PairMinFilter:
    def __init__(self, pair):
        self.pair = tuple(pair)

    def __call__(self, df):
        # Keep only rows that match the pair
        matches_pair = df.apply(lambda row: tuple(row['pair']) == self.pair, axis=1)
        df = df[matches_pair].copy()
        
        # And return the lowest cost result
        return pandas.DataFrame([df.sort_values(by='cost').iloc[0]])
    
class UidFilter:
    def __init__(self, uid):
        self.uid = uid

    def __call__(self, df):
        # Keep only the row that matches the given uid
        matches_uid = df.apply(lambda row: tuple(row['pair']) == self.uid, axis=1)
        df = df[matches_uid].copy()
        return df 

def build_mutant(teqp_names : List[str], path : str, spec: dict, *, flags=None):
    if flags is None:
        flags = {'estimate': 'Lorentz-Berthelot'}
    basemodel = teqp.build_multifluid_model(teqp_names, path, path+'/dev/mixtures/mixture_binary_pairs.json', flags)
    basemodels = [teqp.build_multifluid_model([name], path, path+'/dev/mixtures/mixture_binary_pairs.json', flags) for name in teqp_names]

    mutant = teqp.build_multifluid_mutant(basemodel, spec)
    teqp.attach_model_specific_methods(mutant)
    return mutant, basemodel, basemodels

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
    opt.terminate_unstable = True; opt.max_steps=200; 

    o = model.trace_VLE_isotherm_binary(T, rhovecL, rhovecV, opt)
    df = pandas.DataFrame(o)
    # df['too_critical'] = df.apply(lambda row: (abs(row['crit. conditions L'][0]) < crit_threshold), axis=1)
    # first_too_critical = np.argmax(df['too_critical'])
    # df = df.iloc[0:(first_too_critical if first_too_critical else len(df))]
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

def plot_critical_locus_history(basemodel, *, stepfiles, override=None, dfcr=None, ylim=None):
    previous_cost = 1e99
    fname = ('' if not override else override) + f'histcrit.pdf'
    with PdfPages(fname) as PDF:
        for N, stepfile in enumerate(stepfiles):
            N += 1
            cost = stepfile['cost']
            if previous_cost > 1e98 or cost < previous_cost:
                # print(cost)
                mutant = teqp.build_multifluid_mutant(basemodel, stepfile['model'])

                cr = calc_critical_curves(model=mutant, basemodel=basemodel, ipure=0, integration_order=5)
                # print(len(cr))
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

def plot_all_dilute_neff(z_1, *, models, aliases):
    assert(len(models) == len(aliases))
    Tvec = np.geomspace(250, 20000)
    for model, alias in zip(models, aliases):
        z = np.array([z_1, 1-z_1])
        neff = [model.get_neff(T, 1e-6, z) for T in Tvec]
        plt.plot(Tvec, neff, label=alias)
    plt.xscale('log')
    plt.legend(loc='best')
    plt.ylim(0,20)
    plt.gca().set(xlabel='$T$ / K', ylabel=r'$n_{\rm eff}$')
    plt.show()

def plot_all_reducing_functions(*, models, aliases, yvar = 'rho'):
    assert(len(models) == len(aliases))
    Tvec = np.geomspace(250, 20000)
    fig, (axT, axv) = plt.subplots(2,1,sharex=True)
    for model, alias in zip(models, aliases):
        z1vec = np.linspace(1e-10,1.0-1e-10,1000)
        Tr = [model.get_Tr(np.array([z_1,1-z_1])) for z_1 in z1vec]
        rhor = np.array([model.get_rhor(np.array([z_1,1-z_1])) for z_1 in z1vec])
        axT.plot(z1vec, Tr, label=alias)
        if yvar == 'rho':
            axv.plot(z1vec, rhor, label=alias)
        else:
            axv.plot(z1vec, 1/rhor*1e6, label=alias)
    axT.legend(loc='best')
    axT.set(ylabel=r'$T_{\rm red}$ / K')
    if yvar == 'rho':
        axv.set(xlabel=r'$z_1$ / mole frac.', ylabel=r'$\rho_{\rm red}$ / mol/m$^3$')
    else:
        axv.set(xlabel=r'$z_1$ / mole frac.', ylabel=r'$v_{\rm red}$ / cm$^3$/mol')
    plt.savefig('reducing_functions.pdf')
    plt.show()

def get_rhovecLV_guess(basemodel, T, ipure):
    # This assumes a multifluid model
    anc = basemodel.build_ancillaries()
    rhoLanc, rhoVanc = anc.rhoL(T), anc.rhoV(T)
    # VLE polish
    rhoL, rhoV = basemodel.pure_VLE_T(T, rhoLanc, rhoVanc, 10)

    rhovecL = np.array([0.0, 0]); rhovecL[ipure] = rhoL
    rhovecV = np.array([0.0, 0]); rhovecV[ipure] = rhoV
    
    return rhovecL, rhovecV

class ModelAssessmentPlotter:

    stepfiles: List[Dict]
    last_stepfile: Dict 

    def __init__(self, result_path, result_filter):
        """
        Args:
            result_path: The path to the results archive, strongly rtecommended to work from a .tar.xz archive, also .zip or path to folder (unzipped) are allowed
            result_filter: A function that filters from the results down to the result that will be used for further post-processing
        """
        self.results = ResultsParser(result_path)
        self.dfresults = result_filter(self.results.dfresults)
        if len(self.dfresults) != 1:
            raise ValueError("Result DataFrame must be one element in length after filtering; current length is "+str(len(self.dfresults)))
        # Get the unique identifier for the run
        self.uid = self.dfresults.uid.iloc[0]
        self.pair = self.dfresults['pair'].iloc[0]
        # Extract the stepfiles and store in the class
        self.stepfiles = self.results.get_stepfiles(self.uid)
        self.last_stepfile = self.stepfiles[-1]
        self.model, self.basemodel, self.basemodels = build_mutant(self.pair, path=teqp.get_datapath(), spec=self.last_stepfile['model'])
    
    def plot_cost_history(self, *, ax, stepfiles=None):
        """
        Plot the history of the cost function over the course of the optimization
        
        Args:
            ax: The axis to plot onto
            stepfiles (optional): The stepfiles, provided as a list of JSON instances
        """
        iter_history = []; cost_history = []
        if stepfiles is None:
            stepfiles = self.stepfiles
        for N, stepfile in enumerate(stepfiles):
            iter_history.append(N+1)
            cost_history.append(stepfile['cost'])
        ax.plot(iter_history, cost_history)
        ax.set_xscale('log')
        ax.set_xlabel('Iteration #')
        ax.set_ylabel('Cost function')

    def plot_B12(self, *, ax, z1_comps, Trange: List[float], labels: List[str], model=None):
        """ 
        Plot the second cross virial coefficient B_12
    
        Args:
            ax: the axis onto which to plot
            z1_comps: the list of compositions of the first component for which B12 curves are desired
            Trange: the two-element list of min and max temperature
            labels (optional): the label for each trace
            model (optional): the teqp.AbstractModel instance, or the default if not provided
        """
        # Add method for adding labels
        if labels is None:
            labels = ['' for i in range(len(z1_comps))]

        if model is None:
            model = self.model

        for comp, label in zip(z1_comps, labels):
            z = np.array([comp, 1-comp])
            Tvec = np.linspace(*Trange)
            assert(len(Trange)==2)
            B12 = np.array([model.get_B12vir(T_, z) for T_ in Tvec])
            ax.plot(Tvec, B12, label=label)
        ax.set_xlabel(r'$T$ / K')
        ax.set_ylabel(r'$B_{12}$ / cm$^3$/mol')
        ax.legend(loc='best')

    def plot_binary_VLE_isotherms(self, *, ax, Tvec: List[float], cmap, ipure, model=None, basemodel=None, options: Dict = None, plot_kwargs: Dict = {}):
        """ 
        Args:
            ax: the axis onto which to plot
            Tvec: the iterable containing the temperatures for which isotherms are desired
            cmap: the callable with method to_rgba(T) that will be used to determine the color of the trace
            ipure: the index, in {0,1}, that is the fluid from which the trace starts
            model (optional): the teqp.AbstractModel instance, or the default if not provided
            basemodel (optional): the teqp.AbstractModel instance for the basemodel, or the default if not provided
            plot_kwargs (optional): a dictionary of common arguments to be applied to liquid and vapor traces
        """

        if model is None:
            model = self.model
        if basemodel is None:
            basemodel = self.basemodel
        
        for T in Tvec:
            opt = teqp.TVLEOptions(); opt.polish=True; opt.integration_order=5; opt.calc_criticality = True
            opt.terminate_unstable = True; opt.max_steps=200; 
            # User can override tracing options if desired
            if options:
                for k, v in options.items():
                    setattr(opt, k, v)
            
            # Pure fluid endpoint
            rhovecL, rhovecV = get_rhovecLV_guess(basemodel, T, ipure)

            # Now trace and plot
            o = model.trace_VLE_isotherm_binary(T, rhovecL, rhovecV, opt)
            df = pandas.DataFrame(o)
            plt.plot(df['xL_0 / mole frac.'], df['pL / Pa']/1e6, c=cmap.to_rgba(T), **plot_kwargs)
            plt.plot(df['xV_0 / mole frac.'], df['pL / Pa']/1e6, c=cmap.to_rgba(T), dashes=[2,2], **plot_kwargs)

        ax.set_xlabel(r'$x_1$, $y_1$ / mole frac.')
        ax.set_ylabel('$p$ / MPa')
        ax.set_yscale('log')

    def plot_binary_critical_locus(self, *, ax, kind, ipure, model=None, basemodel=None, options: Dict = None, plot_kwargs: Dict = {}):
        """ 
        Args:
            ax: the axis onto which to plot
            kind: the variables to be plotted, one of {'XP','TP'}
            ipure: the index of the fluid, in {0,1}, from which the trace starts
            model (optional): the teqp.AbstractModel instance, or the default if not provided
            basemodel (optional): the teqp.AbstractModel instance for the basemodel, or the default if not provided
            options (optional): key-value pairs to overwrite sensible defaults in teqp.TCABOptions
            plot_kwargs (optional): a dictionary of common arguments to be applied to the trace
        """

        if model is None:
            model = self.model
        if basemodel is None:
            basemodel = self.basemodel

        Tcvec = basemodel.get_Tcvec()
        vcvec = basemodel.get_vcvec()
        opt = {"alternative_pure_index": ipure, "alternative_length": 2}
        [T0, rho0] = model.solve_pure_critical(Tcvec[ipure], 1.0/vcvec[ipure], opt)
        rhovec0 = np.array([0.0, 0])
        rhovec0[ipure] = rho0

        opt = teqp.TCABOptions()
        opt.polish = True
        opt.integration_order = 5
        opt.init_dt = 100
        opt.max_dt = 1000
        if options:
            for k, v in options.items():
                setattr(opt, k, v)

        cr = pandas.DataFrame(model.trace_critical_arclength_binary(T0, rhovec0, '', opt))
        # print(len(cr))
        cr['z_0 / mole frac.'] = cr['rho0 / mol/m^3']/(cr['rho0 / mol/m^3']+cr['rho1 / mol/m^3'])
        if kind == 'XP':
            ax.plot(cr['z_0 / mole frac.'], cr['p / Pa']/1e6, **plot_kwargs)
            ax.set_xlabel(r'$x_1$, $y_1$ / mole frac.')
            ax.set_ylabel('$p$ / MPa')
            ax.set_yscale('log')
        elif kind == 'TP':
            ax.plot(cr['T / K'], cr['p / Pa']/1e6, **plot_kwargs)
            ax.set_xlabel(r'$T$ / K')
            ax.set_ylabel('$p$ / MPa')
            ax.set_yscale('log')
        else:
            raise ValueError()