import os, timeit, json, uuid, shutil, random, time
from multiprocessing import current_process

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import teqp
import CoolProp.CoolProp as CP

from temo.fit import mutant_factories, data_loaders, cost_contributions

class DataVault():

    def __init__(self, dataroot, job, *, teqp_data_root):
        """ 
        Initialize the data vault which contains the experimental 
        data in pandas DataFrames, and builds some base models needed
        for other calculation
        """

        self.names = job['names']
        molar_masses = job['molar_mass / kg/mol']
        abbrevs = ['PVT', 'SOS', 'VLE'] # shorthand for the kinds of data, just to simplify what follows
        for abbrev in abbrevs:
            load_func = getattr(data_loaders, 'load_' + abbrev)
            df = load_func(dataroot, identifier='FLD', identifiers=self.names, molar_masses=molar_masses)
            setattr(self, 'df_'+abbrev, df) # makes an attribute like: self.df_PVT

        # Build the model (here used to hold the pure fluid information)
        self.model = teqp.build_multifluid_model(self.names, teqp_data_root, teqp_data_root+'/dev/mixtures/mixture_binary_pairs.json', {'estimate':'Lorentz-Berthelot'})

    def cost_function(self, *, params, mutant=None, mutant_factory=None, mutant_kwargs={}):
        """ The cost function to be minimized 

        The scheme to be used to obtain the mutant is as follows:
        1. If mutant argument is provided, use the provided mutant directly
        2. If instead a factory function is provided, generate a mutant with the set of parameters
        """

        if mutant is None:
            mutant = mutant_factory(self.model, params=params, **mutant_kwargs)

        costs = cost_contributions
        
        costrho = 0
        errrho = costs.calc_errrho(df=self.df_PVT, model=mutant, step=10, iterate=False)
        costrho = np.abs(errrho).mean()

        costSOS = 0
        errSOS =  costs.calc_errSOS(df=self.df_SOS, model=mutant, step=10)
        costSOS = np.abs(errSOS).mean()

        costVLE = 0
        errVLE = costs.calc_errVLE(df=self.df_VLE, model=mutant, step=40)
        costVLE = np.abs(errVLE).mean()

        costB12 = 0
        # errB12 = self.calc_errB12(self.df_B12, model=mutant, step=1, z0 = 0.5)
        # costB12 = np.abs(errB12).mean()*1000*1000

        cost = 4*costrho + 1*costSOS + 20*costVLE + costB12

        return cost

def do_fit(FLDs, deptype, Ndep, root, *, Nrep=1, dv, mutant_kwargs={}):
    """
    This function actually runs the fitting of the thermodynamic models
    """

    # Seed the random number to make sure you *hopefully* get different results in 
    # each process
    seed = (os.getpid() * int(time.time())) % 314159
    np.random.seed(seed)
    random.seed(seed)

    if deptype == 'Gaussian':
        # bounds on: betaT,gammaT,betaV,gammaV,n,t,eta,beta,gamma,epsilon in a Gaussian term
        bounds = [(0.75,1.25)]*4 + [(-3,3)]*Ndep + [(0,4)]*Ndep + [(0,3)]*(2*Ndep) + [(0,2)]*(2*Ndep)
        mutant_factory = mutant_factories.get_mutant_Gaussian
    elif deptype == 'Exponential':
        # bounds on: betaT,gammaT,betaV,gammaV,n,t in an Exponential term
        bounds = [(0.9,1.1)]*4 + [(-3.0,3.0)]*Ndep + [(0.0,5)]*Ndep
        mutant_factory = mutant_factories.get_mutant_exponential
    elif deptype == 'Gaussian+Exponential':
        Npoly = mutant_kwargs['Npoly']
        Ngaussian = Ndep-Npoly
        bounds = ([(0.75,1.25)]*4 # betaT,gammaT,betaV,gammaV
                + [(-3.0,3.0)]*Ndep # n
                + [(0.0,5)]*Ndep  # t
                + [(0,3)]*(2*Ngaussian) # eta, beta
                + [(0,3)]*(2*Ngaussian) # gamma, epsilon
                )
        mutant_factory = mutant_factories.get_mutant_exponentialGaussian
    else:
        raise ValueError('bad deptype')

    print(f'{len(bounds)} adjustable parameters to be optimized for {deptype}')
    
    for rep in range(Nrep):
        uid = str(uuid.uuid1())
        tic = timeit.default_timer()

        def objective(x):
            tic = timeit.default_timer()
            v = dv.cost_function(params=x, mutant_factory=mutant_factory, mutant_kwargs=mutant_kwargs)
            toc = timeit.default_timer()
            return v

        class Callbacker:
            def __init__(self, basemodel):
                self.step_number = 0
                self.basemodel = basemodel

            def __call__(self, *args, **kwargs):
                try:
                    x = args[0].tolist()
                    convergence = kwargs['convergence']
                    mutant = mutant_factory(model=self.basemodel, params=x, **mutant_kwargs)
                    mutant_json = json.loads(mutant.get_meta())
                    AAD_PVT = np.mean(np.abs(cost_contributions.calc_errrho(model=mutant, df=dv.df_PVT, iterate=True)))
                    AAD_SOS = np.mean(np.abs(cost_contributions.calc_errSOS(model=mutant, df=dv.df_SOS, max_iter=10)))
                    print(f"AAD(PVT): {AAD_PVT:0.2f} %; AAD(SOS): {AAD_SOS:0.2f} %")
                    with open(f'{root}/{uid}_step{self.step_number}.json', 'w') as fp:
                        fp.write(json.dumps({
                            'x': x,
                            'cost': objective(x),
                            'convergence': convergence,
                            'model': mutant_json,
                            'AAD(PVT, iterate) / %': AAD_PVT,
                            'AAD(SOS) / %': AAD_SOS,
                        }))
                except BaseException as be:
                    print(be)

                print(args, kwargs)
                self.step_number += 1

        callback = Callbacker(dv.model)

        res = scipy.optimize.differential_evolution(
            objective, bounds, disp=True, maxiter=10000, callback=callback, polish=False
        )
        
        toc = timeit.default_timer()
        with open(f'{root}/{uid}.json','w') as fp:
            model = json.loads(mutant_factory(dv.model, params=res.x, **mutant_kwargs).get_meta())
            cost = res.fun
            print(res)
            fp.write(json.dumps({'model':model,'cost':cost,'pair': FLDs,
                                 'Ndep':Ndep,'uid':uid, 'bounds':bounds, 
                                 'mutant_kwargs':mutant_kwargs, 
                                 'x': res.x.tolist(), 'elapsed / s':toc-tic}))

def initialize_CoolProp():
    CP.set_config_bool(CP.OVERWRITE_FLUIDS, False)
    # CP.add_fluids_as_JSON('HEOS', json.dumps([json.load(open('NEWR1234YF.json'))]))
    # CP.apply_simple_mixing_rule('754-12-1xxx','811-97-2','linear')
    # CP.apply_simple_mixing_rule('29118-24-9','754-12-1xxx','linear')
    # CP.apply_simple_mixing_rule('29118-24-9','811-97-2','linear')
    # CP.apply_simple_mixing_rule('29118-24-9','75-10-5','linear')
    # CP.apply_simple_mixing_rule('75-10-5','754-12-1xxx','linear')
    # CP.apply_simple_mixing_rule('29118-24-9','431-89-0','linear')
    # CP.apply_simple_mixing_rule('75-37-6','754-12-1xxx','linear')

if current_process().name == 'MainProcess' or __name__ != '__main__':
    initialize_CoolProp()
    
if __name__ == '__main__':
    pairs = [('AMMONIA', 'WATER'),]
    dataroot = 'NH3H2O'

    # Make sure all the runs you want to do can load the data properly
    for FLDs in pairs:
        job = {
            'names': FLDs, 
            'molar_mass / kg/mol': [CP.PropsSI('molemass', f) for f in FLDs]
        }
        dv = DataVault(dataroot=dataroot, job=job, teqp_data_root=teqp.get_datapath())

    root = 'run'+str(uuid.uuid1())
    os.makedirs(root)
    print(f'Starting optimization in {root}')
    shutil.copy2(__file__, root)
    shutil.copytree(dataroot, root + '/fitdataroot')

    # Serial
    Ndep = 2
    Npoly = Ndep
    kwargs = dict(Npoly=Npoly, Ngaussian=Ndep-Npoly)
    do_fit(pairs[0], 'Gaussian+Exponential', Ndep, root, dv=dv, mutant_kwargs=kwargs)

    # # Call the spawner to fit in parallel
    # deptype = 'Gaussian+Exponential'
    # from spawn import Spawner
    # args = []
    # for pair in pairs:
    #     for Ndep in range(6):
    #         for Npoly in [Ndep]:#range(Ndep+1):
    #             for N in range(2): # number of repeats
    #                 kwargs = dict(Npoly=Npoly, Ngaussian=Ndep-Npoly)
    #                 args.append(dict(target=do_fit, args = [pair, deptype, Ndep, root], kwargs=dict(mutant_kwargs=kwargs)))
    # spawner = Spawner(args, Nproc_max = 16)
    # print(spawner.run())

    print(f'Finished optimization in {root}')
    shutil.make_archive(root, format='xztar', root_dir=root)