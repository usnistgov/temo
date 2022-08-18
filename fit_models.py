import os, timeit, json, uuid, shutil, random, time
from itertools import repeat, cycle, islice
from multiprocessing import current_process

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import teqp
print(f'teqp version: {teqp.__version__}')

from temo.fit import mutant_factories, data_loaders, cost_contributions

class DataVault():

    step_PVT = 10
    step_PVT_P = 10
    step_SOS = 10
    step_VLE = 40

    def __init__(self, dataroot, job, *, teqp_data_root):
        """
        Initialize the data vault which contains the experimental
        data in pandas DataFrames, and builds some base models needed
        for other calculation
        """

        self.names = job['names']
        molar_masses = job['molar_mass / kg/mol']
        abbrevs = ['PVT', 'PVT_P', 'SOS', 'VLE', 'CRIT'] # shorthand for the kinds of data, just to simplify what follows
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

        tic = timeit.default_timer()
        if mutant is None:
            mutant = mutant_factory(self.model, params=params, **mutant_kwargs)
        time_mutant = timeit.default_timer()-tic

        costs = cost_contributions # convenience alias
        get_weight = lambda df: 1.0 if 'weight' not in df else df['weight']

        costrho = 0
        tic = timeit.default_timer()
        errrho = costs.calc_errrho(df=self.df_PVT, model=mutant, step=self.step_PVT, iterate=False)
        errrho *= get_weight(self.df_PVT)
        costrho = np.abs(errrho).mean()
        time_PVT = timeit.default_timer()-tic

        costrho_P = 0
        tic = timeit.default_timer()
        errrho_P = costs.calc_errrho_devp(df=self.df_PVT_P, model=mutant, step=self.step_PVT_P)
        errrho_P *= get_weight(self.df_PVT_P)
        costrho_P = np.abs(errrho_P).mean()
        time_PVT_P = timeit.default_timer()-tic

        costSOS = 0
        tic = timeit.default_timer()
        errSOS =  costs.calc_errSOS(df=self.df_SOS, model=mutant, step=self.step_SOS)
        errSOS *= get_weight(self.df_SOS)
        costSOS = np.abs(errSOS).mean()
        time_SOS = timeit.default_timer()-tic

        costVLE = 0
        tic = timeit.default_timer()
        errVLE = costs.calc_errVLE(df=self.df_VLE, model=mutant, step=self.step_VLE)
        errVLE *= get_weight(self.df_VLE)
        costVLE = np.abs(errVLE).mean()
        time_VLE = timeit.default_timer()-tic

        costcrit = 0
        # ipure = 0
        # T0 = self.model.get_Tcvec()[ipure]
        # rhovec0 = np.array([1.0/self.model.get_vcvec()[ipure]]*2)
        # rhovec0[1-ipure] = 0 # Other component has no molar concentration (pure of fluid at index ipure)
        # errcrit = costs.calc_errtracecrit(df=self.df_CRIT, model=mutant, T0=T0, rhovec0=rhovec0, errscheme='TdevP')
        # costcrit = np.abs(errcrit).mean()

        costcritpts = 0
        # errcrit = costs.calc_errcritPT(df=self.df_CRIT, model=mutant)
        # errcrit *= get_weight(self.df_CRIT)
        # costcritpts = np.abs(errcrit).mean()*10

        costB12 = 0
        # errB12 = self.calc_errB12(self.df_B12, model=mutant, step=1, z0 = 0.5)
        # costB12 = np.abs(errB12).mean()*1000*1000

        # print(f'[timing] mutant: {time_mutant} PVT: {time_PVT} PVT_P: {time_PVT_P} SOS: {time_SOS} VLE: {time_VLE}')

        cost = 4*costrho + costrho_P + 1*costSOS + 1*costVLE + costB12 + costcritpts + costcrit
        # cost = costcrit

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
                + [(-3,8)]*Ndep  # t
                + [(0,3)]*(2*Ngaussian) # eta, beta
                + [(0,3)]*(2*Ngaussian) # gamma, epsilon
                )
        mutant_factory = mutant_factories.get_mutant_exponentialGaussian
    else:
        raise ValueError('bad deptype')

    if isinstance(dv, dict):
        dv = DataVault(**dv)

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
                    AAD_PVT = np.mean(np.abs(cost_contributions.calc_errrho(model=mutant, df=dv.df_PVT, step=dv.step_PVT, iterate=True)))
                    AAD_PVT_P = np.mean(np.abs(cost_contributions.calc_errrho_devp(model=mutant, df=dv.df_PVT_P, step=dv.step_PVT_P)))
                    AAD_SOS = np.mean(np.abs(cost_contributions.calc_errSOS(model=mutant, df=dv.df_SOS, step=dv.step_SOS, max_iter=10)))
                    AAD_VLE = np.mean(np.abs(cost_contributions.calc_errVLE(model=mutant, df=dv.df_VLE, step=dv.step_VLE)))
                    print(f"AAD(PVT): {AAD_PVT:0.2f} %; AAD(PVT_P): {AAD_PVT_P:0.2f} %; AAD(SOS): {AAD_SOS:0.2f} %; AAD(VLE): {AAD_VLE:0.2f} %")
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

if __name__ == '__main__':
    import CoolProp.CoolProp as CP
    pairs = [('AMMONIA', 'WATER'),]
    dataroot = 'NH3H2O'
    def get_dv_args(FLDs):
        job = {
            'names': FLDs,
            'molar_mass / kg/mol': [CP.PropsSI('molemass', f) for f in FLDs]
        }
        return dict(dataroot=dataroot, job=job, teqp_data_root=teqp.get_datapath())

    # Make sure all the runs you want to do can load the data properly
    for FLDs in pairs:
        dv = DataVault(**get_dv_args(FLDs))

    root = 'run'+str(uuid.uuid1())
    os.makedirs(root)
    print(f'Starting optimization in {root}')
    shutil.copy2(__file__, root)
    shutil.copytree(dataroot, root + '/fitdataroot')
    
    # Serial
    Ndep = 6
    Npoly = Ndep
    dvals = [1,2,3,4,5,6]
    d = list(roundrobin(*repeat(dvals, 2)))[0:Ndep] # [1,1,2,2,3,3...]
    kwargs = dict(Npoly=Npoly, Ngaussian=Ndep-Npoly, d=d, l=[1]*Ndep)
    do_fit(pairs[0], 'Gaussian+Exponential', Ndep, root, dv=dv, mutant_kwargs=kwargs)

    # # Call the spawner to fit in parallel
    # deptype = 'Gaussian+Exponential'
    # from spawn import Spawner
    # args = []
    # for pair in pairs:
    #     for Ndep in range(6):
    #         for Npoly in [Ndep]:#range(Ndep+1):
    #             for N in range(2): # number of repeats
    #                 Ndep = Npoly
    #                 # Npoly = Ndep
    #                 dvals = [1,2,3,4,5,6]
    #                 d = list(roundrobin(*repeat(dvals, 2)))[0:Npoly] # [1,1,2,2,3,3...]
    #                 kwargs = dict(Npoly=Npoly, Ngaussian=Ndep-Npoly, d=d, l=[1]*Npoly)
    #                 datavault_kwargs = get_dv_args(pair)
    #                 args.append(dict(target=do_fit, args = [pair, deptype, Ndep, root], kwargs=dict(dv=datavault_kwargs, mutant_kwargs=kwargs)))
    # spawner = Spawner(args, Nproc_max=12)
    # print(spawner.run())

    print(f'Finished optimization in {root}')
    shutil.make_archive(root, format='xztar', root_dir=root)