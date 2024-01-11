import itertools
import numpy as np 
import teqp

def get_mutant_exponential(model, params, d=None, l=None):
    """ 
    Build a teqp-based exponential mutant from the model parameters 

    Args:
        model : The base model that is used to form the mutant
        params (iterable): iterable that contains the parameters in a flat iterable object
        d (list, optional): set of exponents on delta, optional
        l (list, optional): set of exponents on delta in exponential, optional

    Term is of the form:

    .. math::

        \\alpha^{\\rm r} = \\sum_{i} n_i\\tau^{t_i}\\delta^{d_i}\\exp(-\\delta^{l_i})

    where the params populate the n and t. Values of d and l are user-specified, or follow
    the automatic logic in this function
    """
    depparams = np.array(params[4::])
    Ndep = len(depparams)//2
    if d is None:
        Diterator = itertools.cycle([1,1,2,3,4])
        d = [next(Diterator) for _ in range(Ndep)]
    if l is None:
        Literator = itertools.cycle([2,2,3,1,2,3])
        l = [next(Literator) for _ in range(Ndep)]
    s = {
        "0":{
            "1": {
                "BIP":{
                    "type": "GERG",
                    "betaT": params[0],
                    "gammaT": params[1],
                    "betaV": params[2],
                    "gammaV": params[3],
                    "Fij": 1.0
                },
                "departure":{
                    "type" : "Exponential",
                    "n" : depparams[0:Ndep].tolist(),
                    "t" : depparams[Ndep::].tolist(),
                    "d" : d,
                    "l" : l
                }
            }
        }
    }
    return teqp.build_multifluid_mutant(model, s)

# def get_mutant_polynomial(model, params, d=None):
#     """  Build a teqp-based mutant from the model parameters with polynomial terms """
#     Ndep = len(depparams)//2
#     l = [0.0]*Ndep
#     return get_mutant_exponential(model, params, d=d, l=l)

def get_mutant_Gaussian(model, params, d=None):
    """ 
    Build a teqp-based Gaussian-bell-shaped mutant from the model parameters 

    Args:
        model : The base model that is used to form the mutant
        params (iterable): iterable that contains the parameters in a flat iterable object
        d (list, optional): set of exponents on delta, optional

    Term is of the form:

    .. math::

        \\alpha^{\\rm r} = \\sum_{i} n_i\\tau^{t_i}\\delta^{d_i}\\exp(-\\eta_i(\delta-\\varepsilon_i)^2-\\beta_i(\\tau-\\gamma_i)^2)

    where the params populate the variables. Values of d are user-specified as a list, or follow
    the automatic logic in this function
    """
    depparams = np.array(params[4::])
    Nvar = 6 # How many variables are being fit
    Ndep = len(depparams)//Nvar # How many departure terms
    assert(Ndep*Nvar == len(depparams))
    n,t,eta,beta,gamma,epsilon = [depparams[k*Ndep:(k+1)*Ndep].tolist() for k in range(Nvar)]
    if d is None:
        Diterator = itertools.cycle([1,2,3,4])
        d = [next(Diterator) for _ in range(Ndep)]
    l = [0]*Ndep

    # Build a dictionary in the format that teqp needs. The conversion to
    # string passed to the C++ interface and upacking into the JSON
    # structure is handled implicitly in the pybind11 interface.
    # 
    # The indices have to be strings (sigh...) as mandated
    # by the JSON standard: https://www.json.org/json-en.html
    s = {
        "0":{
            "1": {
                "BIP":{
                    "type": "GERG",
                    "betaT": params[0],
                    "gammaT": params[1],
                    "betaV": params[2],
                    "gammaV": params[3],
                    "Fij": 1.0
                },
                "departure":{
                    "type" : "Gaussian+Exponential",
                    "Npower": 0,
                    "n" : n,
                    "t" : t,
                    "d" : d,
                    "l" : l,
                    "eta": eta,
                    "beta": beta,
                    "gamma": gamma,
                    "epsilon": epsilon
                }
            }
        }
    }
    return teqp.build_multifluid_mutant(model, s)

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def get_mutant_exponentialGaussian(model, params, *, Npoly, Ngaussian, d=None,l=None, Nbg=4, bgindices=None):
    """
    Build a teqp-based Gaussian+exponential mutant from the model parameters
    
    Args:
        model : The base model that is used to form the mutant
        params (iterable): iterable that contains the parameters in a flat iterable object
        Npoly (int): number of polynomial-like terms
        Ngaussian (int): number of Gaussian terms
        d (list, optional): set of exponents on delta, optional
        l (list, optional): set of exponents on delta in exponential, optional
        Nbg (int, optinal): the number of beta and gamma parameters being fit, usually 4 to indicate that all four are being fit
        bgindices (sequence, optional): the indices {0: betaT, 1: gammaT, 2: betaV, 3: gammaV} to be fit

    Term is of the form:

    .. math::

        \\alpha^{\\rm r} = \\alpha^{\\rm r}_{\\rm G} + \\alpha^{\\rm r}_{\\rm P}

    with 

    .. math::

        \\alpha^{\\rm r}_{\\rm G} = \\sum_{i} n_i\\tau^{t_i}\\delta^{d_i}\\exp(-\\eta_i(\delta-\\varepsilon_i)^2-\\beta_i(\\tau-\\gamma_i)^2)

    .. math::

        \\alpha^{\\rm r}_{\\rm P} = \\sum_{i} n_i\\tau^{t_i}\\delta^{d_i}\\exp(-\\delta^{l_i})

    where the params populate the n and t. Values of d and l are user-specified, or follow
    the automatic logic in this function
    """
    depparams = np.array(params[Nbg::])
    Ndep = Npoly + Ngaussian
    assert(Ndep*2 + Ngaussian*4 == len(depparams))

    if Ndep > 0:
        n, t = [g for g in chunked_iterable(depparams[0:(Ndep*2)], Ndep)]
    else:
        n, t = [],[]
    if Ngaussian > 0:
        eta,beta,gamma,epsilon = [[0.0]*Npoly + list(k) for k in chunked_iterable(depparams[(Ndep*2)::], Ngaussian)]
    else:
        eta,beta,gamma,epsilon = [0.0]*Npoly,[0.0]*Npoly,[0.0]*Npoly,[0.0]*Npoly
    if d is None:
        Diterator = itertools.cycle([1,2,3,4,5,6,4,3])
        d = [next(Diterator) for _ in range(Ndep)]
    if l is None:
        Literator = itertools.cycle([1,1,1,1,2,2,2,3,3,3])
        l = [next(Literator) for _ in range(Ndep)]

    if Nbg == 4:
        betaT, gammaT, betaV, gammaV = params[0:Nbg]
    else:
        assert(len(bgindices) == Nbg)
        bgparams = np.array([1.0]*4)
        bgparams[bgindices] = params[0:Nbg]
        betaT, gammaT, betaV, gammaV = bgparams

    # Build a dictionary in the format that teqp needs. The conversion to
    # string passed to the C++ interface and upacking into the JSON
    # structure is handled implicitly in the pybind11 interface.
    # 
    # The indices have to be strings (sigh...) as mandated
    # by the JSON standard: https://www.json.org/json-en.html
    s = {
        "0":{
            "1": {
                "BIP":{
                    "type": "GERG",
                    "betaT": betaT,
                    "gammaT": gammaT,
                    "betaV": betaV,
                    "gammaV": gammaV,
                    "Fij": 1.0
                },
                "departure":{
                    "type" : "Gaussian+Exponential",
                    "Npower": Npoly,
                    "n" : n,
                    "t" : t,
                    "d" : d,
                    "l" : l,
                    "eta": eta,
                    "beta": beta,
                    "gamma": gamma,
                    "epsilon": epsilon
                }
            }
        }
    }
    return teqp.build_multifluid_mutant(model, s)

def get_mutant_doubleexponential(model, params, *, d=None, ld=None):
    """
    Build a teqp-based double-exponential mutant from the model parameters
    
    Args:
        model: the base model that is used to form the mutant
        params: iterable that contains the parameters in a flat iterable object
        d: set of exponents on delta, optional
        ld (list, optional): set of exponents on delta in exponential, optional
    """
    depparams = np.array(params[4::])
    Ndep = len(depparams)//5
    if Ndep > 0:
        n, t, gd, lt, gt = [g for g in chunked_iterable(depparams[0:(Ndep*5)], Ndep)]
    if d is None:
        Diterator = itertools.cycle([1,2,3,4,5,2,3])
        d = [next(Diterator) for _ in range(Ndep)]
    if ld is None:
        Literator = itertools.cycle([2,2,2,2,1,1,3,3,3])
        ld = [next(Literator) for _ in range(Ndep)]

    # print(n,t,d,gd,ld,gt,lt)

    # Build a dictionary in the format that teqp needs. The conversion to
    # string passed to the C++ interface and upacking into the JSON
    # structure is handled implicitly in the pybind11 interface.
    # 
    # The indices have to be strings (sigh...) as mandated
    # by the JSON standard: https://www.json.org/json-en.html
    s = {
        "0":{
            "1": {
                "BIP":{
                    "type": "GERG",
                    "betaT": params[0],
                    "gammaT": params[1],
                    "betaV": params[2],
                    "gammaV": params[3],
                    "Fij": 1.0
                },
                "departure":{
                    "type" : "DoubleExponential",
                    "n" : n,
                    "t" : t,
                    "d" : d,
                    "ld" : ld,
                    "gd" : gd,
                    "lt" : lt,
                    "gt" : gt,
                }
            }
        }
    }
    # print(json.dumps(s,indent=1))
    return teqp.build_multifluid_mutant(model, s)

def get_mutant_Gaussian_invariant(model, params, d=None):
    """ Build a teqp-based Gaussian-bell-shaped mutant from the model parameters and with an invariant reducing function
    
    Args:
        model: the base model that is used to form the mutant
        params: iterable that contains the parameters in a flat iterable object
        d: set of exponents on delta, optional
        
    """
    depparams = np.array(params[4::])
    Nvar = 6 # How many variables are being fit
    Ndep = len(depparams)//Nvar # How many departure terms
    assert(Ndep*Nvar == len(depparams))
    n,t,eta,beta,gamma,epsilon = [depparams[k*Ndep:(k+1)*Ndep].tolist() for k in range(Nvar)]
    if d is None:
        Diterator = itertools.cycle([1,2,3,4])
        d = [next(Diterator) for _ in range(Ndep)]
    l = [0]*Ndep

    # Build a dictionary in the format that teqp needs. The conversion to
    # string passed to the C++ interface and upacking into the JSON
    # structure is handled implicitly in the pybind11 interface.
    # 
    # The indices have to be strings (sigh...) as mandated
    # by the JSON standard: https://www.json.org/json-en.html
    s = {
        "0":{
            "1": {
                "BIP":{
                    "type": "invariant",
                    "phiT": params[0],
                    "lambdaT": params[1],
                    "phiV": params[2],
                    "lambdaV": params[3],
                    "Fij": 1.0
                },
                "departure":{
                    "type" : "Gaussian+Exponential",
                    "Npower": 0,
                    "n" : n,
                    "t" : t,
                    "d" : d,
                    "l" : l,
                    "eta": eta,
                    "beta": beta,
                    "gamma": gamma,
                    "epsilon": epsilon
                }
            }
        }
    }
    return teqp.build_multifluid_mutant_invariant(model, s)

def get_mutant_Chebyshev2D(model, params, *, taumin: float, taumax: float, deltamin: float, deltamax: float, Ntau: int, Ndelta: int):
    """ Build a teqp-based Gaussian+exponential mutant from the model parameters 
    params: iterable that contains the parameters in a flat iterable object
    """
    depparams = np.array(params[4::])

    # Build a dictionary in the format that teqp needs. The conversion to
    # string passed to the C++ interface and upacking into the JSON
    # structure is handled implicitly in the pybind11 interface.
    # 
    # The indices have to be strings (sigh...) as mandated
    # by the JSON standard: https://www.json.org/json-en.html
    s = {
        "0":{
            "1": {
                "BIP":{
                    "type": "GERG",
                    "betaT": params[0],
                    "gammaT": params[1],
                    "betaV": params[2],
                    "gammaV": params[3],
                    "Fij": 1.0
                },
                "departure":{
                    "type" : "Chebyshev2D",
                    "a": depparams.tolist(),
                    "taumin": taumin,
                    "taumax": taumax,
                    "deltamin": deltamin,
                    "deltamax": deltamax,
                    "Ntau": Ntau,
                    "Ndelta": Ndelta
                }
            }
        }
    }
    return teqp.build_multifluid_mutant(model, s)
