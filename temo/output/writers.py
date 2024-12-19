import os

from pathlib import Path
from typing import Callable

import pandas 
        
def teqp_info_from_summary(summary_file:Path, CASprovider:Callable, round_digits=None):
    """Take summary file and generate files in teqp/CoolProp format.
    The summary file should have a column called "selected" and for 
    each binary pair, one and only one row should be non-empty, 
    indicating which model is selected for that binary pair

    Args:
        summary_file (Path): The file to be processed to write models
        CASprovider (Calllable): A callable that takes a fluid identifier and returns the CAS# as a string
    """
    df = pandas.read_csv(summary_file)
    
    index = 0
    jbin, jdep = [], []
    for pair, gp in df.groupby('pair'):
        if 'selected' not in gp:
            raise ValueError("The selected column must be in the summary file")
        rows = gp[~pandas.isnull(gp.selected)]
        # Only one row can indicated as selected
        if not len(rows) == 1:
            raise ValueError("Only one row for this binary pair may be selected, as indicated by any contents in the \"selected\" column")
        row = rows.iloc[0]
        function_name = f'B{index:02d}'
        pair = eval(pair)
        
        m = eval(row["model"])
        BIP = m["0"]["1"]["BIP"]
        BIP.update({"F":BIP["Fij"], 'function': function_name,
            "Name1":pair[0], "Name2": pair[1],
            "CAS1": CASprovider(pair[0]), "CAS2": CASprovider(pair[1]), 'BibTeX': 'N/A'
            })
        del BIP["Fij"]
        if round_digits is not None:
            for k in ['betaT','gammaT','betaV','gammaV']:
                BIP[k] = round(BIP[k], round_digits)
        jbin.append(BIP)
        
        dep = m["0"]["1"]['departure']
        
        if round_digits is not None:
            for k in ['n','t']:
                dep[k] = [round(_, 12) for _ in dep[k]]
        dep["Name"] = function_name
        dep['BibTeX'] = 'N/A'
        dep["aliases"] = []
        jdep.append(dep)
        
    return jbin, jdep

HMX_header = """HMX               !Mnemonic for mixture model, must match hfmix on call to SETUP.
5                 !Version number

! Changelog:
! ---------

#BNC              !Binary mixing coefficients
BNC
? Binary mixing coefficients for the various mixing rules used with the HMX model:
?
? KWi:  (i = 1,2,3,...,A,B,...)  --->  Kunz-Wagner mixing rules
?   model     BetaT     GammaT    BetaV     GammaV    Fij      not used
?
!"""

HMX_footer = """
@END
c        1         2         3         4         5         6         7         8
c2345678901234567890123456789012345678901234567890123456789012345678901234567890
"""

BIN_template = """
?{name1}/{name2}                        [{name1}/{name2}]
?FITTED {annotation}
  {hash1}/{hash2}
    {model}     {betaT:14.11f} {gammaT:14.11f} {betaV:14.11f} {gammaV:14.11f} {Fij:14.11f}  0.             0. 0. 0. 0. 0. 0.
    TC5    0.0      0.0     0.0       0.             0.             0.             0. 0. 0. 0. 0. 0.
    VC5    0.0      0.0     0.0       0.             0.             0.             0. 0. 0. 0. 0. 0.
!"""

MODEL_template = """#MXM              !Mixture model specification
{model} {annotation}
?
!```````````````````````````````````````````````````````````````````````````````
 BetaT    GammaT   BetaV    GammaV    Fij    not used      !Descriptors for binary-specific parameters
  1.0      1.0      1.0      1.0      0.0      0.0         !Default values (i.e. ideal-solution)
  {Nexp} {Ntermsexp}      0        {NKW} {NtermsKW}      {NGaussian} {NtermsGaussian}      0 0      0 0         !# terms and # coefs/term for normal terms, Kunz-Wagner terms, and Gaussian terms.  3rd column is not used.
  """

def HMX_from_teqp(jbin:list, jdep:list, hashprovider:Callable):
    '''
    Read in CoolProp-format departure term JSON structure and build a REFPROP-format HMX.BNC
    
    Args:
      jbin (list[dict]): The binary interaction parameters
      jdep (list[dict]): The departure functions
    '''

    """ Determine acceptable names for functions """
    def get_function_names():
        function_names = set([el['Name'] for el in jdep])
        return function_names
    function_names = get_function_names()
    if len(function_names) > 100:
        raise ValueError("Max of 100 functions allowed")
    RP_function_names = {f:f'B{i:2d}'.replace(' ','0') for i,f in enumerate(function_names)}

    out = HMX_header

    # The top part with the definitions of what model to use for each binary pair and the 
    # model name
    for el in jbin:
        if 'function' not in el:
            RP_function_number = 'BA0'
        else:
            if el['function'] == 'XR0':
                RP_function_number = 'XR0'
            else:
                RP_function_number = RP_function_names[el['function']]
        if 'xi' in el and 'zeta' in el:
            raise NotImplementedError("TODO: Need to convert values to gammaT and gammaV.........................")
            continue
        else:
            template_values = {
                'betaT': el['betaT'], 
                'gammaT': el['gammaT'], 
                'betaV': el['betaV'], 
                'gammaV': el['gammaV'], 
                'Fij': el['F'],
                'name1': el['Name1'],
                'name2': el['Name2'],
                'annotation': 'time/date',
                'hash1': hashprovider(el['Name1']),
                'hash2': hashprovider(el['Name2']),
                'model': RP_function_number
            }
            entry = BIN_template.format(**template_values)
            out += entry

    out += '\n\n'

    for el in jdep:
        # Build the departure term
        model = RP_function_names[el["Name"]]
        annotation = f'{el["Name"]} '

        Nexp = Ntermsexp = NKW = NtermsKW = NGaussian = NtermsGaussian = 0
        if el['type'] == 'Exponential':
            Nexp = len(el['n'])
            Ntermsexp = 4 if 'l' in el and isinstance(el['l'], list) else 3
            
            # print(Nexp, Ntermsexp)
            if Ntermsexp == 4:
                n, t, d, l = el['n'], el['t'], el['d'], el['l']
                rows = []
                for i in range(len(t)):
                    rows.append(f'{n[i]} {t[i]} {d[i]:0.10f} {l[i]:0.10f}')
                    if i == 0:
                        rows[-1] += f' ! n(i),t(i),d(i),l(i) in term n_i*tau^t_i*delta^d_i*exp(-delta^l_i)'
            elif Ntermsexp == 3:
                n, t, d = el['n'], el['t'], el['d']
                first_row = f'{n[0]} {t[0]} {d[0]:0.1f} ! n(i),t(i),d(i) in term n_i*tau^t_i*delta^d_i'
                rows = []
                for i in range(len(t)):
                    rows.append(f'{n[i]} {t[i]} {d[i]:0.10f}')
                    if i == 0:
                        rows[-1] += f' ! n(i),t(i),d(i) in term n_i*tau^t_i*delta^d_i'
            else:
                raise ValueError()

        elif el['type'] == 'GERG-2008':
            # print(el)
            Nexp = el['Npower']
            Ntermsexp = 3
            assert('l' not in el)
            NKW = len(el['n'])-Nexp
            NtermsKW = 7
            
            n, t, d, eta, epsilon, beta, gamma = el['n'], el['t'], el['d'], el['eta'], el['epsilon'], el['beta'], el['gamma']
            l = None
            rows = []
            if Nexp > 0:
                for i in range(Nexp):
                    rows.append(f'{n[i]} {t[i]} {d[i]:0.10f} ')
                    if i == 0:
                        rows[-1] += '! n(i),t(i),d(i) in term n_i*tau^t_i*delta^d_i'
            if NKW > 0:
                for i in range(Nexp,len(t)):
                    rows.append(f'{n[i]} {t[i]} {d[i]:0.10f} {eta[i]} {epsilon[i]} {beta[i]} {gamma[i]} ')
                    if i == Nexp:
                        rows[-1] += '! n(i),t(i),d(i),eta(i),epsilon(i),beta(i),gamma(i) in term n_i*tau^t_i*delta^d_i*exp(-eta*(delta-epsilon)^2-beta*(delta-gamma))'

        elif el['type'] == 'Gaussian+Exponential':
            # print(el)
            Nexp = el['Npower']
            NGaussian = len(el['n'])-Nexp
            
            n, t, d, l, eta, epsilon, beta, gamma = el['n'], el['t'], el['d'], el['l'], el['eta'], el['epsilon'], el['beta'], el['gamma']
            rows = []
            if Nexp > 0:
                Ntermsexp = 4
                for i in range(Nexp):
                    rows.append(f'{n[i]} {t[i]} {d[i]:0.16f} {l[i]:0.16f} ')
                    if i == 0:
                        rows[-1] += '! n(i),t(i),d(i),l(i) in term n_i*tau^t_i*delta^d_i*exp(-sgn(l_i)*delta^l_i)'
            if NGaussian > 0:
                NtermsGaussian = 12
                for i in range(Nexp,len(t)):
                    negetai = -eta[i]
                    negbetai = -beta[i]
                    rows.append(f'{n[i]} {t[i]} {d[i]:0.16f} 2.0 2.0 {negetai} {negbetai} {gamma[i]} {epsilon[i]} 0.0 0.0 0.0 0.0')
                    if i == Nexp:
                        rows[-1] += '! n(i),t(i),d(i),_,_,eta(i),beta(i),gamma(i),epsilon(i),_,_,_,_ in term n_i*tau^t_i*delta^d_i*exp(eta*(delta-epsilon)^2+beta*(tau-gamma)^2)'
        else:
            raise KeyError(el['type'])
            
        out += MODEL_template.format(**locals())
        out += '\n'.join(rows) + '\n\n'

    out += '\n\n' + HMX_footer

    return out
        
if __name__ == '__main__':
    
    import ctREFPROP.ctREFPROP as ct 
    if 'RPPREFIX' in os.environ:
        root = os.getenv('RPPREFIX')
    else:
        root = '/Users/ihb/REFPROP10' # adjust as needed if REFPROP is not installed in the location matching the RPPREFIX environment variable
        
    RP = ct.REFPROPFunctionLibrary(root)
    RP.SETPATHdll(root)
    
    def get_CAS(p):
        r = RP.REFPROPdll(os.path.abspath(p+'.FLD'), '','CAS#',0,0,0,0,0,[])
        if r.ierr == 0:
            return r.hUnits
        else:
            raise ValueError(p)
        
    def get_hash(p):
        r = RP.REFPROPdll(os.path.abspath(p+'.FLD'), '','HASH',0,0,0,0,0,[])
        if r.ierr == 0:
            return r.hUnits
        else:
            raise ValueError(p)
    
    jbin, jdep = teqp_info_from_summary(open('.latest').read()+'_summary.csv', CASprovider=get_CAS)
    HMX = HMX_from_teqp(jbin=jbin, jdep=jdep, hashprovider=get_hash)
    with open('HMX.new.BNC','w') as fp:
        fp.write(HMX)
        
    RP.FLAGSdll('HMX', 1)
    print(RP.SETUPdll(2, 'R1336MZZZ*'+os.path.abspath('R1130E.FLD'), os.path.abspath('HMX.new.BNC'), 'DEF'))
    print(RP.GETKTVdll(1,2))