from multiprocessing.sharedctypes import Value
import os
import pandas
import numpy as np 

def add_homogenous_density_REFPROP(df, *, RP):
    def add_rho(row):
        T = row['T / K']; p = row['p / Pa']
        z_1 = row['z_1 / mole frac.']
        z = np.array([z_1, 1-z_1])
        r = RP.REFPROPdll('', 'TP','D',RP.MOLAR_BASE_SI, 0,0,T,p,z)
        if r.ierr > 0:
            raise ValueError(r.herr)
        return r.Output[0]
    df['rho(EOS) / mol/m^3'] = df.apply(add_rho, axis=1)
    return df

def add_Ao20_REFPROP(df, *, RP):
    def add_Ao20(row):
        T = row['T / K']; p = row['p / Pa']
        z_1 = row['z_1 / mole frac.']
        z = np.array([z_1, 1-z_1])
        r = RP.REFPROPdll('', 'TD','PHIG20',RP.MOLAR_BASE_SI, 0,0,T,0,z)
        if r.ierr > 0:
            raise ValueError(r.herr)
        return r.Output[0]
    df['Ao20'] = df.apply(add_Ao20, axis=1)
    return df

def add_coexisting_concentrations_REFPROP(df, *, RP, Q=0):
    # Store guess values for densities given the existing model in REFPROP
    def add_rhos(row):
        T = row['T / K']
        if Q == 0:
            x_1 = row['x_1 / mole frac.']
            z = np.array([x_1,1-x_1])
        elif Q == 1:
            y_1 = row['y_1 / mole frac.']
            z = np.array([y_1, 1-y_1])
        else:
            raise ValueError(Q)
        r = RP.REFPROPdll('', 'QT','DLIQ,DVAP', RP.MOLAR_BASE_SI, 0,0,Q,T,z)
        if r.ierr > 0:
            raise ValueError(r.herr)
        DLIQ, DVAP = r.Output[0:2]
        y = r.y
        return DLIQ*z[0], DLIQ*z[1], DVAP*y[0], DVAP*y[1]
    df[['rhoL_1 / mol/m^3', 'rhoL_2 / mol/m^3', 'rhoV_1 / mol/m^3', 'rhoV_2 / mol/m^3']] = df.apply(add_rhos, axis=1, result_type='expand')
    return df