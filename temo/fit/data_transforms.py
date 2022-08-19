from multiprocessing.sharedctypes import Value
import os
import pandas
import numpy as np 

def add_homogeneous_density_REFPROP(df, *, RP):
    def add_rho(row):
        T = row['T / K']; p = row['p / Pa']
        z_1 = row['z_1 / mole frac.']
        z = np.array([z_1, 1-z_1])
        r = RP.REFPROPdll('', 'TP','D',RP.MOLAR_BASE_SI, 0,0,T,p,z)
        if r.ierr > 0:
            print(r.herr)
            return np.nan
            raise ValueError(r.herr)
        return r.Output[0]
    df['rho(EOS) / mol/m^3'] = df.apply(add_rho, axis=1)
    return df

def add_Ao20_REFPROP(df, *, RP):
    def add_Ao20(row):
        T = row['T / K']
        z_1 = row['z_1 / mole frac.']
        z = np.array([z_1, 1-z_1])
        r = RP.REFPROPdll('', 'TD','PHIG20',RP.MOLAR_BASE_SI, 0,0,T,0,z)
        if r.ierr > 0:
            raise ValueError(r.herr)
        return r.Output[0]
    df['Ao20'] = df.apply(add_Ao20, axis=1)
    return df

def add_pure_crit_REFPROP(df, *, RP, ifluid):
    def add(row):
        T = row['T / K']
        z = np.array([0.0, 0.0])
        z[ifluid] = 1
        r = RP.REFPROPdll('', 'TQ','DLIQ;DVAP',RP.MOLAR_BASE_SI, 0,0,T,0,z)
        rhovecL = np.array([0.0, 0.0])
        rhovecL[ifluid] = r.Output[0]
        rhovecV = np.array([0.0, 0.0])
        rhovecV[ifluid] = r.Output[1]
        
        if r.ierr > 0:
            raise ValueError(r.herr)
        return rhovecL[0], rhovecL[1], rhovecV[0], rhovecV[1]
    df[['rhoL_pure_1 / mol/m^3','rhoL_pure_2 / mol/m^3','rhoV_pure_1 / mol/m^3','rhoV_pure_2 / mol/m^3']] = df.apply(add, axis=1, result_type='expand')
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
        r = RP.REFPROPdll('', 'QT','DLIQ,DVAP,P', RP.MOLAR_BASE_SI, 0,0,Q,T,z)
        if r.ierr > 0:
            print('ERROR:', r.herr)
            return [np.nan]*5
        DLIQ, DVAP = r.Output[0:2]
        P = r.Output[2]
        y = r.y
        return DLIQ*z[0], DLIQ*z[1], DVAP*y[0], DVAP*y[1], P
    df[['rhoL_1 / mol/m^3', 'rhoL_2 / mol/m^3', 'rhoV_1 / mol/m^3', 'rhoV_2 / mol/m^3','p(EOS) / Pa']] = df.apply(add_rhos, axis=1, result_type='expand')
    df.loc[pandas.isnull(df['rhoL_1 / mol/m^3']), 'skip'] = "Iteration to get guess values didn't succeed"
    return df