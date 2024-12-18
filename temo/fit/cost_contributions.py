import timeit
import numpy as np
import teqp
import scipy
import pandas
import scipy.interpolate

def calc_errrho(*, model, df, step=1, iterate=False):
    """ 
    Deviation function from PVT data

    Returns percentage signed relative deviation in density

    It is not actually the relative difference in density, because that
    would require iterative calculations. Instead, the Maclaurin series
    expansion around the experimental density is used to obtain a
    non-iterative estimate of this error. This error metric breaks down
    whn dp/drho|T is very close to zero.

    The iterate keyword argument can be used to turn on the iterative
    calculations, but they are much slower
    """

    def o(row):
        T = row['T / K']; rho = row['rho / mol/m^3']
        z = np.array([row['z_1 / mole frac.'], row['z_2 / mole frac.']])
        Ar0n = model.get_Ar02n(T, rho, z)
        R = model.get_R(z)
        dpdrho = R*T*(1 + 2*Ar0n[1] + Ar0n[2])
        p = rho*R*T*(1 + Ar0n[1])
        err_noniterative = -(p-row['p / Pa'])/rho/dpdrho*100

        if iterate:
            rho = row['rho / mol/m^3']*1.1
            for i in range(10):
                Ar0n = model.get_Ar02n(T, rho, z)
                Ar01 = Ar0n[1]; Ar02 = Ar0n[2]
                R = model.get_R(z)
                pEOS = rho*R*T*(1+Ar01)
                dpdrho = R*T*(1 + 2*Ar0n[1] + Ar0n[2])
                res = (pEOS-row['p / Pa'])/(row['p / Pa'])
                dresdrho = dpdrho/(row['p / Pa'])
                change = -res/dresdrho
                if abs(change/rho-1) < 1e-10 or abs(res) < 1e-12:
                    break
                rho += change
                # print(res)
            return (1-rho/row['rho / mol/m^3'])*100
        else:
            return err_noniterative
    return df.iloc[0:len(df):step].apply(o, axis=1)

def calc_errrho_devp(*, model, df, step=1):
    """ 
    Deviation function from PVT data where the deviation is in pressure

    Recommended for critical region where density deviations don't make sense
    """
    def o(row):
        T = row['T / K']; rho = row['rho / mol/m^3']
        z = np.array([row['z_1 / mole frac.'], row['z_2 / mole frac.']])
        Ar01 = model.get_Ar01(T, rho, z)
        R = model.get_R(z)
        p = rho*R*T*(1 + Ar01)
        return 100*(1-p/row['p / Pa'])
    return df.iloc[0:len(df):step].apply(o, axis=1)

def calc_errSOS(model, df, *, step=1, max_iter=10):
    """ 
    Deviation function from discrete speed of sound data at given temperature
    and pressure
    """
    def o(row):
        z_1 = row['z_1 / mole frac.']
        z = np.array([z_1, 1-z_1])
        T = row['T / K']; M = row['M / kg/mol']; p_Pa = row['p / Pa']

        # This is the starting density for iteration
        rho = row['rho(EOS) / mol/m^3']

        Ao20 = row['Ao20'] # This does not depend on the mixture model so long as the pure fluids are the same

        # A few steps of Newton polishing on density
        R = model.get_R(z)
        for i in range(max_iter):
            Ar0n = model.get_Ar02n(T, rho, z)
            Ar01 = Ar0n[1]
            pEOS = rho*R*T*(1+Ar01)
            dpdrho = R*T*(1 + 2*Ar01 + Ar0n[2])
            res = (pEOS-p_Pa)/p_Pa
            if abs(res) < 1e-8:
                break
            dresdrho = dpdrho/p_Pa
            rho += -res/dresdrho

        Ar0n = model.get_Ar02n(T, rho, z)
        Ar01 = Ar0n[1]; Ar02 = Ar0n[2]
        Ar11 = model.get_Ar11(T, rho, z)
        Ar20 = model.get_Ar20(T, rho, z)

        # M*w^2/(R*T) where w is the speed of sound
        # from the definition w = sqrt(dp/drho|s)
        Mw2RT = 1 + 2*Ar01 + Ar02 - (1 + Ar01 - Ar11)**2/(Ao20 + Ar20)
        if Mw2RT < 0:
            return 1e6
        w = (Mw2RT*R*T/M)**0.5
        if not np.isfinite(w):
            return 1e20

        return (1-w/row['w / m/s'])*100
    return df.iloc[0:len(df):step].apply(o, axis=1)

def calc_errrhosat(model, df, *, step=1, Q):
    """ 
    Deviation function for saturated liquid and/or vapor densities
    """
    def o(row):
        rho_meas = row['rho / mol/m^3']
        T = row['T / K']
        rhovecL = np.array([row['rhoL_1 / mol/m^3'], row['rhoL_2 / mol/m^3']])
        rhovecV = np.array([row['rhoV_1 / mol/m^3'], row['rhoV_2 / mol/m^3']])
        if np.isfinite(row['x_1 / mole frac.']):
            x_0 = row['x_1 / mole frac.']
            z = np.array([x_0, 1-x_0])
            swap = False 
        elif np.isfinite(row['y_1 / mole frac.']):
            y_0 = row['y_1 / mole frac.']
            z = np.array([y_0, 1-y_0])
            swap = True
        else:
            raise ValueError("Neither x_1 nor y_2 is provided")

        try:
            if not swap:
                code, rhovecLnew, rhovecVnew = model.mix_VLE_Tx(T, rhovecL, rhovecV, z, 1e-8, 1e-8, 1e-8, 1e-8, 20)
            else:
                code, rhovecVnew, rhovecLnew = model.mix_VLE_Tx(T, rhovecV, rhovecL, z, 1e-8, 1e-8, 1e-8, 1e-8, 20)

            if sum(~np.isfinite(rhovecLnew)) > 0:
                return 1e20
            if sum(~np.isfinite(rhovecVnew)) > 0:
                return 1e20
            # Check for trivial solutions and penalize them
            if np.max(np.abs(rhovecLnew - rhovecVnew)) < 1e-6*np.sum(rhovecLnew):
                return 1e20
            rho_model = sum(rhovecLnew) if Q == 0 else sum(rhovecVnew)
            rho_err = abs(1-rho_model/rho_meas)*100
            if not np.isfinite(rho_err):
                return 1e20
            else:
                return rho_err
        except BaseException as BE:
            print(BE)
            return 1e20
    tic = timeit.default_timer()
    res = df.iloc[0:len(df):step].apply(o, axis=1)
    toc = timeit.default_timer()
    # print(toc-tic, 's for rhosat error')
    return res

def calc_errVLE(model, df, *, step=1):
    """ 
    Deviation function from VLE data
    """
    def o(row):
        T = row['T / K']
        rhovecL = np.array([row['rhoL_1 / mol/m^3'], row['rhoL_2 / mol/m^3']])
        rhovecV = np.array([row['rhoV_1 / mol/m^3'], row['rhoV_2 / mol/m^3']])
        if np.isfinite(row['x_1 / mole frac.']):
            x_0 = row['x_1 / mole frac.']
            z = np.array([x_0, 1-x_0])
            swap = False 
        elif np.isfinite(row['y_1 / mole frac.']):
            y_0 = row['y_1 / mole frac.']
            z = np.array([y_0, 1-y_0])
            swap = True
        else:
            raise ValueError("Neither x_1 nor y_2 is provided")
        
        p_meas = row['p / Pa']

        try:
            if not swap:
                code, rhovecLnew, rhovecVnew = model.mix_VLE_Tx(T, rhovecL, rhovecV, z, 1e-8, 1e-8, 1e-8, 1e-8, 20)
            else:
                code, rhovecVnew, rhovecLnew = model.mix_VLE_Tx(T, rhovecV, rhovecL, z, 1e-8, 1e-8, 1e-8, 1e-8, 20)
            # print(rhovecL, rhovecV, 'LV]]')
            
            # print('----->', rhovecLnew, rhovecVnew, 'LV after]')
            if sum(~np.isfinite(rhovecLnew)) > 0:
                return 1e20
            if sum(~np.isfinite(rhovecVnew)) > 0:
                return 1e20
            # Check for trivial solutions and penalize them
            if np.max(np.abs(rhovecLnew - rhovecVnew)) < 1e-6*np.sum(rhovecLnew):
                print('trivial solution')
                return 1e20
            # Check for invalid pressures
            p = rhovecLnew.sum()*model.get_R(z)*T*(1+model.get_Ar01(T, rhovecLnew.sum(), z))
            if not np.isfinite(p):
                print('not finite p')
                return 1e20
            
            # y = rhovecVnew/rhovecVnew.sum()
            # pV = rhovecVnew.sum()*model.get_R(y)*T + model.get_pr(T, rhovecVnew)
            p_err = abs(1-p/p_meas)*100
            # print(p_err)
            # if abs(pV/p-1) > 0.01:
            #     print('bad solver!', code, pV/p-1)
            if not np.isfinite(p_err):
                return 1e20
            else:
                return p_err
        except BaseException as BE:
            print(BE)
            return 1e20
    tic = timeit.default_timer()
    res = df.iloc[0:len(df):step].apply(o, axis=1)
    toc = timeit.default_timer()
    # print(toc-tic, 's for vle error')
    return res

def calc_errVLE_x(model, df, *, step=1):
    """ 
    Deviation function from VLE data in the x direction
    """
    def o(row):
        T = row['T / K']
        p_meas = row['p / Pa']
        rhovecL = np.array([row['rhoL_1 / mol/m^3'], row['rhoL_2 / mol/m^3']])
        rhovecV = np.array([row['rhoV_1 / mol/m^3'], row['rhoV_2 / mol/m^3']])
        try:
            opt = teqp.MixVLETpFlags()
            r = model.mix_VLE_Tp(T, p_meas, rhovecL, rhovecV, opt)
            rhovecLnew = r.rhovecL; rhovecVnew = r.rhovecV
            x = rhovecLnew/rhovecLnew.sum()
            # Check for trivial solutions and penalize them
            if np.max(np.abs(rhovecLnew - rhovecVnew)) < 1e-6*np.sum(rhovecLnew):
                return 1e20
            # Penalize failed iterations where the residuals are not all close to zero
            if np.max(np.abs(r.r)) > 1e-2:
                # print(r.num_iter, r.r, r.return_code)
                # return 1e6
                pass

            x_err = 100*(x[0] - row['x_1 / mole frac.'])
            # pL = rhovecL.sum()*model.get_R(x)*T*(1+model.get_Ar01(T, rhovecL.sum(), x))
            # pV = rhovecV.sum()*model.get_R(y)*T*(1+model.get_Ar01(T, rhovecV.sum(), y))
            # # Penalize failed iterations where the pressures are not equal
            # if abs(pL/pV-1) > 0.01:
            #     return 1e20
            
            if not np.isfinite(x_err) or x[0] > 1 or x[0] < 0:
                return 1e20
            else:
                # print(x_err, row['x_1 / mole frac.'], x)
                if np.abs(x_err) > 100:
                    print(x_err)
                return x_err
        except BaseException as BE:
            print(BE)
            return 1e20
    tic = timeit.default_timer()
    res = df.iloc[0:len(df):step].apply(o, axis=1)
    toc = timeit.default_timer()
    # print(toc-tic, 's for vle error')
    return res

def calc_errB12(*, model, df, step=1, z0):
    """
    B12 should not have dependence on composition, but alas, it usually does
    """
    z = np.array([z0, 1-z0])
    def o(row):
        B12calc = model.get_B12vir(row['T / K'], z)
        return (B12calc-row['B12 / m^3/mol'])
    return df.iloc[0:len(df):step].apply(o, axis=1)

def calc_errtracecrit(model, df, *, T0, rhovec0, errscheme, step=1):
    """ 
    Deviation function from tracing critical curve
    """

    tic = timeit.default_timer()
    try:
        opt = teqp.TCABOptions()
        # print(dir(opt))
        opt.init_dt = 100 # step in the arclength parameter
        opt.integration_order = 5
        opt.max_step_count = 300
        # opt.rel_err = 1e-10
        # opt.abs_err = 1e-13
        opt.max_dt = 1000
        opt.polish = True
        opt.polish_reltol_T = 100 # very generous polishing bounds
        opt.polish_reltol_rho = 100 # vey generous polishing bounds
        
        curveJSON = model.trace_critical_arclength_binary(T0, rhovec0, '', opt)
        crit = pandas.DataFrame(curveJSON)
        
        if errscheme == 'log(p)xdistance':
            rhotot = crit['rho0 / mol/m^3']+crit['rho1 / mol/m^3']
            crit['z0 / mole frac.'] = crit['rho0 / mol/m^3']/rhotot
            # We are tracing to the critical point, so both phases should have zero distance
            XA = np.c_[crit['z0 / mole frac.'], crit['p / Pa']/1e6] # From the trace
            XB = np.c_[df['z_1 / mole frac.'], df['p / Pa']/1e6] # From the measurements
            e1 = float(scipy.spatial.distance.cdist(XA, XB).min(axis=0))
            toc3 = timeit.default_timer()
            return e1
        elif errscheme in ['T','T*']:
            rhotot = crit['rho0 / mol/m^3']+crit['rho1 / mol/m^3']
            crit['z0 / mole frac.'] = crit['rho0 / mol/m^3']/rhotot
            Tinterp = scipy.interpolate.interp1d(crit['z0 / mole frac.'], crit['T / K'], bounds_error=False, fill_value=1e20)(df['z_1 / mole frac.'])
            if errscheme == 'T*':
                print(Tinterp)
            return np.mean(np.abs(Tinterp-df['T / K']))
        elif errscheme == 'TdevP':
            # Interpolate for given value of T to find p along critical curve, compare
            # with the measured critical pressure
            def get_err(row):
                try:
                    p_crit_curve = scipy.interpolate.interp1d(crit['T / K'], crit['p / Pa'])(row['T / K'])
                    return 100*(1-p_crit_curve/row['p / Pa'])
                except BaseException as be:
                    # print(np.min(crit['T / K']), np.max(crit['T / K']))
                    # print(be)
                    return 100
            return df.iloc[0:len(df):step].apply(get_err, axis=1)
        else:
            raise KeyError("Bad errscheme")
    except BaseException as be:
        print(be)
        return 1000.0

def calc_errcritPVT(model, df, *, step=1):
    """ 
    Deviation function from critical points for which temperature 
    and density are specified
    """
    def o(row):
        z_1 = row['z_1 / mole frac.']
        z = np.array([z_1, 1-z_1])
        rho = row['rho / mol/m^3']
        if not pandas.isnull(z_1) and not pandas.isnull(rho):
            conds = model.get_criticality_conditions(row['T / K'], z*rho)
            return np.sum(np.abs(conds)*1e4)
        else:
            return 0
    return df.iloc[0:len(df):step].apply(o, axis=1)

def calc_err_critisoT(model, df, *, step=1):
    """ 
    Deviation function for critical points, tracing from a pure
    fluid endpoint. Tracing the isotherms ensures that
    the isotherms are well-behaved in the critical region. While
    the isotherm tracing to the critical point is relatively slow,
    it is really important to ensure well-shaped isotherms in the
    critical region

    Two parts come into the deviation term:
    1) The pressure is supposed to be close to the measured critical pressure
    2) The isotherm needs to close, so that the liquid and vapor traces have
       converged to the same composition given by the critical point. 
       The slope dp/dx at the critical point should be zero, but this is not checked
    """
    def o(row):
        opt = teqp.TVLEOptions(); opt.polish=True; opt.integration_order=5; 
        opt.calc_criticality=True; opt.max_steps=200; opt.terminate_unstable=True
        rhovecL = np.array([row['rhoL_pure_1 / mol/m^3'], row['rhoL_pure_2 / mol/m^3']])
        rhovecV = np.array([row['rhoV_pure_1 / mol/m^3'], row['rhoV_pure_2 / mol/m^3']])
        o = model.trace_VLE_isotherm_binary(row['T / K'], rhovecL, rhovecV, opt)
        d = pandas.DataFrame(o)
        ipmax = np.argmax(np.array(d['pL / Pa']))
        pmax = d['pL / Pa'].iloc[ipmax]
        deltaxL = np.abs(d['xV_0 / mole frac.'].iloc[ipmax] - row['z_1 / mole frac.'])
        deltaxV = np.abs(d['xL_0 / mole frac.'].iloc[ipmax] - row['z_1 / mole frac.'])
        if not np.isfinite(deltaxL):
            deltaxL = 1e4
        if not np.isfinite(deltaxV):
            deltaxV = 1e4
        if not np.isfinite(pmax):
            pmax = 1e4
        return np.abs(100*(1-pmax/row['p / Pa'])) + np.abs(deltaxL)*100 + np.abs(deltaxV)*100
    return df.iloc[0:len(df):step].apply(o, axis=1)

def calc_errcritPT(model, df, *, step=1):
    """ 
    Deviation function from critical points for which temperature 
    and pressure are specified, and density guess is provided
    The rho(T,p) solver is executed to solve for the matching
    density and then the criticality conditions are checked
    """
    def o(row):
        z_1 = row['z_1 / mole frac.']
        rho = row['rho / mol/m^3']
        T = row['T / K']
        if not pandas.isnull(z_1) and not pandas.isnull(rho):
            z = np.array([z_1, 1-z_1])
            # Solve for density
            for i in range(10):
                Ar0n = model.get_Ar02n(T, rho, z)
                Ar01 = Ar0n[1]; Ar02 = Ar0n[2]
                R = model.get_R(z)
                pEOS = rho*R*T*(1+Ar01)
                dpdrho = R*T*(1 + 2*Ar01 + Ar02)
                res = (pEOS-row['p / Pa'])/(row['p / Pa'])
                dresdrho = dpdrho/(row['p / Pa'])
                change = -res/dresdrho
                if abs(change/rho-1) < 1e-10 or abs(res) < 1e-12:
                    break
                rho += change
                # print(res)
            conds = model.get_criticality_conditions(T, z*rho)
            return np.sum(np.abs(conds)*1e4)
        else:
            return 0
    return df.iloc[0:len(df):step].apply(o, axis=1)