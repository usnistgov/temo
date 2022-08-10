import timeit
import numpy as np
import teqp

def calc_errrho(*, model, df, step=1, iterate=False):
    """ 
    Deviation function from PVT data

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

def calc_errtraceSOS(model, df, *, step=1, isotherms):
    """ 
    Deviation function for speed of sound data from tracing VLE curves
    """
    tic = timeit.default_timer()
    def do_one_isotherm(T):
        # Collect the data rows that are close to the nominal isotherm
        gp = df[np.abs(df['T / K'] - T) < 1e-1]

        # Calculate the starting points for the pure
        rhovecL = np.array([CP.PropsSI('Dmolar','T',T,'Q',0,'NEON'), 0])
        rhovecV = np.array([CP.PropsSI('Dmolar','T',T,'Q',1,'NEON'), 0])
        SOS_meas = gp['w / m/s']

        # Now do the trace along the isotherm
        opts = teqp.TVLEOptions(); opts.max_steps = 200
        j = teqp.trace_VLE_isotherm_binary(model, T, rhovecL, rhovecV, opts)
        vle = pandas.DataFrame(j)

        # And calculate the mole fractions
        def get_xmole(row, key):
            rho = row[key]
            return rho[0]/sum(rho)
        vle['x_0 / mole frac.'] = vle.apply(get_xmole, axis=1, key='rhoL / mol/m^3')
        vle['y_0 / mole frac.'] = vle.apply(get_xmole, axis=1, key='rhoV / mol/m^3')

        # toc2 = timeit.default_timer(); print(toc2-tic, 'sec. for SOS tracing')

        # And calculate the SOS
        def add_SOS(row):
            z0 = row['x_0 / mole frac.']
            z = np.array([z0, 1-z0])
            rho = sum(row['rhoL / mol/m^3'])
            T = row['T / K']
            return self.calc_SOS(self.AS, model, T, rho, z)
        vle['w / m/s'] = vle.apply(add_SOS, axis=1)

        # Calculate the distance
        XA = np.c_[vle['x_0 / mole frac.'], vle['w / m/s']/1000]
        XB = np.c_[gp['z_1 / mole fraction'], SOS_meas/1000]
        e = scipy.spatial.distance.cdist(XA, XB).min(axis=0)

        # toc3 = timeit.default_timer(); print(toc3-toc2, 'sec. for error calc')

        return e

    errs = 0
    for T in isotherms:
        iso = do_one_isotherm(T)
        err = iso.sum()
        if np.isfinite(err):
            errs += err
        else:
            errs += 10000000
    toc = timeit.default_timer()
    # print(toc-tic, 'sec. for SOS tracing error')
    return errs

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
        R = model.get_R(z)

        # M*w^2/(R*T) where w is the speed of sound
        # from the definition w = sqrt(dp/drho|s)
        Mw2RT = 1 + 2*Ar01 + Ar02 - (1 + Ar01 - Ar11)**2/(Ao20 + Ar20)
        w = (Mw2RT*R*T/M)**0.5

        return (1-w/row['w / m/s'])*100
    return df.iloc[0:len(df):step].apply(o, axis=1)

def calc_errtraceVLE(model, df, *, step=1, isotherms):
    """ 
    Deviation function from tracing VLE data
    """
    tic = timeit.default_timer()
    def do_one_isotherm(T):
        # Collect the data rows that are close to the nominal isotherm
        gp = df[np.abs(df['T / K'] - T) < 1e-1]

        # Extract data from the DataFrame
        #rhovecL = np.array([gp['rhoLpure_0 / mol/m^3'].mean(), 0])
        #rhovecV = np.array([gp['rhoVpure_0 / mol/m^3'].mean(), 0])

        rhovecL = np.array([CP.PropsSI('Dmolar','T',T,'Q',0,'NEON'), 0])
        rhovecV = np.array([CP.PropsSI('Dmolar','T',T,'Q',1,'NEON'), 0])
        p_meas = gp['p / Pa']
        # print(T, rhovecL)

        # Now do the trace along the isotherm
        opts = teqp.TVLEOptions(); opts.max_steps = 200
        j = teqp.trace_VLE_isotherm_binary(model, T, rhovecL, rhovecV, opts)
        toc2 = timeit.default_timer()
        # print(toc2-tic,'trace')
        vle = pandas.DataFrame(j)
        # And calculate the mole fractions
        def get_xmole(row, key):
            rho = row[key]
            return rho[0]/sum(rho)
        vle['x_0'] = vle.apply(get_xmole, axis=1, key='rhoL / mol/m^3')
        vle['y_0'] = vle.apply(get_xmole, axis=1, key='rhoV / mol/m^3')

        XA = np.c_[vle['x_0'], vle['pL / Pa']/1e6]
        XB = np.c_[gp['x0 / mole frac.'], p_meas/1e6]
        e1 = scipy.spatial.distance.cdist(XA, XB).min(axis=0)

        XA = np.c_[vle['y_0'], vle['pL / Pa']/1e6]
        XB = np.c_[gp['y0 / mole frac.'], p_meas/1e6]
        e2 = scipy.spatial.distance.cdist(XA, XB).min(axis=0)
        toc3 = timeit.default_timer()
        # print(toc3-toc2,'dist')

        return e1 + e2

        # # Calculate the pressure error for a single data row
        # def get_pmodel(z_0):
        #     if vle['x_0'].min() < z_0 < vle['x_0'].max():
        #         # Within range of composition, interpolate
        #         p_model = scipy.interpolate.interp1d(, 'cubic')(z_0)
        #     else:
        #         # Composition is outside the range of the trace, find the closest composition
        #         imin = np.argmin(np.abs(vle['x_0']-z_0))
        #         p_model = vle['pL / Pa'].iloc[imin]
        #     return p_model
        # p_model = np.array([get_pmodel(z_0) for z_0 in gp['x0 / mole frac.']])
        # print(p_model, p_meas)
        # return (p_meas/p_model-1)*100

    errs = 0
    for T in isotherms:
        iso = do_one_isotherm(T)
        err = iso.sum()
        if np.isfinite(err):
            errs += err
        else:
            errs += 10000000
    toc = timeit.default_timer()
    # print(toc-tic, 's for vle error')
    return errs

def calc_errVLE(model, df, *, step=1):
    """ 
    Deviation function from VLE data
    """
    def o(row):
        T = row['T / K']
        rhovecL = np.array([row['rhoL_1 / mol/m^3'], row['rhoL_2 / mol/m^3']])
        rhovecV = np.array([row['rhoV_1 / mol/m^3'], row['rhoV_2 / mol/m^3']])
        x_0 = row['x_1 / mole frac.']
        z = np.array([x_0, 1-x_0])
        p_meas = row['p / Pa']
        try:
            code, rhovecLnew, rhovecVnew = teqp.mix_VLE_Tx(model, T, rhovecL, rhovecV, z, 1e-8, 1e-8, 1e-8, 1e-8, 10)
            # Check for trivial solutions and penalize them
            if np.max(np.abs(rhovecLnew - rhovecVnew)) < 1e-6*np.sum(rhovecLnew):
                return 1e20
            p = rhovecLnew.sum()*model.get_R(z)*T + teqp.get_pr(model, T, rhovecLnew)
            p_err = (1-p/p_meas)*100
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

def calc_errB12(model, df, *, step=1, z0):
    """
    B12 should not have dependence on composition, but alas, it usually does
    """
    z = np.array([z0, 1-z0])
    def o(row):
        B12calc = teqp.get_B12vir(model, row['T / K'], z)
        return B12calc-row['B12 / m^3/mol']
    return df.iloc[0:len(df):step].apply(o, axis=1)