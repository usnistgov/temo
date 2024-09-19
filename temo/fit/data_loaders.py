import pandas
import numpy as np
import CoolProp.CoolProp as CP

def only_the_fluids(df, identifier, identifiers):
    num_matches = 0
    for key in df:
        if key.startswith(identifier):
            df = df[df[key].isin(identifiers)]
            num_matches += 1
    if num_matches == 0:
        raise ValueError(f"Unable to match a single instance of your identifier root: {identifier}")
    return df.copy()

def read_and_subset(path, identifier, identifiers, apply_skip, sep=','):
    df = pandas.read_csv(path, comment='#', sep=sep)
    df = only_the_fluids(df, identifier, identifiers)
    if 'skip' in df and apply_skip:
        df = df[pandas.isnull(df.skip)]
    return df



def load_SOS(dataroot, *, apply_skip=True, identifier, identifiers, output_csv=None, molar_masses, verbosity=1, sep=','):
    """ Loader for speed of sound data """
    df = read_and_subset(dataroot+'/SOS.csv', identifier=identifier, identifiers=identifiers, apply_skip=apply_skip, sep=sep)
    if df.empty:
        raise ValueError(f"No rows remained after loading the SOS and applying filter:: {{identifier: {identifier}, identifiers: {identifiers}}}")
    z_1 = df['z_1 / mole frac.']
    df['M / kg/mol'] = z_1*molar_masses[0] + (1-z_1)*molar_masses[1]
    required_columns = ['T / K', 'Ao20', 'bibkey']
    missing_columns = [col for col in required_columns if col not in df]
    if any(missing_columns):
        raise KeyError("Required column not found in SOS data: " + str(missing_columns))
    
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    if verbosity > 0:
        print(f"Loaded {len(df)} rows from {dataroot+'/SOS.csv'}")
    
    def get_p_Pa(row):
        if 'p / Pa' in row and not pandas.isnull(row['p / Pa']):
            return row['p / Pa']
        else:
            factors = {'p / Pa': 1.0, 'p / kPa': 1e3, 'p / MPa': 1e6, 'p / GPa': 1e9}
            for k, factor in factors.items():
                if k in row and not pandas.isnull(row[k]):
                    return row[k]*factor
            raise ValueError("no pressure was specified; allowed values are:"+str(factors.keys()))
    
    df['p / Pa'] = df.apply(get_p_Pa, axis=1)
    
    return df

def _density_processing(df, molar_masses=None):
    allowed_press_cols = ['p / Pa', 'p / kPa', 'p / MPa', 'p / GPa']
    allowed_density_cols = ['rho / kg/m^3', 'rho / mol/m^3']

    for key, gp in df.groupby('bibkey'):

        # Pressure in Pa, kPa, MPa, or GPa
        provided_press_cols = [col for col in allowed_press_cols if (col in gp and all(~pandas.isnull(gp[col])))]
        if len(provided_press_cols) != 1:
            raise ValueError(f"One and only one of the pressure options {allowed_press_cols} must be"
            f" provided for bibkey of {key}. You provided: {provided_press_cols}")

        # Density either in kg/m^3 or mol/m^3    
        provided_density_cols = [col for col in allowed_density_cols if (col in gp and all(~pandas.isnull(gp[col])))]
        if len(provided_density_cols) != 1:
            raise ValueError(f"One and only one of the density options [{allowed_density_cols}] must be"
            f" provided for bibkey of {key}. You provided: {provided_density_cols}")

    def get_molar_density(row):
        if 'rho / mol/m^3' in row and not pandas.isnull(row['rho / mol/m^3']):
            return row['rho / mol/m^3']
        else:
            M = row['z_1 / mole frac.']*molar_masses[0] + row['z_2 / mole frac.']*molar_masses[1]
            return row['rho / kg/m^3']/M
    
    # Convert to molar density
    df['rho / mol/m^3'] = df.apply(get_molar_density, axis=1)

    def get_p_Pa(row):
        if 'p / Pa' in row and not pandas.isnull(row['p / Pa']):
            return row['p / Pa']
        else:
            factors = {'p / Pa': 1.0, 'p / kPa': 1e3, 'p / MPa': 1e6, 'p / GPa': 1e9}
            for k, factor in factors.items():
                if k in row and not pandas.isnull(row[k]):
                    return row[k]*factor
            raise ValueError("no pressure was specified; allowed values are:"+str(factors.keys()))
    df['p / Pa'] = df.apply(get_p_Pa, axis=1)
    return df.copy()

def load_PVT(dataroot, *, identifier, identifiers, apply_skip=True, output_csv=None, molar_masses, verbosity=1, sep=','):
    """ Loader for p-v-T data """
    df = read_and_subset(dataroot+'/PVT.csv', identifier=identifier, identifiers=identifiers, apply_skip=apply_skip, sep=sep)
    if df.empty:
        raise ValueError(f"No rows remained after loading the PVT and applying filter:: {{identifier: {identifier}, identifiers: {identifiers}}}")
    df = _density_processing(df, molar_masses=molar_masses)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    if verbosity > 0:
        print(f"Loaded {len(df)} rows from {dataroot+'/PVT.csv'}")

    return df

def load_PVT_P(dataroot, *, identifier, identifiers, apply_skip=True, output_csv=None, molar_masses, verbosity=1, sep=','):
    """ Loader for p-v-T data with pressure deviations """
    df = read_and_subset(dataroot+'/PVT_P.csv', identifier=identifier, identifiers=identifiers, apply_skip=False, sep=sep)
    if df.empty:
        raise ValueError(f"No rows remained after loading the PVT_P and applying filter:: {{identifier: {identifier}, identifiers: {identifiers}}}")
    df = _density_processing(df, molar_masses=molar_masses)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    if verbosity > 0:
        print(f"Loaded {len(df)} rows from {dataroot+'/PVT_P.csv'}")

    return df

def load_VLE(dataroot, identifier, identifiers, apply_skip=True, output_csv=None, verbosity=1, molar_masses=None, sep=','):
    """ Loader for VLE data """
    df = read_and_subset(dataroot+'/VLE.csv', identifier=identifier, identifiers=identifiers, apply_skip=apply_skip, sep=sep)
    if df.empty:
        raise ValueError(f"No rows remained after loading the VLE and applying filter:: {{identifier: {identifier}, identifiers: {identifiers}}}")

    required_columns = ['T / K', 'kind']
    missing_columns = [col for col in required_columns if col not in df]
    if missing_columns:
        raise KeyError("Required column not found in VLE data: " + str(missing_columns))
    
    def force_skip_pure(df, key):
        df = df[~(df[key] == 0.0)]
        df = df[~(df[key] == 1.0)]
        return df
        
    for kind, gp in df.groupby('kind'):
        if kind == 'PTXY':
            required_columns = ['x_1 / mole frac.', 'y_1 / mole frac.']
            missing_columns = [col for col in required_columns if col not in gp]
            if missing_columns:
                raise KeyError("Required column not found in VLE data: " + str(missing_columns))
            for col in required_columns:
                df = force_skip_pure(df, col)
        elif kind == 'BUB':
            required_columns = ['x_1 / mole frac.']
            missing_columns = [col for col in required_columns if col not in gp]
            if missing_columns:
                raise KeyError("Required column not found in BUB data: " + str(missing_columns))
            for col in required_columns:
                df = force_skip_pure(df, col)
        elif kind == 'DEW':
            required_columns = ['y_1 / mole frac.']
            missing_columns = [col for col in required_columns if col not in gp]
            if missing_columns:
                raise KeyError("Required column not found in DEW data: " + str(missing_columns))
            for col in required_columns:
                df = force_skip_pure(df, col)
        else:
            raise KeyError(f'Provided kind of "{kind}" is not in the set of {{"PTXY","BUB","DEW"}}')

    def get_p_Pa(row):
        if 'p / Pa' in row and not pandas.isnull(row['p / Pa']):
            return row['p / Pa']
        else:
            factors = {'p / Pa': 1.0, 'p / kPa': 1e3, 'p / MPa': 1e6, 'p / GPa': 1e9}
            for k, factor in factors.items():
                if k in row and not pandas.isnull(row[k]):
                    return row[k]*factor
            raise ValueError("no pressure was specified; allowed values are: " + str(factors.keys()))
    df['p / Pa'] = df.apply(get_p_Pa, axis=1)
    
    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    if verbosity > 0:
        print(f"Loaded {len(df)} rows from {dataroot+'/VLE.csv'}")

    return df

def load_CRIT(dataroot, identifier, identifiers, apply_skip=True, output_csv=None, verbosity=1, molar_masses=None, sep=','):
    """ Loader for critical point data """
    df = read_and_subset(dataroot+'/CRIT.csv', identifier=identifier, identifiers=identifiers, apply_skip=apply_skip, sep=sep)

    required_columns = ['T / K', 'p / Pa']
    missing_columns = [col for col in required_columns if col not in df]
    if missing_columns:
        raise KeyError("Required column not found in CRIT data: " + str(missing_columns))

    def get_molar_density(row):
        if 'rho / mol/m^3' in row and pandas.isnull(row['rho / mol/m^3']) and 'rho / kg/m^3' in row and pandas.isnull(row['rho / kg/m^3']):
            return np.nan
        elif 'rho / mol/m^3' in row and not pandas.isnull(row['rho / mol/m^3']):
            return row['rho / mol/m^3']
        elif 'rho / kg/m^3' in row and not pandas.isnull(row['rho / kg/m^3']) :
            M = row['z_1 / mole frac.']*molar_masses[0] + row['z_2 / mole frac.']*molar_masses[1]
            return row['rho / kg/m^3']/M
        else:
            return np.nan
    
    # Convert to molar density
    df['rho / mol/m^3'] = df.apply(get_molar_density, axis=1)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    if verbosity > 0:
        print(f"Loaded {len(df)} rows from {dataroot+'/CRIT.csv'}")

    return df

# Parse B12data
def load_B12(dataroot, identifier, identifiers, apply_skip=True, output_csv=None, verbosity=1, molar_masses=None, sep=','):
    df = read_and_subset(dataroot+'/B12.csv', identifier=identifier, identifiers=identifiers, apply_skip=apply_skip, sep=sep)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df.copy()

if __name__ == '__main__':
    identifiers = ['AMMONIA', 'WATER']
    molar_masses=[CP.PropsSI('molemass',f) for f in identifiers]
    load_PVT('NH3H2O', identifier='FLD', identifiers=identifiers, molar_masses=molar_masses, output_csv=None)
    load_SOS('NH3H2O', identifier='FLD', identifiers=identifiers, molar_masses=molar_masses, output_csv=None)
    load_VLE('NH3H2O', identifier='FLD', identifiers=identifiers, molar_masses=molar_masses, output_csv=None)
    load_CRIT('NH3H2O', identifier='FLD', identifiers=identifiers, molar_masses=molar_masses, output_csv=None)