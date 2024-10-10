import pandas 
import scipy.interpolate
import teqp 
import numpy as np

class BinaryVLEIsothermFitter:
    
    def __init__(self, *, ipure: int, T_K: float, anc, df_isoT:pandas.DataFrame, identifiers:list[str], component_json:list[dict], p_required:bool=True, tidy:bool=False):
        """ A helper class for fitting data points along vapor-liquid equilibrium isotherms for binary mixtures

        Args:
            ipure (int): 0-based index for the fluid from which tracing should start
            T_K (float): temperature
            anc: ancillary functions
            df_isoT (pandas.DataFrame): Contents used for the isotherm fitting
            identifiers (list[str]): List of identifiers to be used for each fluid, usually the FLD from REFPROP, or could be some other string thing
            component_json (list[json]): A list of JSON-compatible python data structures to be passed to make_model
            p_required(bool): Should the pressure be required to be found in a row. Not necessary for saturated states where T, Q fixes the state
            tidy(bool): If true, do the cleanup of the DataFrame to remove invalid rows
        """
        self.ipure = ipure
        self.T_K = T_K
        self.anc = anc 
        self.identifiers = identifiers
        if tidy:
            self.df_isoT = self._tidy_dataframe(df_isoT, p_required=p_required)
        else:
            self.df_isoT = df_isoT
        if len(self.df_isoT) == 0:
            raise ValueError("No valid fitting points can be found")
        self.component_json = component_json
        
        rhoL0 = self.anc.rhoL(self.T_K)
        rhoV0 = self.anc.rhoV(self.T_K)
        
        modelpure = teqp.build_multifluid_model([component_json[ipure]], teqp.get_datapath())
        rhoL0, rhoV0 = modelpure.pure_VLE_T(T_K, rhoL0, rhoV0, 10)
        self.rhovecL0 = np.array([0.0, 0.0]); self.rhovecL0[ipure] = rhoL0
        self.rhovecV0 = np.array([0.0, 0.0]); self.rhovecV0[ipure] = rhoV0
        
    def _tidy_dataframe(self, gp, *, p_required):
        """ 
        A convenience function to strip out pure fluid data points as well
        as rows where the pressure is not specifified
        """
        gp = gp.copy()
        gp.dropna(axis=0, subset=['x_1 / mole frac.'], inplace=True)
        if p_required:
            if 'p / kPa' in gp:
                gp = gp[~pandas.isnull(gp['p / kPa'])]
            if 'p / Pa' in gp:
                gp = gp[~pandas.isnull(gp['p / Pa'])]
        gp = gp[gp['x_1 / mole frac.'] != 0]
        gp = gp[gp['x_1 / mole frac.'] != 1]
        gp = gp[gp['y_1 / mole frac.'] != 0]
        gp = gp[gp['y_1 / mole frac.'] != 1]
        return gp
        
    def build_model(self, gammaT:float) -> teqp.AbstractModel:
        """ Construct the teqp.AbstractModel instance based on the specified value
        of 𝛾_T
        """
        BIP = [{
            'BibTeX': 'fitting_in_progress',
            'CAS1': '?', 'CAS2': '?',
            'F': 0.0,
            'Name1': self.identifiers[0],
            'Name2': self.identifiers[1],
            'betaT': 1.0,
            'betaV': 1.0,
            'gammaT': gammaT,
            'gammaV': 1.0
        }]
        jmodel = {
            "components": self.component_json,
            "BIP": BIP,
            "root": teqp.get_datapath()
        }        
        model = teqp.make_model({'kind': 'multifluid', 'model': jmodel})
        return model 
        
    def trace_isotherm(self, gammaT:float=None, model:teqp.AbstractModel=None) -> pandas.DataFrame:
        """ Given either a model instance or a value of 𝛾_T, trace the isotherm"""
        if model is None:
            model = self.build_model(gammaT)
        trace = pandas.DataFrame(model.trace_VLE_isotherm_binary(self.T_K, self.rhovecL0, self.rhovecV0))
        return trace
        
    def cost_function(self, gammaT:float) -> float:
        """ Evaluate the cost function that is to be minimized.
        
        Here the cost function is based solely on the interpolated values of the bubble-point pressure
        """
        # print('start trace')
        trace = self.trace_isotherm(gammaT)
        # print('end trace')
        tx_interpolator = scipy.interpolate.interp1d(trace['xL_0 / mole frac.'], trace['t'],fill_value=np.nan, bounds_error=False)
        pt_interpolator = scipy.interpolate.interp1d(trace['t'], trace['pL / Pa'], fill_value=np.nan, bounds_error=False)

        tx = tx_interpolator(self.df_isoT['x_1 / mole frac.'])

        # Cost function is currently in pressure only
        p_interp = pt_interpolator(tx)
        p_dev = np.abs(p_interp/(self.df_isoT['p / Pa'])-1)
        AAD_p = np.mean(np.abs(p_dev))

        cost = AAD_p
        if not np.isfinite(cost):
            return 1e6
        # print(gammaT, cost)
        return cost
    
class BinaryVLEIsothermInterpolator:
    """
    A class to hold some interpolator classes along VLE isothems
    """
    def __init__(self, trace: pandas.DataFrame):
        RHOL = np.array(trace['rhoL / mol/m^3'].tolist())
        RHOV = np.array(trace['rhoV / mol/m^3'].tolist())
        
        self.t_interpolator_rhoL = scipy.interpolate.interp1d(RHOL.sum(axis=1), trace['t'], fill_value=np.nan, bounds_error=False)
        self.t_interpolator_rhoV = scipy.interpolate.interp1d(RHOV.sum(axis=1), trace['t'], fill_value=np.nan, bounds_error=False)
        
        self.t_interpolator_x1 = scipy.interpolate.interp1d(trace['xL_0 / mole frac.'], trace['t'], fill_value=np.nan, bounds_error=False)
        self.t_interpolator_y1 = scipy.interpolate.interp1d(trace['xV_0 / mole frac.'], trace['t'], fill_value=np.nan, bounds_error=False)
        self.t_interpolator_p = scipy.interpolate.interp1d(trace['pL / Pa'], trace['t'], fill_value=np.nan, bounds_error=False)

        self.rhoL_interpolator_t = scipy.interpolate.interp1d(trace['t'], RHOL.sum(axis=1), fill_value=np.nan, bounds_error=False)
        self.rhoV_interpolator_t = scipy.interpolate.interp1d(trace['t'], RHOV.sum(axis=1), fill_value=np.nan, bounds_error=False)

        self.rhoL1_interpolator_t = scipy.interpolate.interp1d(trace['t'], RHOL[:, 0], fill_value=np.nan, bounds_error=False)
        self.rhoL2_interpolator_t = scipy.interpolate.interp1d(trace['t'], RHOL[:, 1], fill_value=np.nan, bounds_error=False)
        self.rhoV1_interpolator_t = scipy.interpolate.interp1d(trace['t'], RHOV[:, 0], fill_value=np.nan, bounds_error=False)
        self.rhoV2_interpolator_t = scipy.interpolate.interp1d(trace['t'], RHOV[:, 1], fill_value=np.nan, bounds_error=False)
        self.p_interpolator_t = scipy.interpolate.interp1d(trace['t'], trace['pL / Pa'], fill_value=np.nan, bounds_error=False)