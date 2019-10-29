import pandas as pd

def drawdown(return_series: pd.Series):
    """
    return Wealth, drawdown for a return series
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = - (previous_peaks - wealth_index) / previous_peaks 
    return pd.DataFrame({"Wealth": wealth_index,
                        "Previous Peak": previous_peaks,
                        "DrawDown": drawdowns})

def get_ffme_returns():
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                             header = 0, index_col = 0 , na_values = -99.99)
    rets = me_m[["Lo 10", "Hi 10"]] / 100
    rets.columns = [ "SmallCap", "LargeCap"]
    rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period("M")
    rets.index.name = "Period"
    
    return rets

def get_hfi_returns():
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                             header = 0, index_col = 0 , parse_dates = True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")
    hfi.index.name = "Period"
    
    return hfi

def get_ind_returns():
    me_m = pd.read_csv("data/ind30_m_vw_rets.csv",
                             header = 0, index_col = 0 , na_values = -99.99)
    rets = me_m / 100
    rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period("M")
    rets.index.name = "Period"
    rets.columns = list(rets.columns.str.strip())
    
    return rets


def skewness(serie_returns:pd.Series):
    deviation = serie_returns - serie_returns.mean()
    deviation3 = (deviation ** 3).mean()
    result = deviation3 / (serie_returns.std(ddof=0))**3
    return result

def skewness(serie_returns:pd.Series, exponent = 3):
    deviation = serie_returns - serie_returns.mean()
    deviation3 = (deviation ** exponent).mean()
    result = deviation3 / (serie_returns.std(ddof=0))**exponent
    return result

def kurtosis(r):
    return skewness(r, exponent = 4)

import scipy.stats
def is_normal(r, level = 0.01):
    """
    """
    return scipy.stats.jarque_bera(r)


def semideviation(r):
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)


import numpy as np
def var_historic(r, level = 5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level, axis = 0)
    else:
        raise TypeError("Expected r to be a pandas Series or DataFrame")
        
from scipy.stats import norm
def var_gaussian(r, level = 5, modified = False):
    z_norm = norm.ppf(level/100)
    z_norm_mod = z_norm
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z_norm_mod = ( z_norm + 
                  (z_norm**2 - 1)*s/6 + 
                  (z_norm**3 - 3*z_norm)*(k-3)/24 -
                  (2 * z_norm**3 - 5 * z_norm)*(s**2)/36
                 )
    
    return -(r.mean() + z_norm_mod * r.std(ddof = 0))