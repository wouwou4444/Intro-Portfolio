import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize

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
    rets.columns = (rets.columns.str.strip())
    
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

##################################################################################
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


def cvar_historic(r, level = 5):
    is_beyond = r <= -var_historic(r, level = level)
    return -r[is_beyond].mean()

##################################################################################
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

def cvar_gaussian(r, level = 5, modified = False):
    is_beyond = r <= -var_gaussian(r, level = level, modified = modified)
    return -r[is_beyond].mean()

##################################################################################
def annualize_ret(r, periods_per_year = 12):
    compound_growth = (1 +r).prod()
    n_periods = r.shape[0]
    return compound_growth**(periods_per_year/n_periods) - 1

def annualize_vol(r, periods_per_year = 12):
    return r.std(ddof=0) * (periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate,periods_per_year):
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_ret(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol



##################################################################################

##  Efficient Frontier

##################################################################################
def portfolio_return(weights, returns):
    return weights.T @ returns


def portfolio_vol(weights, cov):
    return (weights.T @ cov @ weights )**0.5

def plot_ef2(n_points, er, cov, style = ".-"):
    if er.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError('can only plot 2-asset Frontiers')
    weights = [np.array([w, 1 -w]) for w in np.linspace(0,1, n_points)]
    rets = [ portfolio_return(w, er) for w in weights]
    vols = [ portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"R": rets, "Vol": vols})
    return ef.plot.line(x = "Vol", y = "R", style = style, figsize = (16, 6))


def plot_ef(n_points, er, cov, style = ".-"):

    weights = optimal_weights(n_points, er, cov) # ???
    
    rets = [ portfolio_return(w, er) for w in weights]
    vols = [ portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"R": rets, "Vol": vols})
    return ef.plot.line(x = "Vol", y = "R", style = style, figsize = (16, 6))


def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [ minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def minimize_vol(target_return, er, cov, disp = False):
    """
    target_return -> weight vector
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda w, er: target_return - portfolio_return(w, er)
        }
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                      args = (cov,),
                      method="SLSQP",
                      options={'disp':  False},
                      constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    #weights in results.x
    if disp:
        return n, er, init_guess, bounds, return_is_target, results, results.x
    else:
        return results.x
    


def msr(riskfree_rate, er, cov, disp = False):
    """
    RiskFree Rate, er, cov -> msr
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
    def neg_sharpe_ratio(w, riskfree_rate, er, cov):
        r = portfolio_return(w, er)
        vol = portfolio_vol(w, cov)
        return -(r - riskfree_rate)/vol
        
    results = minimize(neg_sharpe_ratio, init_guess,
                      args = (riskfree_rate, er, cov,),
                      method="SLSQP",
                      options={'disp':  False},
                      constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    #weights in results.x
    if disp:
        return n, er, init_guess, bounds, return_is_target, results, results.x
    else:
        return results.x
    

    