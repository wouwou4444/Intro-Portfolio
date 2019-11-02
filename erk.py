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


def get_ind_size():
    ind = pd.read_csv("data/ind30_m_size.csv",
                             header = 0, index_col = 0 , na_values = -99.99)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.index.name = "Period"
    ind.columns = (ind.columns.str.strip())
    
    return ind

def get_ind_nfirms():
    ind = pd.read_csv("data/ind30_m_nfirms.csv",
                             header = 0, index_col = 0 , na_values = -99.99)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.index.name = "Period"
    ind.columns = (ind.columns.str.strip())
    
    return ind

def get_total_market_index_returns():
    ind_return = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis = "columns")
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = "rows")
    return (ind_capweight * ind_return).sum(axis = "columns")
    
##################################################################################
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


def sharpe_ratio(r, riskfree_rate,periods_per_year = 12 ):
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


def plot_ef(n_points, er, cov, show_CML = False,style = ".-", riskfree_rate = 0, show_ew = False, show_gmv = False, show_gmv2 = False):

    weights = optimal_weights(n_points, er, cov) # ???
    
    rets = [ portfolio_return(w, er) for w in weights]
    vols = [ portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x = "Volatility", y = "Returns", style = style, figsize = (16,12))
 
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv= portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color = "midnightblue", marker = "o",  markersize = 9)

    if show_gmv2:
        w_gmv2 = gmv2(cov)
        r_gmv2= portfolio_return(w_gmv2, er)
        vol_gmv2 = portfolio_vol(w_gmv2, cov)
        ax.plot([vol_gmv2], [r_gmv2], color = "darkred", marker = "o",  markersize = 9)

    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew= portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color = "goldenrod", marker = "o",  markersize = 12)
        
    if show_CML:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er , cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = "green", marker = "o", linestyle = "dashed", markersize = 12, linewidth = 2)
    return ax


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
    
def gmv(cov, riskfree_rate = 0, value = 1):
    n = cov.shape[0]
    return msr(riskfree_rate, np.repeat(value, n), cov)

def gmv2(cov):
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0, 1),) * n
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
    results = minimize(portfolio_vol,
                       init_guess,
                       args = (cov,),
                       method = "SLSQP",
                       options = { "disp": False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
    )
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
    

########################################################################
def run_ccpi(risky_r, safe_r = None, m = 3, start = 1000, floor = 0.8, riskfree_rate = 0.03, drawdown = None ):
    
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame({"R": risky_r})
        
    account_value_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_weight_history = pd.DataFrame().reindex_like(risky_r)
    

    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = riskfree_rate/12
    
    peak = start
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_weight = m * cushion
        risky_weight = np.minimum(risky_weight, 1)
        risky_weight = np.maximum(risky_weight, 0)
        
        safe_weight = 1 - risky_weight
        
        risky_allocation = account_value * risky_weight
        safe_allocation = account_value * safe_weight
        
        # Update account value for this allocation
        account_value = (risky_allocation * (1 + risky_r.iloc[step]) + 
                         safe_allocation * (1 + safe_r.iloc[step]) )
        cushion_history.iloc[step] = cushion
        risky_weight_history.iloc[step] = risky_weight
        account_value_history.iloc[step] = account_value
        
        
    risky_wealth = start * (1+ risky_r).cumprod()
    ax1 = account_value_history.plot(figsize = (16,8))
    risky_wealth.plot(ax = ax1, style = "--")
    
    if drawdown is None:
        ax1.axhline(y = floor_value)
    
    result = {
        "Wealth": account_value_history,
        "Risky Wealth": risky_wealth,
        "Risky Budget": cushion_history,
        "Risky Allocation": risky_weight_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    
    return result

def summary_stats(r, riskfree_rate = 0.03):
    ann_r = r.aggregate(annualize_ret)
    ann_vol = r.aggregate(annualize_vol)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate = riskfree_rate)
    dd = r.aggregate(lambda r: drawdown(r).DrawDown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified = True)
    hist_cvar5 = r.aggregate(cvar_historic)
    
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": ann_sr,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher Var(5%)": cf_var5,
        "Historic CVar(5%)": hist_cvar5,
        "Max Drawdown": dd
    })
    