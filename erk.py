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