import pandas as pd
import numpy as np
import gym_anytrading
from finta import TA

import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data




def get_data():
    start_date = "2015-01-01"
    end_date = "2023-12-31"  
    df = get_stock_data("ETH-USD", start_date, end_date)
  
    # Fill NaN values
    df.fillna(0, inplace=True)
    df['TEMA'] = TA.TEMA(df, 30) # Triple Exponential Moving Average --> in ce directie se indreapta pretul pe parcursul a 12 zile cu accent mare pe ultimele zile
    df['ER'] = TA.ER(df) # Kaufman Efficiency Indicator --> confirmare a trendului (-1,1) : -1 downtrend, 1 uptrend, aprox 0 random (trend confirmation)
    df['RSI'] = TA.RSI(df) # Relative Strength Index --> intre 0 si 100, 30-70 intervalul de interes, 30-0 oversold, 70-100 overbought (trend confirmation)
    df['OBV'] = TA.OBV(df) # On Balance Volume --> confirmare a trendului (trend confirmation) --> se bazeaza pe volumul tranzactiilor
    df['STOCH'] = TA.STOCH(df) # Stochastic Oscillator --> confirmare a trendului (trend confirmation) --> intre 0 si 100, 20-80 intervalul de interes, 20-0 oversold, 80-100 overbought
    df.fillna(0, inplace=True)
    
    return df