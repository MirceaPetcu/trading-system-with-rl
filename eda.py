import pandas as pd
import numpy as np
import gym_anytrading
from finta import TA


def get_data():
    df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
    df['TEMA'] = TA.TEMA(df, 12) # Triple Exponential Moving Average --> in ce directie se indreapta pretul pe parcursul a 12 zile cu accent mare pe ultimele zile
    df['ER'] = TA.ER(df) # Kaufman Efficiency Indicator --> confirmare a trendului (-1,1) : -1 downtrend, 1 uptrend, aprox 0 random (trend confirmation)
    df['RSI'] = TA.RSI(df) # Relative Strength Index --> intre 0 si 100, 30-70 intervalul de interes, 30-0 oversold, 70-100 overbought (trend confirmation)
    df['OBV'] = TA.OBV(df) # On Balance Volume --> confirmare a trendului (trend confirmation) --> se bazeaza pe volumul tranzactiilor
    df['STOCH'] = TA.STOCH(df) # Stochastic Oscillator --> confirmare a trendului (trend confirmation) --> intre 0 si 100, 20-80 intervalul de interes, 20-0 oversold, 80-100 overbought
    df.fillna(0, inplace=True)
    
    return df