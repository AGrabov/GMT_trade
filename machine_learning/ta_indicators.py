import talib
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TAIndicators:
    def __init__(self, data=None) -> None:
        self.data = data        

    def _calculate_indicators(self) -> pd.DataFrame:
        """Calculate indicators"""

        df = self.data
        # Ensure df is not None or empty
        if df is None or df.empty:
            raise ValueError("Invalid dataframe provided for indicator calculation.")
        
        # Store the initial data
        self.initial_data = df.copy()

        df['close_change_pct'] = df['close'].pct_change()
    
        # Adding lag features for 'close' column
        for lag in range(1, 6):  # Adding 5 lag features
            df[f'lag_{lag}'] = df['close'].shift(lag)

        
        # Rolling statistics
        df['rolling_mean'] = df['close'].rolling(window=20).mean()
        df['rolling_std'] = df['close'].rolling(window=20).std()
        df['rolling_min'] = df['close'].rolling(window=20).min()
        df['rolling_max'] = df['close'].rolling(window=20).max()
        df['rolling_skew'] = df['close'].rolling(window=20).skew()
        df['rolling_kurt'] = df['close'].rolling(window=20).kurt()
        df['rolling_median'] = df['close'].rolling(window=20).median()
        df['rolling_25_quantile'] = df['close'].rolling(window=20).quantile(0.25)
        df['rolling_75_quantile'] = df['close'].rolling(window=20).quantile(0.75)
        df['rolling_variance'] = df['close'].rolling(window=20).var()
    
        # Calculate Heikin Ashi candlesticks
        df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
        df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)

        ha_diff = df['HA_Close'] - df['HA_Open']
        df['HA_Diff'] = talib.MA(ha_diff, timeperiod=10, matype=2)

        # Calculate Price Trend
        df['AVGPRICE'] = talib.AVGPRICE(df['open'], df['high'], df['low'], df['close'])
        df['MEDPRICE'] = talib.MEDPRICE(df['high'], df['low'])
        df['TYPPRICE'] = talib.TYPPRICE(df['high'], df['low'], df['close'])
        df['WCLPRICE'] = talib.WCLPRICE(df['high'], df['low'], df['close'])

        # Calculate Moving Average
        df['MA'] = talib.MA(df['close'], timeperiod=10, matype=0)            

        # Calculate Moving Average Convergence Divergence
        df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACDEXT(df['close'], 
                                                                     fastperiod=12, fastmatype=2,
                                                                     slowperiod=26, slowmatype=1, 
                                                                     signalperiod=9, signalmatype=8)

        # Calculate Bollinger Bands
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df['close'], 
                                                                       timeperiod=20, nbdevup=2, 
                                                                       nbdevdn=2, matype=0)
        
        # Calculate Moving Average with variable period
        df['DEMA'] = talib.DEMA(df['close'], timeperiod=25)

        # Calculate MESA Adaptive Moving Average         
        df['MAMA'], df['FAMA'] = talib.MAMA(df['close'], fastlimit=0.5, slowlimit=0.05)

        # Calculate RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)

        # Calculate Commodity Channel Index
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calculate Absolute Price Oscillator
        df['APO'] = talib.APO(df['close'], fastperiod=3, slowperiod=13, matype=2)

        # Calculate Chande Momentum Oscillator
        df['CMO'] = talib.CMO(df['close'], timeperiod=14)

        # Calculate Chaikin A/D Oscillator
        df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

        # Calculate Normalized Average True Range
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=13)

        df['ST_RSI_K'], df['ST_RSI_D'] = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=1)

        df['STOCH_K'], df['STOCH_D'] = talib.STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=1)

        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

        df['BETA'] = talib.BETA(df['high'], df['low'], timeperiod=5)
        df['CORREL'] = talib.CORREL(df['high'], df['low'], timeperiod=30)
        df['LINEARREG'] = talib.LINEARREG(df['close'], timeperiod=14)
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['close'], timeperiod=14)
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['close'], timeperiod=14)
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)
        df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=5, nbdev=1)
        df['TSF'] = talib.TSF(df['close'], timeperiod=14)
        df['VAR'] = talib.VAR(df['close'], timeperiod=5, nbdev=1)

        # Calculate Hilbert Transform
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close']) # Instantaneous Trendline        
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['close']) # Dominant Cycle Period        
        df['HT_DCPHASE'] = talib.HT_DCPHASE(df['close']) # Dominant Cycle Phase        
        df['HT_INPHASE'], df['HT_QUADRATURE'] = talib.HT_PHASOR(df['close']) # Dominant Cycle Inphase        
        df['HT_SINE'], df['HT_LEAD_SINE'] = talib.HT_SINE(df['close']) # Dominant Cycle Sine        
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['close']) # Trend Mode

        self._handle_nan_values(df)

        return df
    
    def _handle_nan_values(self, df=None):
        """Handle NaN values in the dataframe."""
        if df is None:
            df = self.df

        if df.isna().any().any():
            # print("Warning: NaN values detected in the normalized data!\n Filling NaN values with various methods.")
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)        