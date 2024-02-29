import ta as ta1
import numpy as np
import pandas as pd
import pandas_ta as ta
import quantstats as qs
from sklearn.exceptions import DataConversionWarning

## Binance ##

API_KEY    = "AcMlboNYSvT4WOGl4xgesFxdpMzuTAfcdznMbdVbH86yiOfzFSaQkR9849zKuCZi"
API_SECRET = "svmsffys0iHndcayEaNHbaj0S30i2oFVYphmaIkO1xp4SzurCfekUoQbRs90Dc49"

import os
qs.extend_pandas()

from sklearn.decomposition import TruncatedSVD
from feature_engine.outliers import OutlierTrimmer
from feature_engine.datetime import DatetimeFeatures

from scipy.stats import iqr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import (SmartCorrelatedSelection, DropHighPSIFeatures)

import warnings
warnings.filterwarnings("ignore")


def fix_dataset_inconsistencies(dataframe, fill_value=None):

    dataframe = dataframe.replace([-np.inf, np.inf], np.nan)

    # This is done to avoid filling middle holes with backfilling.
    if fill_value is None:
        #dataframe.iloc[0,:] = dataframe.apply(lambda column: column.iloc[column.first_valid_index()], axis='index')
        dataframe.bfill().ffill()
    else:
        dataframe.iloc[0,:] = dataframe.iloc[0,:].fillna(fill_value)

    return dataframe.fillna(axis='index', method='pad').dropna(axis='columns')

def precalculate_ground_truths(data, column='close', threshold=None):
    tool = Tools()
    returns = tool.get_returns(data, column=column)
    gains = tool.estimate_outliers(returns) if threshold is None else threshold
    binary_gains = (returns[column] > gains).astype(int)
    
    return binary_gains

class Tools:
    def __init__(self):
        pass

    def estimate_outliers(self, data):
        return iqr(data) * 1.5

    def estimate_percent_gains(self, data, column='close'):
        returns = self.get_returns(data, column=column)
        gains = self.estimate_outliers(returns)
        return gains

    def get_returns(self, data, column='close'):
        return fix_dataset_inconsistencies(data[[column]].pct_change(), fill_value=0)

    def is_null(self, data):
        return data.isnull().sum().sum() > 0

    def is_sparse(self, data, column='close', n_bins=5):
        binary_gains = precalculate_ground_truths(data, column=column)
        bins = [n * (binary_gains.shape[0] // n_bins) for n in range(n_bins)]
        bins += [binary_gains.shape[0]]
        bins = [binary_gains.iloc[bins[n]:bins[n + 1]] for n in range(n_bins)]
        return all([bin.astype(bool).any() for bin in bins])

    def is_data_predictible(self, data, column):
        return not self.is_null(data) & self.is_sparse(data, column)

class TechnicalIndicator:
    def __init__(self):
        pass

    def rsi(self, price: 'pd.Series[pd.Float64Dtype]', period: float) -> 'pd.Series[pd.Float64Dtype]':

        r = price.diff()
        upside = np.minimum(r, 0).abs()
        downside = np.maximum(r, 0).abs()
        rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
        return 100*(1 - (1 + rs) ** -1)

    def macd(self, price: 'pd.Series[pd.Float64Dtype]', fast: float, slow: float, signal: float) -> 'pd.Series[pd.Float64Dtype]':
        fm = price.ewm(span=fast, adjust=False).mean()
        sm = price.ewm(span=slow, adjust=False).mean()
        md = fm - sm
        signal = md - md.ewm(span=signal, adjust=False).mean()
        return signal

    def generate_all_default_quantstats_features(self, data):
        excluded_indicators = [
            'compare',
            'greeks',
            'information_ratio',
            'omega',
            'r2',
            'r_squared',
            'rolling_greeks',
            'warn',
        ]
        
        indicators_list = [f for f in dir(qs.stats) if f[0] != '_' and f not in excluded_indicators]
        
        df = data.copy()
        df = df.set_index('date')
        df.index = pd.DatetimeIndex(df.index)

        for indicator_name in indicators_list:
            try:
                #print(indicator_name)
                indicator = qs.stats.__dict__[indicator_name](df['close'])
                if isinstance(indicator, pd.Series):
                    indicator = indicator.to_frame(name=indicator_name)
                    df = pd.concat([df, indicator], axis='columns')
            except (pd.errors.InvalidIndexError, ValueError):
                pass

        df = df.reset_index()
        return df

    def generate_features(self, data):
        # Automatically-generated using pandas_ta
        df = data.copy()

        strategies = ['candles', 
                    'cycles', 
                    'momentum', 
                    'overlap', 
                    'performance', 
                    'statistics', 
                    'trend', 
                    'volatility', 
                    'volume']

        df.index = pd.DatetimeIndex(df.index)

        cores = os.cpu_count()
        df.ta.cores = cores

        for strategy in strategies:
            df.ta.study(strategy, exclude=['kvo'])

        df = df.set_index('date')

        # Generate all default indicators from ta library
        ta1.add_all_ta_features(data, 
                                'open', 
                                'high', 
                                'low', 
                                'close', 
                                'volume', 
                                fillna=True)

        # Naming convention across most technical indicator libraries
        data = data.rename(columns={'open': 'Open', 
                                    'high': 'High', 
                                    'low': 'Low', 
                                    'close': 'Close', 
                                    'volume': 'Volume'})
        data = data.set_index('date')

        # Custom indicators
        features = pd.DataFrame.from_dict({
            'prev_open': data['Open'].shift(1),
            'prev_high': data['High'].shift(1),
            'prev_low': data['Low'].shift(1),
            'prev_close': data['Close'].shift(1),
            'prev_volume': data['Volume'].shift(1),
            'vol_5': data['Close'].rolling(window=5).std().abs(),
            'vol_10': data['Close'].rolling(window=10).std().abs(),
            'vol_20': data['Close'].rolling(window=20).std().abs(),
            'vol_30': data['Close'].rolling(window=30).std().abs(),
            'vol_50': data['Close'].rolling(window=50).std().abs(),
            'vol_60': data['Close'].rolling(window=60).std().abs(),
            'vol_100': data['Close'].rolling(window=100).std().abs(),
            'vol_200': data['Close'].rolling(window=200).std().abs(),
            'ma_5': data['Close'].rolling(window=5).mean(),
            'ma_10': data['Close'].rolling(window=10).mean(),
            'ma_20': data['Close'].rolling(window=20).mean(),
            'ma_30': data['Close'].rolling(window=30).mean(),
            'ma_50': data['Close'].rolling(window=50).mean(),
            'ma_60': data['Close'].rolling(window=60).mean(),
            'ma_100': data['Close'].rolling(window=100).mean(),
            'ma_200': data['Close'].rolling(window=200).mean(),
            'ema_5': ta1.trend.ema_indicator(data['Close'], window=5, fillna=True),
            'ema_10': ta1.trend.ema_indicator(data['Close'], window=10, fillna=True),
            'ema_20': ta1.trend.ema_indicator(data['Close'], window=20, fillna=True),
            'ema_60': ta1.trend.ema_indicator(data['Close'], window=60, fillna=True),
            'ema_64': ta1.trend.ema_indicator(data['Close'], window=64, fillna=True),
            'ema_120': ta1.trend.ema_indicator(data['Close'], window=120, fillna=True),
            'lr_open': np.log(data['Open']).diff().fillna(0),
            'lr_high': np.log(data['High']).diff().fillna(0),
            'lr_low': np.log(data['Low']).diff().fillna(0),
            'lr_close': np.log(data['Close']).diff().fillna(0),
            'r_volume': data['Close'].diff().fillna(0),
            'rsi_5': self.rsi(data['Close'], period=5),
            'rsi_10': self.rsi(data['Close'], period=10),
            'rsi_100': self.rsi(data['Close'], period=100),
            'rsi_7': self.rsi(data['Close'], period=7),
            'rsi_28': self.rsi(data['Close'], period=28),
            'rsi_6': self.rsi(data['Close'], period=6),
            'rsi_14': self.rsi(data['Close'], period=14),
            'rsi_26': self.rsi(data['Close'], period=24),
            'macd_normal': self.macd(data['Close'], fast=12, slow=26, signal=9),
            'macd_short': self.macd(data['Close'], fast=10, slow=50, signal=5),
            'macd_long': self.macd(data['Close'], fast=200, slow=100, signal=50),
        })

        # Concatenate both manually and automatically generated features
        data = pd.concat([data, features], axis='columns').fillna(method='pad')

        # Remove potential column duplicates
        data = data.loc[:,~data.columns.duplicated()]

        # Revert naming convention
        data = data.rename(columns={'Open': 'open', 
                                    'High': 'high', 
                                    'Low': 'low', 
                                    'Close': 'close', 
                                    'Volume': 'volume'})

        # Concatenate both manually and automatically generated features
        data = pd.concat([data, df], axis='columns').fillna(method='pad')

        # Remove potential column duplicates
        data = data.loc[:,~data.columns.duplicated()]

        data = data.reset_index()

        # Generate all default quantstats features
        df_quantstats = self.generate_all_default_quantstats_features(data)

        # Concatenate both manually and automatically generated features
        data = pd.concat([data, df_quantstats], axis='columns').fillna(method='pad')

        # Remove potential column duplicates
        data = data.loc[:,~data.columns.duplicated()]
        data = self.DatetimeFeatures(data)

        # A lot of indicators generate NaNs at the beginning of DataFrames, so remove them
        data = data.iloc[200:]
        data = data.reset_index(drop=True)
        data = fix_dataset_inconsistencies(data, fill_value=None)

       
        return data

    def DatetimeFeatures(self, data): # gets new good features


        cyclical = DatetimeFeatures(variables=None, features_to_extract=None,
                                    drop_original=False, missing_values='raise', 
                                    dayfirst=False, yearfirst=False, utc=None)

        data = cyclical.fit_transform(data)
        return data



class EnginnerFeatures:
    def __init__(self):
        pass

    def VarianceThresholdFeatureSelection(self, data, threshold=.8):

        sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
        date = data[['date']].copy()
        data = data.drop(columns=['date'])
        sel.fit(data)

        #Get a mask, or integer index, of the features selected.
        data[data.columns[sel.get_support(indices=True)]]
        data = pd.concat([date, data], axis='columns')

        return data

    def SmartCorrelatedSelection(self, data):
        # set up the selector
        sel = SmartCorrelatedSelection(
            variables=None, method="pearson",
            threshold=0.8, missing_values="raise",
            selection_method="variance", estimator=None,
        )

        data = sel.fit_transform(data)
        features_to_drop = sel.features_to_drop_
        to_drop = list(set(features_to_drop) - set(['open', 'high', 'low', 'close', 'volume']))

        return features_to_drop, to_drop

    def DropHighPSIFeatures(self, data):
        tr = DropHighPSIFeatures(split_col=None, split_frac=0.5, split_distinct=False,
                                 cut_off=None, switch=False, threshold=0.20, bins=10, strategy='equal_frequency',
                                  min_pct_empty_bins=0.0001, missing_values='raise', variables=None)

        data = tr.fit_transform(data)
        features_to_drop = tr.features_to_drop_
        to_drop = list(set(features_to_drop) - set(['open', 'high', 'low', 'close', 'volume']))
        
        return features_to_drop, to_drop 


    def FeatureSelection(self, data):

        # Variance Threshold
        data = self.VarianceThresholdFeatureSelection(data, .8)

        # Smart Correlated Features
        features_to_drop, to_drop  = self.SmartCorrelatedSelection(data)
        data = data.drop(columns=to_drop)

        # High PSI Remover
        features_to_drop, to_drop  = self.DropHighPSIFeatures(data)
        data = data.drop(columns=to_drop)

        return data

class Normalizer:
    def __init__(self):
        pass

    scaler_type = MinMaxScaler

    def get_feature_scalers(self, X, scaler_type=scaler_type):
        scalers = []
        for name in list(X.columns[(X.columns != 'date')]):
            scalers.append(scaler_type().fit(X[name].values.reshape(-1, 1)))
        return scalers

    def get_scaler_transforms(self, X, scalers):
        X_scaled = []
        for name, scaler in zip(list(X.columns[(X.columns != 'date')]), scalers):
            X_scaled.append(scaler.transform(X[name].values.reshape(-1, 1)))
        X_scaled = pd.concat([pd.DataFrame(column, columns=[name]) for name, column in \
                            zip(list(X.columns[(X.columns != 'date')]), X_scaled)], axis='columns')
        return X_scaled

    def normalize_data(self , X_train,scaler_type=scaler_type):

        X_train_test_dates = X_train[['date']]
        X_train_test = X_train.drop(columns=['date'])

        train_test_scalers  = self.get_feature_scalers(X_train_test, scaler_type=scaler_type)
        X_train_test_scaled = self.get_scaler_transforms(X_train_test, train_test_scalers)


        X_train_test_dates  = X_train_test_dates.reset_index(drop=True)
        X_train_test_scaled = X_train_test_scaled.reset_index(drop=True)
        X_train_test_scaled = pd.concat([X_train_test_dates, X_train_test_scaled], axis='columns')

        X_train_scaled = X_train_test_scaled.iloc[:X_train.shape[0]]

        # remove outlier
        X_train_scaled = self.OutlierTrimmer(X_train_scaled)
        return (X_train_scaled)

    def OutlierTrimmer(self, data):
        tr = OutlierTrimmer(capping_method='gaussian', tail='right', fold=3, variables=None, missing_values='raise')
        data  = tr.fit_transform(data)
        return data

class Split:
    def __init__(self):
        pass

    def pre_split_data(self, data): # change splitting size
    
        X = data.copy()
        y = X['close'].pct_change()

        X_train_test, X_valid, y_train_test, y_valid = \
            train_test_split(X, y, train_size=0.67, test_size=0.33, shuffle=False)

        X_train, X_test, y_train, y_test = \
            train_test_split(X_train_test, y_train_test, train_size=0.50, test_size=0.50, shuffle=False)

        return X_train, X_test, X_valid, y_train, y_test, y_valid

    def split_data(self, data, train_path, test_path, valid_path):

        # y data is ignored
        X_train, X_test, X_valid, _, _, _ = self.pre_split_data(data)

        cwd = os.getcwd()
        train_csv = os.path.join(cwd, train_path)
        test_csv = os.path.join(cwd, test_path)
        valid_csv = os.path.join(cwd, valid_path)
        X_train.to_csv(train_csv, index=False)
        X_test.to_csv(test_csv, index=False)
        X_valid.to_csv(valid_csv, index=False)

        return X_train, X_test, X_valid



## Solo Fuctions ##


def prepare_data(df):
    df['volume'] = np.int64(df['volume'])
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')
    return df


def load_csv(filename):

    df = pd.read_csv(filename)
    df.drop(columns=['tic'], inplace=True)

    return prepare_data(df)

# Feature Pipline 
def pipeline(data, path):
    lib1 = TechnicalIndicator()
    data = lib1.generate_features(data)
    print('Done creating all alpha factors!')
    lib2 = EnginnerFeatures()
    data  = lib2.FeatureSelection(data)
    print('Done extracting useful alphas!')
    data.to_csv(path, index=False)
    return data