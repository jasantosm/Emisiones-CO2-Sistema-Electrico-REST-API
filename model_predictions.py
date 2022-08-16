from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import load
import pandas as pd
import numpy as np


def sliding_time(ts, window_size=1):

  n = ts.shape[0] - window_size  
  X = np.empty((n, window_size))
  y = np.empty(n)

  for i in range(window_size, ts.shape[0]):   
    y[i - window_size] = ts[i]
    X[i- window_size, 0:window_size] = np.array(ts[i - window_size:i])
    
  return X, y

def multi_var_slinding_time(df, window_size, objetive_var):
  X_train, y_train = sliding_time(df[objetive_var].values, window_size=window_size)

  for feature in df.columns:
    if feature!=objetive_var:
      X_multi, y_multi = sliding_time(df[feature].values, window_size=window_size)
      X_multi = np.append(X_multi, y_multi.reshape(-1,1), axis=1)
      X_train = np.append(X_train, X_multi, axis=1)

  return X_train, y_train

def read_data():
    return pd.read_csv('./data/test_data.csv').set_index('Date')

def read_test_data():
    k = 20
    df = read_data()
    X_test, y_test = multi_var_slinding_time(df, k,'daily_emision_CO2_eq')
    return df, y_test.tolist(), k
    

def predict():
    k = 20
    df = read_data()
    X_test, y_test = multi_var_slinding_time(df, k,'daily_emision_CO2_eq')
    model = load('./model/perceptron.joblib') 
    y_pred = model.predict(X_test)
    x = df.reset_index()['Date'].to_list()
    return y_pred.tolist(), x[k:]