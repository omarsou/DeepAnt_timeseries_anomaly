import scipy.stats as stats

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
from fbprophet import Prophet

import numpy as np
import pandas as pd
from .utils import visualize_anomaly


def get_anomaly_hotelling(df):
    hotelling_df = pd.DataFrame()
    hotelling_df['timestamp'] = df['timestamp']
    hotelling_df['value'] = df['value']
    mean = hotelling_df['value'].mean()
    std = hotelling_df['value'].std()
    hotelling_df['anomaly_score'] = [((x - mean)/std) ** 2 for x in hotelling_df['value']]
    hotelling_df['anomaly_threshold'] = stats.chi2.ppf(q=0.95, df=1)
    hotelling_df['anomaly'] = hotelling_df.apply(lambda x : 1 if x['anomaly_score'] > x['anomaly_threshold'] else 0,
                                                  axis=1)
    return hotelling_df


def get_anomaly_OCsvm(df):
    ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
    ocsvm_ret = ocsvm_model.fit_predict(df['value'].values.reshape(-1, 1))
    ocsvm_df = pd.DataFrame()
    ocsvm_df['timestamp'] = df['timestamp']
    ocsvm_df['value'] = df['value']
    ocsvm_df['anomaly'] = [1 if i == -1 else 0 for i in ocsvm_ret]
    return ocsvm_df


def get_anomaly_isolationforest(df):
    iforest_model = IsolationForest(n_estimators=100, contamination=0.1, max_samples=40)
    iforest_ret = iforest_model.fit_predict(df['value'].values.reshape(-1, 1))
    iforest_df = pd.DataFrame()
    iforest_df['timestamp'] = df['timestamp']
    iforest_df['value'] = df['value']
    iforest_df['anomaly'] = [1 if i == -1 else 0 for i in iforest_ret]
    return iforest_df


def get_anomaly_LOF(df):
    lof_model = LocalOutlierFactor(n_neighbors=60, contamination=0.1)
    lof_ret = lof_model.fit_predict(df['value'].values.reshape(-1, 1))
    lof_df = pd.DataFrame()
    lof_df['timestamp'] = df['timestamp']
    lof_df['value'] = df['value']
    lof_df['anomaly'] = [1 if i == -1 else 0 for i in lof_ret]
    return lof_df


def get_anomaly_VARBM(df):
    sigma_df = pd.DataFrame()
    sigma_df['value'] = df['value']
    std = sigma_df['value'].std()
    mean = sigma_df['value'].mean()
    sigma_df['anomaly_threshold_3r'] = mean + 1.5*std
    sigma_df['anomaly_threshold_3l'] = mean - 1.5*std
    sigma_df['anomaly']  = sigma_df.apply(lambda x : 1 if (x['value'] > x['anomaly_threshold_3r']) or
                                                          (x['value'] < x['anomaly_threshold_3l']) else 0, axis=1)
    sigma_df['timestamp'] = df['timestamp']
    return sigma_df


def get_anomaly_prophet(df):
    prophet_df = df.reset_index()[['timestamp', 'value']].rename({'timestamp':'ds', 'value':'y'}, axis='columns')
    model = Prophet(changepoint_range=0.95)
    model.fit(prophet_df)
    forecast = model.predict(prophet_df[['ds']])
    results = pd.concat([prophet_df.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower', 'yhat_upper']]],
                axis=1)
    #Calculating the error in prediction
    results['error'] = results['y'] - results['yhat']
    #Calculating the uncertainity- the region where the predicted values are less likely to fall
    results['uncertainity'] = (results['yhat_upper'] - results['yhat_lower'])/2
    results['anomaly'] = results.apply(lambda x: 1
                               if(np.abs(x['error']) > x['uncertainity']) else 0,axis=1)
    results = results.reset_index(drop=True)
    results['timestamp'] = df['timestamp']
    results['value'] = df['value']
    return results


def get_anomaly_deepant(df, y_pred, y_true):
    results = pd.DataFrame()
    results['timestamp'] = df['timestamp']
    results['value'] = y_true
    results['error'] = abs(y_pred - y_true)
    threshold = np.sort(results['error'].values)[-int(0.1*len(y_true)):][0]
    results['anomaly'] = results.apply(lambda x : 1 if x['error'] >= threshold else 0, axis=1)
    return results


def sum_up_anomaly(pipeline_anomaly, df, data_name="NYC Taxi", model_name="Hotelling T2", y_pred=None, y_true=None):
    if model_name == 'DeepAnt':
        df_predicted_anomaly = pipeline_anomaly(df, y_pred, y_true)
    else:
        df_predicted_anomaly = pipeline_anomaly(df)
    print(f"For {data_name} : F1 Score = {f1_score(df['anomaly'], df_predicted_anomaly['anomaly'])}")
    visualize_anomaly(df_predicted_anomaly, data_name, f"Detected Anomaly with {model_name}")