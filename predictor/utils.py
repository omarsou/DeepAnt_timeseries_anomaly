import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


"""VISUALIZATION"""


def plot_pred(true, pred, j, intervalle):
    '''
    plot the prediction of pred for the j_th dimension and for the intervalle.
    args :
        true : real predictions
        pred : predictions of the neural network
        j : number of the time serie you want to visualize
        intervalle : the intervalle you want to see. For example np.arange(0,1000): will plot the first 1000 predictions
    returns : None
    '''
    fig = plt.figure(figsize = (15,8))
    plt.plot(intervalle, pred[intervalle,j], label = 'Predicted')
    plt.plot(intervalle, true[intervalle,j], label = 'Ground Truth')
    MSE = np.sqrt(np.mean((pred[intervalle,j]-true[intervalle,j])**2))
    plt.title(f'RMSE : {round(MSE, 4)}')
    plt.legend()
    plt.show()


"""Dataset preprocessing"""


def slicing_data(dataset, len_window):
    N, D = dataset.shape[0], dataset.shape[1]
    if isinstance(dataset, pd.DataFrame):
        data = dataset.numpy()
    else:
        data = dataset.copy()
    X = np.zeros((N - len_window, len_window, D))
    y = data[len_window:, :]
    for i in range(N - len_window):
        X[i] = data[i: i + len_window, :]
    return X, y


def make_dataset(data, cols, len_window, ratio_train=0.4, scale=True):
    df = data.copy()
    df = df[cols]
    df = df.fillna(df.mean())  # Replace NaN values by the mean
    if scale:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    else:
        df = df.to_numpy()

    length_train = int(df.shape[0] * ratio_train)

    train_set = df[0:length_train]
    test_set = df[length_train:]
    X_train, y_train = slicing_data(train_set, len_window)
    X_test, y_test = slicing_data(test_set, len_window)

    return X_train, y_train, X_test, y_test, train_set, test_set