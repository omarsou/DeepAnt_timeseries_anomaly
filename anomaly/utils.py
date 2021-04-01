import matplotlib.pyplot as plt
import pandas as pd


def resample_ts(df, freq, method="sum"):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if method == "sum":
        return df.resample(freq, on="timestamp").sum().reset_index()
    else:
        return df.resample(freq, on="timestamp").mean().reset_index()


def label_anomaly(df, anomaly_points):
    df['anomaly'] = 0
    for start, end in anomaly_points:
        df.loc[((df['timestamp'] >= start) & (df['timestamp'] <= end)), 'anomaly'] = 1
    return df


def visualize_anomaly(df, label, title):
    fig, ax = plt.subplots(figsize=(10,10))
    anomaly = df.loc[df['anomaly'] == 1, :]
    ax.plot(df['timestamp'], df['value'], color='blue', label='Normal')
    ax.scatter(anomaly['timestamp'], anomaly['value'], color='red', label='Anomaly')
    plt.xlabel('Date Time')
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.show()