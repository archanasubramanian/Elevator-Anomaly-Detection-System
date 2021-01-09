import adtk
from adtk import data
import pandas as pd
import sys
from adtk.visualization import plot
import matplotlib.pyplot as plt


df = pd.read_csv("/scratch/techsafe/train.csv").iloc[int(sys.argv[1]):int(sys.argv[2])]
#print(df.head())
df['date'] = pd.DatetimeIndex(df['date'])

from scipy.signal import butter,filtfilt# Filter requirements.

fs = 25
cutoff = 2
nyq = 0.5 * fs
order = 2


datax = df['x'].to_numpy()
datay = df['y'].to_numpy()

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

filtx = butter_lowpass_filter(datax, cutoff, fs, order)
filty = butter_lowpass_filter(datay, cutoff, fs, order)

df['x'] = filtx
df['y'] = filty

mean = df['x'].mean()
sd = df['x'].std()

df['x'] = df['x']-mean
df['x'] = df['x']/sd

mean = df['y'].mean()
sd = df['y'].std()

df['y'] = df['y']-mean
df['y'] = df['y']/sd

df_x = df['x']
df_y = df['y']


s = data.validate_series(df_x)
s2 = data.validate_series(df_y)

from adtk.detector import GeneralizedESDTestAD
esd_ad = GeneralizedESDTestAD(alpha=0.1)

anomalies_x = esd_ad.fit_detect(s)
anomalies_y = esd_ad.fit_detect(s2)


#join based on datetime


plot(s, anomaly=anomalies_x, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

plt.savefig("xx.png")

anomalies_x = anomalies_x.reset_index()
anomalies_y = anomalies_y.reset_index()

df = pd.merge(df, anomalies_x, on='date').rename(columns={"x_x": "x", "x_y": "anomaly_x"})
df = pd.merge(df, anomalies_y, on='date').rename(columns={"y_x": "y", "y_y": "anomaly_y"})


print(df)
