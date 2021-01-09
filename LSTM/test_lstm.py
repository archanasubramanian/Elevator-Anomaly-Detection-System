#ADAPTED FROM https://www.curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/#lstm-autoencoders

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from tensorflow.keras.models import load_model
import sys

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

register_matplotlib_converters()

df = pd.read_csv("/scratch/techsafe/train2.csv").iloc[int(sys.argv[1]):int(sys.argv[2]),:]

def f(row):
    if row['z'] < -0.84162123:
        val = -2
    elif row['z'] > -0.84162123 and row['z'] <= -0.2533471:
        val = -1
    elif row['z'] > -0.2533471 and row['z'] <= 0.2533471:
        val = 0
    elif row['z'] > 0.2533471 and row['z'] <= 0.84162123:
        val = 1
    else:
        val = 2
    return val


df['sax'] = df.apply(f, axis=1)

df_move = df[df['sax'] != 0]


test = df_move

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 5

# reshape to [samples, time_steps, n_features]

X_test, y_test = create_dataset(test['sax'], test.sax, TIME_STEPS)

print(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))

model = load_model("my_model_c.h5")
print("Loaded model from disk")

THRESHOLD = 1.1

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

test_score_df = pd.DataFrame()
test_score_df['ind'] = test[TIME_STEPS:].ind
test_score_df['elevator_id'] = test[TIME_STEPS:].elevator_id
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold


test_join = pd.merge(df, test_score_df, how="left", on=['ind', 'elevator_id'])
#Change NaN to False

test_join = test_join.sort_values('ind')

anomalies = test_join[test_join.anomaly == True]

test_join.to_csv("/scratch/techsafe/anom2.csv")