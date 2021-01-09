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
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

df = pd.read_csv("/scratch/techsafe/train2.csv")


#Cut values based on an alphabet of 3
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

df = df[df['sax'] != 0]


train_size = int(len(df) * 0.95)
test_size = len(df) - train_size

train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

print(train.shape, test.shape)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values #v is the size of a time step
        Xs.append(v)
        ys.append(y.iloc[i + time_steps]) #y is just one dimension
    return np.array(Xs), np.array(ys)

TIME_STEPS = 5

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train['z'], train.sax, TIME_STEPS)
X_test, y_test = create_dataset(test['z'], test.sax, TIME_STEPS)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))

print(X_train.shape)

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=100,
    activation='relu',
    input_shape=(X_train.shape[1], 1)
))


model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=100, return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1)))
model.compile(loss='mae', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

X_train_pred = model.predict(X_train)

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)


THRESHOLD = 1.1

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold

test_score_df['z'] = test[TIME_STEPS:].sax

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()

print(anomalies)


model.save('my_model_c.h5')
print("Model saved to disk")
# later...

# load json and create model
model = load_model("my_model.h5")
print("Loaded model from disk")

THRESHOLD = 1.1

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['date'] = test[TIME_STEPS:].date
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold


test_score_df['z'] = test[TIME_STEPS:].z
test_score_df['sax'] = test[TIME_STEPS:].sax

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()

print(test_score_df)
