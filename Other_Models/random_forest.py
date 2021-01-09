import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

df = pd.read_csv("EDS_1.csv", nrows=10000000)

def f(row):
    if row['normalized'] < -0.4307273:
        val = -1
    elif row['normalized'] > 0.4307273:
        val = 1
    else:
        val = 0
    return val

df['sax'] = df.apply(f, axis=1)

df = df[df['sax'] != 0]

train_size = int(len(df) * 0.95)
test_size = len(df) - train_size

train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

def create_dataset(X, y):
    Xs, ys = [], []
    time_steps =1
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values 
        Xs.append(v)
        ys.append(y.iloc[i + time_steps]) 
    return np.array(Xs), np.array(ys)

X_train, y_train = create_dataset(train['normalized'], train.sax)
X_test, y_test = create_dataset(test['normalized'], test.sax)

model = RandomForestRegressor(max_depth=4)

model.fit(X_train, y_train)
X_train_pred = model.predict(X_train)
X_test_pred = model.predict(X_test)



train_loss = mean_absolute_error(X_train, X_train_pred)
test_loss = mean_absolute_error(X_test,X_test_pred)

print(train_loss)
print(test_loss)

THRESHOLD = 0.2
TIME_STEPS = 30
test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)

test_score_df['loss'] = test_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold


test_score_df['normalized'] = test[TIME_STEPS:].normalized
test_score_df['sax'] = test[TIME_STEPS:].sax

anomalies = test_score_df[test_score_df.anomaly == True]

plt.plot(
  test[TIME_STEPS:].index,
  test[TIME_STEPS:].normalized,
  label='normalized'
);

sns.scatterplot(
  anomalies.index,
  anomalies.normalized,
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.xticks(rotation=25)
plt.legend();
plt.savefig("random_forest.png")