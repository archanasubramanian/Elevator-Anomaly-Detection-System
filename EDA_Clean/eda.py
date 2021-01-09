import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

df = pd.read_csv("/scratch/techsafe/bigdata/EDS_"+ sys.argv[1] +".csv")


df_move = df[(df['normalized'] > 0.4307273) | (df['normalized'] < -0.4307273)]


print("MAX = " + str(df['normalized'].max()))
print("MIN = " + str(df['normalized'].min()))
print("% = " + str(len(df_move)/len(df)))