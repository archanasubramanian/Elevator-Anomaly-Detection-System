import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

#df_trips = pd.read_csv("/scratch/techsafe/data_trips.csv").iloc[0:10000,:]
df_move = pd.read_csv("/scratch/techsafe/anom2.csv").iloc[int(sys.argv[1]):int(sys.argv[2]),:]

#print(df_trips.head())
print(df_move[df_move['sax'] != 0])

df_anom = df_move[df_move['anomaly']==True]

plt.plot(
df_move['ind'], 
df_move['z'],
label='Acceleration',
  
  
);

sns.scatterplot(
  df_anom['ind'],
  df_anom['z'],
  color=sns.color_palette()[3],
  s = 65,
  alpha=0.5,
  label='Anomaly'
)
plt.xticks(rotation=25)
plt.xlabel('Index')
plt.ylabel('Acceleration (g)')
axes = plt.gca()
fig = plt.gcf()
axes.set_ylim([-20,25])
fig.set_size_inches(11,8)
plt.legend();

plt.savefig("part_img.png")
