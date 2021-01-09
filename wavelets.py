import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pywt.data

data =pd.read_csv('/home/techsafedata/EDS_1.csv', names = ['timestamp','x','y','z'])
data = data[data['timestamp'].between('2018-07-09 12:00:00','2018-08-09 12:00:00')]
axis_val = data['z'].values
print(len(axis_val))

axis_val =axis_val[~np.isnan(axis_val)]

coef, freq = pywt.cwt(axis_val, scales=1, wavelet = 'mexh')

plt.imshow(coef, extent=[-1, 1, 1, 31], cmap='RdBu_r', aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())

cA, cD = pywt.dwt(axis_val,'db1')
p1, =plt.plot(cA + cD ,'r')
p2, =plt.plot(axis_val, 'b')
plt.xlabel('time')
plt.ylabel('z axis')
plt.legend([p2,p1], ["original signal", "cA+cD"])
plt.show()
plt.show()