import sys
from pyspark.sql import types
from pyspark.sql import SparkSession, functions, types
import pandas as pd
import numpy as np
from scipy.signal import butter,filtfilt
spark = SparkSession.builder.appName('example code').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

nyq = 0.5 * fs

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#the program execution starts here
def main(inputs,output):
    X = ['EDS_3','EDS_5','EDS_8','EDS_11','EDS_14']
    Y = 'EDS_12'
    Z = ['EDS_1','EDS_2','EDS_4','EDS_6','EDS_7','EDS_9','EDS_10','EDS_13','EDS_15']
    neg = ['4','6','8','9','10','12','13','14']
    for i in X:
        if i in inputs:
            id1 = i[-1]
            filename = pd.read_csv(inputs,names =['timestamp','primary_axis','y','z'])
            filename['elevator_id']=id1
            cols = ["elevator_id","timestamp","primary_axis","y","z"]
            filename = filename.reindex(columns = cols)

    if Y in inputs:
        id1 = Y[-1]
        filename = pd.read_csv(inputs,names =['timestamp','x','primary_axis','Z'])
        filename['elevator_id']=id1
        cols = ["elevator_id","timestamp","x","primary_axis","z"]
        filename = filename.reindex(columns = cols)

    else:
        for j in Z:
            if j in inputs:
                id1 = j[-1]
                filename = pd.read_csv(inputs,names =['timestamp','x','y','primary_axis'])
                filename['elevator_id']= id1
                cols = ["elevator_id","timestamp","x","y","primary_axis"]
                filename = filename.reindex(columns = cols)
                
#convert timestamp to datetime datatype
    filename['timestamp'] = pd.to_datetime(filename['timestamp'],format='')
#filtering data between the given date range
    filename = filename[filename['timestamp'].between('2018-07-09 12:00:00','2018-08-09 12:00:00')]
    
#flip sign for negative acceleration data
    if id1 in neg:
        filename['normalized'] = filename['primary_axis']*(-1)
    else:
        filename['normalized'] = filename['primary_axis']
        
    fs = 25
    cutoff = 5
    order = 2

    data = filename['normalized'].to_numpy()  
    filt = butter_lowpass_filter(data, cutoff, fs, order)
    filename['normalized'] = filt

#normalizing primary axis data
    mean = filename['normalized'].mean()
    std = filename['normalized'].std()
    filename['normalized'] = (filename['normalized']-mean)/std
    
#dropping additional unnamed column
    filename.drop(filename.filter(regex=' '),axis =1,inplace = True)
    print(filename.columns)
    
#write output to csv file
    filename.to_csv(output, index = False)

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs,output)


