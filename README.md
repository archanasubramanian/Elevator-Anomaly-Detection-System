**CMPT 733 Project**

**Elevator Movement Anomaly Detection: Building a System that Works on Many Levels**

By Archana Subramanian, Asha Tambulker, Carmen Riddel, Shreya Shetty

Professional Master of Science in Big Data
Simon Fraser University

**Introduction:**

This project is  anomaly detection in elevators, which uses IOT and sensor data collected from accelerometers in the elevators. 
The results can be useful for technical contractors and building managers to decide on when the elevator would require a maintanance.

**Dataset:**

We received the dataset from Technical Safety BC which was about 75GB, consisting 15 different elevators.The data had four columns: timestamp, x-axis acceleration, y-axis acceleration, z-axis acceleration.

**Requirements:**

*  Spark 2.4+
*  Python 3.5+
*  Pandas
*  Numpy
*  Keras/Tensorflow
*  Scikit-learn
*  Plotly
*  Dash

**Analysis:**

1. Data Cleaning: We used the code normalized.py to pre-process the dataset from 15 different elevators.
2. EDA: We used the file eda.py to create some EDA results
3. Models: We experimented with various models to detect anamolies,refer to codes LSTM.py & files in other_models
4. Visualization:code for visualizations are available within the models.
5. Data product: The code for streaming system is available in stream_demo.py

**Final Result:**

The final product of this project is a streaming dashboard built using plotly dash.

Run 'Stream/stream_demo.py'

Application should run in location provided in Terminal.

or here:http://127.0.0.1:8050/






