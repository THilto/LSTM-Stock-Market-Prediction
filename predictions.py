import sys
sys.path.append('../../')

from datetime import datetime
import pandas as pd
import pandas_datareader as web
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import adam_v2

import matplotlib.pyplot as plt

# Dates + ticker values
ticker = 'GME'
start_date = '2021-02-08'
end_date = datetime.today().strftime('%Y-%m-%d')

# Read Data 
df = web.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date)

# Convert data 
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) *.5)

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Training dataset
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

test_data_days = 60

# Inserts values into array increasing one at a time
for i in range(test_data_days, len(train_data)):
    x_train.append(train_data[i-test_data_days:i, 0])
    y_train.append(train_data[i, 0])

# Convert array and reshape to 3d for LSTM
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile + Train Model
model.compile(optimizer=adam_v2.Adam(learning_rate=0.0001), loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=30)

# Create testing data
test_data = scaled_data[training_data_len - test_data_days: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(test_data_days, len(test_data)):
    x_test.append(test_data[i - test_data_days:i, 0])

# Convert + Reshape numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Unscale data + Predict price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate model - RMSE
rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# Plot Data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.title("Main Model")
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower left')
plt.show()