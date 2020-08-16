import math
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt


from datetime import datetime
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


today = datetime.today().strftime('%Y-%m-%d')
symbols = get_nasdaq_symbols().index

#get data from single stock
df = pdr.DataReader('AAPL', data_source = 'yahoo', start = '1998-01-01', end = today)

#visualize the closing price history
plt.title('Close Price History')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USE ($)',fontsize = 18)
plt.plot(df['Close'])
plt.figure(figsize=(20,10))
plt.show()

#Create new dataframe with only the "Close" column
data = df.filter(['Close'])
#Convert the df to a numpy array
dataset = data.values
#use 80% of the data to train use the other 20% to test
training_data_len = math.floor(len(dataset) * 0.8)

#Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#Create the scaled training dataset
training_data = scaled_data[0:training_data_len]

#Split the data into x and y train datasets
x_train, y_train = [], []

for i in range(0, len(training_data) - 60):
    x_train.append(training_data[i:i+60, 0])
    y_train.append(training_data[i+60, 0])

#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

'''The above steps are all data preparation'''

#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

#train the model
history = model.fit(x_train,y_train,epochs = 20, batch_size=50)

#score = model.evaluate(x_test,y_test, batch_size = 30)

history_dict = history.history

plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.plot(history_dict['mean_absolute_error'])
plt.show()

#creating the test dataset
test_data = scaled_data[training_data_len - 60: , :]

#create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] #actual data

for i in range(0, len(test_data) - 60):
    x_test.append(test_data[i:i+60, 0 ])

#convert the data to a numpy array and make it 3D
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the predicted prices
predictions = model.predict(x_test)

#convert values between 0 and 1 back to normal stock price
predictions = scaler.inverse_transform(predictions)

#get the mean absolute error
mae = np.mean(np.abs(predictions - y_test))


#plotting the data
train = data[:training_data_len]
actual = data[training_data_len:]
actual['Predictions'] = predictions

#visualize data
plt.figure(figsize=(16,8))
plt.title("model")
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price $USD', fontsize = 18)

plt.plot(train['Close'])
plt.plot(actual[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc = 'lower right')
plt.show()

#show the actual and predicted prices
#actual

#get quote
apple_quote = pdr.DataReader('AAPL', data_source= 'yahoo', start = '1998-01-01', end = '2020-05-01')

#create new dataframe
new_df = apple_quote.filter(['Close'])
#get the last 60 day closing price value and convert the df to array
last_60_days = new_df[-60:].values

#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)


#create an empty list
new_x_test = []
#add the last 60 days
new_x_test.append(last_60_days_scaled)
#convert the x_test data to numpy array
new_x_test = np.array(new_x_test)
new_x_test = np.reshape(new_x_test, (new_x_test.shape[0], new_x_test.shape[1], 1))

#get the predicted scaled price
pred_price = model.predict(new_x_test)

#undo scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

