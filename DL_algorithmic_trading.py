"""
Implementing  paper : Application of Deep Learning to Algorithmic Trading

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pdb

# Building the RNN
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from opts import parse_opts
from sklearn.preprocessing import MinMaxScaler


opt = parse_opts()
"""
If time steps = 10, this means that at each time 't' the RNN is going to look at the prevous 10 stock prices + other features.
Before time 't' that is the stock price is between 10 days before time 't' .
And based on the trends it is capturing during the 10 time steps will try to predict the next
output. So 10 time steps are the past information from which our RNN is going to try to learn and understand
some correlations or some trends.
"""
time_steps = opt.time_steps
epochs= opt.epochs


#Importing the training set
dataset_train= pd.read_csv(opt.input_train)
y_true_train = dataset_train['Adj Close']
# drop column for y_pred

#Importing the dev set
dataset_dev= pd.read_csv( opt.input_validation)
y_true_dev = dataset_dev['Adj Close']
# drop column for y_pred

#Importing the test set
dataset_test= pd.read_csv( opt.input_test)
y_true_test = dataset_test['Adj Close']
# drop column for y_pred

training_set = dataset_train.iloc[:,1:].values
dev_set = dataset_dev.iloc[:,1:].values


#Feature Scaling ( Standardisation vs Normalisation) normalisation for sigmoid
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
y_true_train_scaled = sc.fit_transform(np.array(y_true_train).reshape(-1,1))



dev_set_scaled = sc.fit_transform(dev_set)
y_true_dev_scaled = sc.fit_transform(np.array(y_true_dev).reshape(-1,1))


X_train = []
y_train = []

for i in range(time_steps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-time_steps:i])
    y_train.append(y_true_train_scaled[i])
    
X_train, y_train = np.array(X_train), np.array(y_train)


X_dev = []
y_dev = []

for i in range(time_steps, len(dev_set_scaled)):
    X_dev.append(dev_set_scaled[i-time_steps:i])
    y_dev.append(y_true_dev_scaled[i])
    
X_dev, y_dev = np.array(X_dev), np.array(y_dev)


def build_regressor():
    
    # since predicting a continuous value, dealing with continuous values
    regressor = Sequential() 
    
    #adding first LSTM and bias to avoid overfitting (units = num_neurons)
    regressor.add(CuDNNLSTM(units = 200, return_sequences=True,
                        input_shape = (X_train.shape[1], X_train.shape[2]),
                        bias_regularizer = regularizers.l2(1e-6))) 
    
    #second LSTM layer
    regressor.add(CuDNNLSTM(units = 200, 
                       return_sequences=True, 
                       bias_regularizer=regularizers.l2(1e-6))) 
     
    #third LSTM layer
    regressor.add(CuDNNLSTM(units = 200, 
                       return_sequences=True, 
                       bias_regularizer=regularizers.l2(1e-6))) 
    
    #fourth LSTM layer
    regressor.add(CuDNNLSTM(units = 200, 
                       return_sequences=True, 
                       bias_regularizer=regularizers.l2(1e-6))) 
     
    #fifth LSTM layer
    regressor.add(CuDNNLSTM(units = 200, 
                       bias_regularizer=regularizers.l2(1e-6))) 
    
    #adding the output layer
    regressor.add(Dense(units=1))
    
    return regressor

# save one model
callbacks = [ ModelCheckpoint(filepath='/dbc/output/weights.h5',save_weights_only=True)]

# save model after every 5 epochs
# callbacks = [ ModelCheckpoint(filepath='/dbc/output/weights{epoch:08d}.h5',save_weights_only=True, period=5)]

#compiling the regressor
regressor = build_regressor()  

#compiling the regressor, optimizer adam, for regression, loss fxn = mse
regressor.compile(optimizer='adam', loss = 'mean_squared_error')   

#Fitting RNN
history = regressor.fit(X_train, y_train, epochs = epochs,
              validation_data = (X_dev, y_dev), callbacks=callbacks )


# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'vall_loss'], loc='upper left')
plt.savefig('/dbc/output/history.png')



# Making the predictions and viewing results
#Getting the predicted stock price 
test_set = dataset_test.iloc[:,1:].values
test_set_scaled = sc.fit_transform(test_set)
X_test = []
for i in range(time_steps, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-time_steps:i])
    
X_test = np.array(X_test)  



# The LSTM needs data with the format of [samples, time steps and features]
predicted_stock_price = regressor.predict(X_test)

#predicted_stock_price = sc.inverse_transform(predicted_stock_price)
y_true_test_scaled = sc.fit_transform(np.array(y_true_test).reshape(-1,1))


#viewing results
test_dates = dataset_test['Date'].values
t = [datetime.strptime(tt, '%Y-%m-%d') for tt in test_dates]

fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(t[10:], y_true_test_scaled[10:], color='red', label='Real intel Stock Price')
ax.plot(t[10:], predicted_stock_price ,color='blue', label='Predicted Intel Stock Price')
plt.title('intel Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Intel Stock Price')
plt.legend()
plt.savefig('/dbc/output/stock_price_plot.png')



score = np.sqrt(metrics.mean_squared_error(predicted_stock_price, y_true_test_scaled[10:]))
print('RMSE after {} epochs = '.format(epochs), score)
print('MSE after {} epochs = '.format(epochs), score*score)

