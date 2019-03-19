# Importing the required packages.
import pandas as pd # for file manipulation.
import matplotlib.pyplot as plt # Visualization.
from sklearn.preprocessing import MinMaxScaler
import numpy as np # Scientific Computing.
from keras.models import Sequential # Deep Learning Library.
from keras.layers import Dense, LSTM, Dropout, Activation
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import time

# Reading the data using pandas.
data = pd.read_csv('TSLA.csv')
# Dropping the value of index 0, since LSTM does not consider the index value 0.
data = data.drop(data.index[0])
# Dropping the Date Column, since it is a useless data.
data = data.drop(['Date'], axis=1)
# Splitting the columns for input and Output.
initial = data.iloc[:,1:]
target = data.iloc[:, 0] # Open Column as the Output.
initial = initial.values # Making the attributes(Features) into a numPY array.
initial = initial.astype('float64') # Making the Datatype as Float64.
scale1 = MinMaxScaler(feature_range = (0, 1)) # Scaling the Data using MinMaxScaler.
print(scale1)
initial = scale1.fit_transform(data)
target = target.values # Making the output into a numPY array.
from sklearn.model_selection import train_test_split
# Splitting the data into training and testing data.
X_train, X_test, Y_train, Y_test = train_test_split(initial, target, test_size=0.25, random_state=42)
# Printing the Shape of the Data (Training and Testing Data).
print("Shape of X_Train:", X_train.shape)
print("Shape of X_Test:", X_test.shape)
print("Shape of Y_Train:", Y_train.shape)
print("Shape of Y_Train:", Y_test.shape)
# Since the neural networks takes the 3D array as an input we should change the dimensions of the input 
# by using the np.reshape() function
print("After Reshaping:")
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print("Shape of X_Train:", X_train.shape)
print("Shape of X_Test:", X_test.shape)
print("Shape of Y_Train:", Y_train.shape)
print("Shape of Y_Train:", Y_test.shape)
model = Sequential() # Sequential Model
# We'll be using the LSTM for our analysis, since it is a time-series data.
model.add(LSTM(100, use_bias=True, recurrent_activation='hard_sigmoid',
               return_sequences=True))
model.add(Dropout(0.2)) # Dropout Layer for Regularization.

model.add(LSTM(100, use_bias=True, recurrent_activation='hard_sigmoid',
               return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100, use_bias=True, recurrent_activation='hard_sigmoid',
               return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100, use_bias=True,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('relu')) # Activation function.

start = time.time()

# Compiling the model.
model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['accuracy'])
print('compilation time : ', time.time() - start)
# fitting our data into the model.
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.05)

# Predicting the model, using our test
Predict = model.predict(X_test)
testScore = math.sqrt(mean_absolute_error(Y_test, Predict))
print('Test Score: %.2f Error' % (testScore))
# accu = accuracy_score(Y_test, Predict)
# print("Accuracy: ", accu)
plt.plot(Y_test, color='Cyan')
plt.plot(Predict, color='Green')
plt.legend(['Original value', 'Predicted value'], loc='upper right')
plt.show()