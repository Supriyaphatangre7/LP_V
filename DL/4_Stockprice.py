import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

uploaded = files.upload()

data = pd.read_csv("Google_Stock_Price_Train.csv")

prices = data.iloc[:, 1:2].values

scaler = MinMaxScaler()
prices = scaler.fit_transform(prices)

X = []
y = []

for i in range(60, len(prices)):
    X.append(prices[i-60:i, 0])
    y.append(prices[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential([
    SimpleRNN(50, input_shape=(X.shape[1],1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=5, batch_size=32)

predicted = model.predict(X)

plt.plot(y, color='blue', label='Actual Price')
plt.plot(predicted, color='red', label='Predicted Price')

plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

plt.show()