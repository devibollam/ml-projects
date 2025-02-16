import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset (skip first two rows & set correct column names)
file_path = r"TataMotors.csv"  # Change this to your actual file path
# Load CSV, skip first two rows, and correctly assign column names
df = pd.read_csv(file_path, skiprows=2)

# Drop the first empty column (index 0)
df = df.iloc[:, 1:]  # Keep only relevant columns

# Rename columns properly
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Convert numeric columns properly
df[["Close", "High", "Low", "Open", "Volume"]] = df[["Close", "High", "Low", "Open", "Volume"]].apply(pd.to_numeric, errors="coerce")

# Sort by date
df = df.sort_values(by="Date")


# Select Close Price for prediction
close_prices = df["Close"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Create dataset with time steps
def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Set time step (past 60 days to predict next day)
time_step = 60
X, y = create_sequences(close_prices_scaled, time_step)

# Reshape input for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build Stacked LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predict on test data
predicted_prices = model.predict(X_test)

# Convert back to original scale
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(df["Date"][-len(y_test):], y_test_actual, label="Actual Prices", color="blue")
plt.plot(df["Date"][-len(y_test):], predicted_prices, label="Predicted Prices", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction using Stacked LSTM")
plt.legend()
plt.show()

# Predict next 30 days
future_input = close_prices_scaled[-time_step:].reshape(1, time_step, 1)
future_predictions = []
for _ in range(30):
    future_price = model.predict(future_input)[0, 0]
    future_predictions.append(future_price)
    future_input = np.roll(future_input, -1)
    future_input[0, -1, 0] = future_price

# Convert back to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future dates
last_date = df["Date"].iloc[-1]
future_dates = pd.date_range(last_date, periods=30, freq="B")  # Business days only

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_predictions, marker="o", linestyle="-", color="green", label="Future Predictions")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Next 30 Days Stock Price Prediction")
plt.legend()
plt.show()
