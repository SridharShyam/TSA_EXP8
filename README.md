# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
#### Name: SHYAM S
#### Reg.No: 212223240156

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/index_1.csv", parse_dates=['datetime'])

data.head()
data.tail()

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
sales_data = data[['money']]

plt.figure(figsize=(12, 6))
plt.plot(sales_data['money'], label='Original Sales', color='blue')
plt.title('Original Coffee Sales Over Time')
plt.xlabel('Datetime')
plt.ylabel('Amount (₹)')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = sales_data['money'].rolling(window=5).mean()
rolling_mean_10 = sales_data['money'].rolling(window=10).mean()

print("\nFirst 10 values of rolling mean (window=5):")
print(rolling_mean_5.head(10))
print("\nFirst 20 values of rolling mean (window=10):")
print(rolling_mean_10.head(20))

plt.figure(figsize=(12, 6))
plt.plot(sales_data['money'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)', color='orange')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', color='green')
plt.title('Moving Average of Coffee Sales')
plt.xlabel('Datetime')
plt.ylabel('Amount (₹)')
plt.legend()
plt.grid()
plt.show()

data_monthly = sales_data.resample('MS').sum()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

scaled_data = scaled_data + 1e-3

plt.figure(figsize=(12, 6))
plt.plot(scaled_data, color='purple', label='Scaled Monthly Sales')
plt.title('Transformed (Scaled) Monthly Coffee Sales')
plt.xlabel('Month')
plt.ylabel('Normalized Sales')
plt.legend()
plt.grid()
plt.show()

x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=3).fit()

test_predictions = model.forecast(steps=len(test_data))

ax = train_data.plot(label='Train', figsize=(12, 6))
test_data.plot(ax=ax, label='Test')
test_predictions.plot(ax=ax, label='Predictions')
ax.set_title('Exponential Smoothing - Coffee Sales Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Sales')
ax.legend()
plt.grid()
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print("\nRoot Mean Square Error (RMSE):", rmse)

model_full = ExponentialSmoothing(
    scaled_data, trend='add', seasonal='mul', seasonal_periods=3
).fit()

future_steps = int(len(data_monthly) / 4)
predictions = model_full.forecast(steps=future_steps)ax = scaled_data.plot(label='Historical', figsize=(12, 6))
predictions.plot(ax=ax, label='Future Forecast', color='red')
ax.set_title('Coffee Sales Forecast (Next Quarter)')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Sales')
ax.legend()
plt.grid()
plt.show()
```
### OUTPUT:
Moving Average
<img width="996" height="547" alt="image" src="https://github.com/user-attachments/assets/b1d0f4bb-1dc6-41dd-a205-8124fe51d3bf" />

Plot Transform Dataset
<img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/9ea99051-c1b0-4923-97aa-df49ba20aef0" />

Exponential Smoothing
<img width="1014" height="563" alt="image" src="https://github.com/user-attachments/assets/45dc2113-d70b-451b-97d2-35fd8786b840" />

Future Forecast
<img width="1001" height="563" alt="image" src="https://github.com/user-attachments/assets/d31728ef-e24e-4ce2-a0c2-caa34079a802" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
