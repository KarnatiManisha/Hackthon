# # Setting Up the Environment
pip install pandas matplotlib seaborn numpy
# # Simulating Data Collection
import pandas as pd
import numpy as np
import datetime

# Simulate data collection
def generate_sensor_data(num_entries=100):
    timestamps = [datetime.datetime.now() - datetime.timedelta(minutes=i) for i in range(num_entries)]
    flow = np.random.uniform(0, 100, num_entries)  # Flow in liters per minute
    pressure = np.random.uniform(1, 10, num_entries)  # Pressure in bars
    quality = np.random.uniform(0, 1, num_entries)  # Water quality index (0-1)
    consumption = np.random.uniform(0, 150, num_entries)  # Consumption in liters

    data = pd.DataFrame({
        'Timestamp': timestamps,
        'Flow': flow,
        'Pressure': pressure,
        'Quality': quality,
        'Consumption': consumption
    })
    return data

# Generate and display data
sensor_data = generate_sensor_data()
print(sensor_data.head())
# # Basic Data Analysis
# Check for anomalies (e.g., sudden drop in pressure)
def detect_anomalies(data):
    anomalies = data[data['Pressure'] < 2]  # Example threshold
    return anomalies
anomalies = detect_anomalies(sensor_data)
print("Detected anomalies:")
print(anomalies)
# # Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Set plot style
sns.set(style="whitegrid")
# Create interactive dashboards
def plot_metrics(data):
    plt.figure(figsize=(14, 10))
    # Subplot for Flow and Consumption
    plt.subplot(2, 1, 1)
    plt.plot(data['Timestamp'], data['Flow'], label='Flow (L/min)', color='blue')
    plt.plot(data['Timestamp'], data['Consumption'], label='Consumption (L)', color='orange')
    plt.title('Flow and Consumption Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.legend()
    # Subplot for Pressure and Quality
    plt.subplot(2, 1, 2)
    plt.plot(data['Timestamp'], data['Pressure'], label='Pressure (bars)', color='green')
    plt.plot(data['Timestamp'], data['Quality'], label='Quality Index', color='red')
    plt.title('Pressure and Quality Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Measurement')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
# Plot metrics
plot_metrics(sensor_data)
# # Predictive Analytics
from sklearn.linear_model import LinearRegression
# Predictive analysis example
def predict_water_demand(data):
    data['Time'] = (data['Timestamp'] - data['Timestamp'].min()).dt.total_seconds() // 60  # Convert to minutes
    X = data[['Time']]
    y = data['Consumption']
    model = LinearRegression()
    model.fit(X, y)
    future_time = np.array([[x] for x in range(len(data), len(data) + 10)])  # Predict next 10 minutes
    predictions = model.predict(future_time)
    plt.figure(figsize=(10, 5))
    plt.plot(data['Timestamp'], data['Consumption'], label='Historical Consumption', color='blue')
    plt.plot(pd.date_range(start=data['Timestamp'].iloc[-1], periods=10, freq='T'), predictions, label='Predicted Consumption', color='orange', linestyle='--')
    plt.title('Predicted Water Consumption')
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption (L)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
# Predict and plot demand
predict_water_demand(sensor_data)




