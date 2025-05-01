import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("final_flights_weather_merged.csv")

# Step 1: Handle missing values
df['Precipitation_Dest'] = df['Precipitation_Dest'].fillna(0)
df['Precipitation_Origin'] = df['Precipitation_Origin'].fillna(0)

# Handle missing weather severity values by replacing them with a default severity (e.g., Mild)
df['Severity_Origin'] = df['WeatherType_Origin'].map({
    'Clear': 'Mild',
    'Fog': 'Mild',
    'Precipitation': 'Mild',
    'Rain': 'Moderate',
    'Snow': 'Severe',
    'Storm': 'Severe',
    'Cold': 'Severe',
    'Hail': 'Severe',
})
df['Severity_Origin'].fillna('Mild', inplace=True)

df['Severity_Dest'] = df['WeatherType_Dest'].map({
    'Clear': 'Mild',
    'Fog': 'Mild',
    'Precipitation': 'Mild',
    'Rain': 'Moderate',
    'Snow': 'Severe',
    'Storm': 'Severe',
    'Cold': 'Severe',
    'Hail': 'Severe',
})
df['Severity_Dest'].fillna('Mild', inplace=True)

# Convert weather severity to numerical values
severity_map = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
df['Severity_Origin_numerical'] = df['Severity_Origin'].map(severity_map)
df['Severity_Dest_numerical'] = df['Severity_Dest'].map(severity_map)

# Fill any remaining missing values in Severity columns (just in case)
df['Severity_Origin_numerical'].fillna(df['Severity_Origin_numerical'].mode()[0], inplace=True)
df['ARR_DELAY'].fillna(df['ARR_DELAY'].mean(), inplace=True)
df['DEP_DELAY'].fillna(df['DEP_DELAY'].mean(), inplace=True)

# Step 2: Treat weather-related cancellations as large delays
df.loc[df['CANCELLATION_CODE'] == 'B', 'ARR_DELAY'] = 400
df.loc[df['CANCELLATION_CODE'] == 'B', 'DEP_DELAY'] = 400

# Step 3: Convert FL_DATE to datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# Step 4: Weather Severity Over Time
daily_severity = df.groupby('FL_DATE')['Severity_Origin_numerical'].mean().reset_index()
daily_severity['date_ordinal'] = daily_severity['FL_DATE'].map(pd.Timestamp.toordinal)

# Drop rows with missing or infinite values in the critical columns
daily_severity.dropna(subset=['Severity_Origin_numerical', 'date_ordinal'], inplace=True)

# Step 5: Correlation Between Severity and Both Arrival and Departure Delays
daily_stats = df.groupby('FL_DATE').agg({
    'Severity_Origin_numerical': 'mean',
    'ARR_DELAY': 'mean',
    'DEP_DELAY': 'mean'
}).reset_index()

# Filter out invalid rows
filtered = daily_stats.dropna(subset=['Severity_Origin_numerical', 'ARR_DELAY', 'DEP_DELAY'])
filtered = filtered[np.isfinite(filtered['Severity_Origin_numerical']) & 
                    np.isfinite(filtered['ARR_DELAY']) & 
                    np.isfinite(filtered['DEP_DELAY'])]

# Combine Arrival and Departure Delays into one average delay for the plot
filtered['Avg_Delay'] = (filtered['ARR_DELAY'] + filtered['DEP_DELAY']) / 2

# Step 6: Plotting on the same plot with a single trend line
plt.figure(figsize=(10, 6))

# Plot combined average delay (blue points)
plt.scatter(filtered['Severity_Origin_numerical'], filtered['Avg_Delay'], alpha=0.5, color='blue', label='Average Delay')

# Add a single trend line for both delays
m, b = np.polyfit(filtered['Severity_Origin_numerical'], filtered['Avg_Delay'], 1)
plt.plot(filtered['Severity_Origin_numerical'], m * filtered['Severity_Origin_numerical'] + b, color='red', alpha=0.7, label='Trend Line')

# Customize the plot
plt.title("Weather Severity vs. Average Delay (Arrival & Departure Combined)")
plt.xlabel("Average Weather Severity ")
plt.ylabel("Average Delay (minutes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()