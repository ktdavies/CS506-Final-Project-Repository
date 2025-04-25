import pandas as pd
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("final_flights_weather_merged.csv")

# Step 1: Handle missing values
df['Precipitation_Dest'] = df['Precipitation_Dest'].fillna(0)
df['Precipitation_Origin'] = df['Precipitation_Origin'].fillna(0)

# ------------------------------
# 1. Is weather severity increasing over time?
# ------------------------------

# Step 2: Map Weather Types to Severity Levels
weather_severity_map = {
    'Clear': 'Mild',
    'Fog': 'Mild',
    'Precipitation': 'Mild',
    'Rain': 'Moderate',
    'Snow': 'Severe',
    'Storm': 'Severe',
    'Cold': 'Severe',
    'Hail': 'Severe',
}

df['Severity_Origin'] = df['WeatherType_Origin'].map(weather_severity_map)
df['Severity_Dest'] = df['WeatherType_Dest'].map(weather_severity_map)

# Step 3: Convert severity to numerical values
severity_map = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
df['Severity_Origin_numerical'] = df['Severity_Origin'].map(severity_map)
df['Severity_Dest_numerical'] = df['Severity_Dest'].map(severity_map)

# Step 4: Treat weather-related cancellations as large delays
df.loc[df['CANCELLATION_CODE'] == 'B', 'ARR_DELAY'] = 400
df.loc[df['CANCELLATION_CODE'] == 'B', 'DEP_DELAY'] = 400


#Step 6: Weather Severity Over Time with Trend Line
# Convert FL_DATE to datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# Group by date and calculate average severity per day
daily_severity = df.groupby('FL_DATE')['Severity_Origin_numerical'].mean().reset_index()

# Fill missing severity with 1 (Mild)
daily_severity['Severity_Origin_numerical'] = daily_severity['Severity_Origin_numerical'].fillna(2)

# Drop only rows missing dates (which shouldn't happen often)
daily_severity = daily_severity.dropna(subset=['FL_DATE'])

# Convert dates to numeric ordinal format for regression
daily_severity['date_ordinal'] = daily_severity['FL_DATE'].map(pd.Timestamp.toordinal)

# Perform linear regression on the data
slope, intercept, r_value, p_value, std_err = linregress(daily_severity['date_ordinal'], daily_severity['Severity_Origin_numerical'])

# Calculate predicted values for the trend line
trend_line = intercept + slope * daily_severity['date_ordinal']

# Print regression results
print(f"Slope: {slope}")
print(f"R-squared: {r_value**2}")
print(f"P-value: {p_value}")

# ------------------------------
# 2. Correlation Between Daily Weather Severity and Average Arrival Delay
# ------------------------------

print("\n--- 2. Correlation Between Daily Weather Severity and Average Arrival Delay ---")

# Group by date to get average delay per day
daily_stats = df.groupby('FL_DATE').agg({
    'Severity_Origin_numerical': 'mean',
    'ARR_DELAY': 'mean'
}).reset_index()

# Merge with ordinal date for consistency
daily_stats['date_ordinal'] = daily_stats['FL_DATE'].map(pd.Timestamp.toordinal)

# Correlation between severity and delay
correlation = daily_stats['Severity_Origin_numerical'].corr(daily_stats['ARR_DELAY'])
print(f"Correlation coefficient: {correlation:.4f}")

# Optional: scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(daily_stats['Severity_Origin_numerical'], daily_stats['ARR_DELAY'], alpha=0.3)
plt.title("Daily Weather Severity vs. Daily Arrival Delay")
plt.xlabel("Average Daily Weather Severity (Origin)")
plt.ylabel("Average Arrival Delay (minutes)")
plt.grid(True)
plt.show()

# ------------------------------
# 3. Does increasing weather severity explain increasing delays?
# ------------------------------
# ------------------------------
# 3. Does increasing weather severity explain increasing delays?
# ------------------------------

print("\n--- 3. Multiple Regression: Daily Arrival Delay ~ Time + Weather Severity ---")

# Reuse or create daily aggregates
daily_stats = df.groupby('FL_DATE').agg({
    'Severity_Origin_numerical': 'mean',
    'ARR_DELAY': 'mean'
}).reset_index()

# Convert date to ordinal for regression
daily_stats['date_ordinal'] = pd.to_datetime(daily_stats['FL_DATE']).map(pd.Timestamp.toordinal)

# Prepare features (independent variables) and target (dependent variable)
X = daily_stats[['date_ordinal', 'Severity_Origin_numerical']]
y = daily_stats['ARR_DELAY']

# Handle any missing or infinite values
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)
valid_rows = X.notnull().all(axis=1) & y.notnull()
X = X[valid_rows]
y = y[valid_rows]

# Add constant term (intercept) to the model
X = sm.add_constant(X)

# Run the regression
model = sm.OLS(y, X).fit()

# Print summary of regression results
print(model.summary())




# Reconstruct X and y


# Get predictions and confidence intervals
predictions = model.get_prediction(X)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI

# Plot
plt.figure(figsize=(12, 6))
plt.plot(daily_df['FL_DATE'], y, label='Actual Delay', alpha=0.4)
plt.plot(daily_df['FL_DATE'], pred_summary['mean'], color='red', label='Predicted Delay (OLS)', linewidth=2)
plt.fill_between(
    daily_df['FL_DATE'],
    pred_summary['mean_ci_lower'],
    pred_summary['mean_ci_upper'],
    color='red',
    alpha=0.2,
    label='95% Confidence Interval'
)

plt.title("OLS Regression: Predicted Arrival Delay Over Time\nWith Weather Severity as a Predictor")
plt.xlabel("Flight Date")
plt.ylabel("Mean Arrival Delay (minutes)")
plt.legend()
plt.tight_layout()
plt.show()
