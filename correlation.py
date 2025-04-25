import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Load the dataset
df = pd.read_csv("final_flights_weather_merged.csv")

# Step 1: Handle missing values
df['Precipitation_Dest'] = df['Precipitation_Dest'].fillna(0)
df['Precipitation_Origin'] = df['Precipitation_Origin'].fillna(0)

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

# Step 5: Correlations (Severity only)
correlation_origin_arrival = df['Severity_Origin_numerical'].corr(df['ARR_DELAY'])
correlation_dest_arrival = df['Severity_Dest_numerical'].corr(df['ARR_DELAY'])
correlation_origin_departure = df['Severity_Origin_numerical'].corr(df['DEP_DELAY'])
correlation_dest_departure = df['Severity_Dest_numerical'].corr(df['DEP_DELAY'])

print(f"Correlation between origin weather severity and arrival delay: {correlation_origin_arrival}")
print(f"Correlation between destination weather severity and arrival delay: {correlation_dest_arrival}")
print(f"Correlation between origin weather severity and departure delay: {correlation_origin_departure}")
print(f"Correlation between destination weather severity and departure delay: {correlation_dest_departure}")

# Step 6: Define a custom weather impact score function
def weather_impact_score(precip, severity):
    return precip * severity

# Step 7: Create score columns
df['WeatherImpactScore_Origin'] = weather_impact_score(df['Precipitation_Origin'], df['Severity_Origin_numerical'])
df['WeatherImpactScore_Dest'] = weather_impact_score(df['Precipitation_Dest'], df['Severity_Dest_numerical'])

# Step 8: Correlation of weather impact score with delays
score_corr_origin_arr = df['WeatherImpactScore_Origin'].corr(df['ARR_DELAY'])
score_corr_dest_arr = df['WeatherImpactScore_Dest'].corr(df['ARR_DELAY'])
score_corr_origin_dep = df['WeatherImpactScore_Origin'].corr(df['DEP_DELAY'])
score_corr_dest_dep = df['WeatherImpactScore_Dest'].corr(df['DEP_DELAY'])

print(f"\nCorrelation between origin weather impact score and arrival delay: {score_corr_origin_arr}")
print(f"Correlation between destination weather impact score and arrival delay: {score_corr_dest_arr}")
print(f"Correlation between origin weather impact score and departure delay: {score_corr_origin_dep}")
print(f"Correlation between destination weather impact score and departure delay: {score_corr_dest_dep}")

# Step 9: Plot Precipitation at Origin vs Arrival Delay with log scale
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df[df['Precipitation_Origin'] > 0],
    x='Precipitation_Origin',
    y='ARR_DELAY',
    scatter_kws={'alpha': 0.3},
    line_kws={"color": "red"},
    ci=None
)
plt.xscale('log')
plt.title('Precipitation at Origin vs Arrival Delay (Log Scale)')
plt.xlabel('Precipitation at Origin (inches, log scale)')
plt.ylabel('Arrival Delay (minutes)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Plot Precipitation at Dest vs Departure Delay
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x='Precipitation_Dest',
    y='DEP_DELAY',
    scatter_kws={'alpha': 0.3},
    line_kws={"color": "red"},
    ci=None   # no confidence interval (cleaner)
)
plt.title('Precipitation at Destination vs Departure Delay')
plt.xlabel('Precipitation at Destination (inches)')
plt.ylabel('Departure Delay (minutes)')
plt.grid(True)
plt.show()


#                 ***********Below this line *******************
#**********************************This needs Fixing **************************

#Step 6: Weather Severity Over Time with Trend Line
# Convert FL_DATE to datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# Group by date and calculate average severity per day
daily_severity = df.groupby('FL_DATE')['Severity_Origin_numerical'].mean().reset_index()

# Fill missing severity with 2 
daily_severity['Severity_Origin_numerical'] = daily_severity['Severity_Origin_numerical'].fillna(2)


# Drop only rows missing dates (which shouldn't happen often)
daily_severity = daily_severity.dropna(subset=['FL_DATE'])

# Convert dates to numeric ordinal format for regression
daily_severity['date_ordinal'] = daily_severity['FL_DATE'].map(pd.Timestamp.toordinal)

# Perform linear regression on the data
slope, intercept, r_value, p_value, std_err = linregress(daily_severity['date_ordinal'], daily_severity['Severity_Origin_numerical'])

# Calculate predicted values for the trend line
trend_line = intercept + slope * daily_severity['date_ordinal']

# Plot weather severity over time with the trend line
plt.figure(figsize=(12, 6))
plt.scatter(daily_severity['FL_DATE'], daily_severity['Severity_Origin_numerical'], alpha=0.5, label='Daily Avg Severity')
plt.plot(daily_severity['FL_DATE'], trend_line, color='red', linewidth=2, label=f'Trend Line (r={r_value:.3f})')
plt.title('Weather Severity Over Time with Trend Line')
plt.xlabel('Date')
plt.ylabel('Average Weather Severity (1=Mild, 3=Severe)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Print regression results
print(f"Slope: {slope}")
print(f"R-squared: {r_value**2}")
print(f"P-value: {p_value}")



##Check for Non-Linear Relationships (Threshold Effects)
# Group severity into bins and compute average delays
df['Severity_Origin_binned'] = pd.cut(df['Severity_Origin_numerical'], bins=[0, 1.5, 2.5, 3.5, 6], labels=['Mild', 'Moderate', 'Severe', 'Extreme'])

# Plot average delay for each severity bin
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Severity_Origin_binned', y='ARR_DELAY')
plt.title('Arrival Delay by Weather Severity Bins')
plt.xlabel('Weather Severity')
plt.ylabel('Arrival Delay (min)')
plt.grid(True)
plt.show()

# Analyze Delays by Airline, Region, and Season
# Extract month and region for seasonal and regional analysis
df['Month'] = pd.to_datetime(df['FL_DATE']).dt.month
df['Season'] = pd.cut(df['Month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])

# Airline-level average delay
airline_delays = df.groupby('AIRLINE')['ARR_DELAY'].mean().sort_values()

plt.figure(figsize=(12, 6))
airline_delays.plot(kind='bar')
plt.title('Average Arrival Delay by Airline')
plt.ylabel('Average Delay (min)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Seasonal delay patterns
# Set observed=False to retain the current behavior (only groups with data will be shown)
seasonal_delays = df.groupby('Season', observed=False)['ARR_DELAY'].mean()


plt.figure(figsize=(6, 4))
seasonal_delays.plot(kind='bar', color='skyblue')
plt.title('Average Arrival Delay by Season')
plt.ylabel('Average Delay (min)')
plt.grid(True)
plt.tight_layout()
plt.show()



# predicting delays w classification 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Binary target: 1 if ARR_DELAY > 15 minutes
df['Delayed'] = df['ARR_DELAY'] > 15

# Select features and drop NA
features = ['Severity_Origin_numerical', 'Severity_Dest_numerical', 'Precipitation_Origin', 'Precipitation_Dest']
X = df[features].dropna()
y = df.loc[X.index, 'Delayed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
