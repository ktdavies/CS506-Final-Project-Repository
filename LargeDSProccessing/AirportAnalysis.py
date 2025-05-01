import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Flights.csv")

# Keep only the columns we care about
df = df[['FL_DATE', 'ARR_DELAY', 'DEP_DELAY', 'CANCELLED']]

# Convert date column to datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# Treat canceled flights: Set ARR_DELAY to a big value (e.g., 300 minutes delay)
# You can adjust the number if you want
df['ARR_DELAY'] = df.apply(lambda row: 300 if row['CANCELLED'] == 1 else row['ARR_DELAY'], axis=1)

# Group by YearMonth and calculate average "delay" (including fake delay for canceled flights)
df['YearMonth'] = df['FL_DATE'].dt.to_period('M')
delay_over_time = df.groupby('YearMonth')['ARR_DELAY'].mean()

# Plot
plt.figure(figsize=(14,8))
delay_over_time.plot(kind='line', marker='o')
plt.title('Average Flight Delay Over Time (Including Cancellations)')
plt.xlabel('Time (Year-Month)')
plt.ylabel('Average Delay (minutes)')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--')  # 0 delay line
plt.show()
