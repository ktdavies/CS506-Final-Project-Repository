import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Weather.csv")

# Drop rows where severity is 'UNK' or 'Other'
df.drop(df[df['Severity'] == 'UNK'].index, inplace=True)
df.drop(df[df['Severity'] == 'Other'].index, inplace=True)

# Convert StartTime(UTC) to datetime
df['StartTime(UTC)'] = pd.to_datetime(df['StartTime(UTC)'])

# (Optional) Just to make sure severity is treated properly
print(df['Severity'].unique())

# Group data: count number of events per month by severity
df['YearMonth'] = df['StartTime(UTC)'].dt.to_period('M')  # "2023-04" style
severity_over_time = df.groupby(['YearMonth', 'Severity']).size().unstack(fill_value=0)

# Plotting
severity_over_time.plot(kind='line', figsize=(14,8))
plt.title('Weather Event Severity Over Time')
plt.xlabel('Time (Year-Month)')
plt.ylabel('Number of Events')
plt.legend(title='Severity Level')
plt.grid(True)
plt.show()
