import pandas as pd
import matplotlib.pyplot as plt

# Reading cleaned data 
df = pd.read_csv('cleaned_FandWFinal.csv')

# Convert FL_DATE to datetime format
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# Add year column for time-based analysis
df['Year'] = df['FL_DATE'].dt.year

# Filter data for cancellations (assuming 'CANCELLED' == 1 means canceled)
df_cancellations = df[df['CANCELLED'] == 1]

# Group by year and count cancellations
cancellations_per_year = df_cancellations.groupby('Year').size()

# Select only years 2019-2022
cancellations_2019_2022 = cancellations_per_year.loc[2019:2022]

# Plot the number of cancellations
plt.figure(figsize=(8, 6))
bars = plt.bar(cancellations_2019_2022.index, cancellations_2019_2022.values, color='skyblue')
plt.title("Number of Flight Cancellations (2019-2022)")
plt.xlabel("Year")
plt.ylabel("Number of Cancellations")
plt.xticks(cancellations_2019_2022.index, rotation=0)

# Add counts on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  # 5 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.tight_layout()
plt.show()
