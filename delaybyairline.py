import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import cleaned 
data = pd.read_csv('cleaned_FandWFinal.csv')
# group by airline  and average departure and arrival 
airDelay = data.groupby('AIRLINE_CODE')[['DEP_DELAY', 'ARR_DELAY']].mean()

# sort by avg dep delay
airDelay = airDelay.sort_values(by='DEP_DELAY', ascending=False)

# plot avg dep delay by airline
plt.figure(figsize=(15, 9))
sns.barplot(x='DEP_DELAY', y='AIRLINE_CODE', data=airDelay, palette='YlGn')
plt.title('Departure Delay by Commercial Carriers')
plt.xlabel('Average Departure Delay (mins)')
plt.ylabel('Airline Code')
plt.show()

# plot average arr by airline
plt.figure(figsize=(15, 9))
sns.barplot(x='ARR_DELAY', y='AIRLINE_CODE', data=airDelay, palette='BuGn')
plt.title('Arrival Delay by Commercial Carriers')
plt.xlabel('Average Arrival Delay (mins)')
plt.ylabel('Airline Code')
plt.show()
