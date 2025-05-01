import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import uncleaned version of data 
df= pd.read_csv('cleaned_FandWFinal.csv')

# group by origin and average delay
odelays = df.groupby('ORIGIN_CITY')[['DEP_DELAY', 'ARR_DELAY']].mean()

# group by dest and average 
ddelays = df.groupby('DEST_CITY')[['DEP_DELAY', 'ARR_DELAY']].mean()
#sorting cities by delay dep
tocities = odelays.sort_values(by='DEP_DELAY', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x='DEP_DELAY', y='ORIGIN_CITY', data=tocities, palette='Reds')
plt.title('Top 20 Cities with Highest Departure Delays')
plt.xlabel('Average Departure Delay (minutes)')
plt.ylabel('Origin City')
plt.show()
#sorting cities by delay arr 
tdcities = ddelays.sort_values(by='ARR_DELAY', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x='ARR_DELAY', y='DEST_CITY', data=tdcities, palette='Blues')
plt.title('Top 20 Cities with Highest Arrival Delays')
plt.xlabel('Average Arrival Delay (minutes)')
plt.ylabel('Destination City')
plt.show()


