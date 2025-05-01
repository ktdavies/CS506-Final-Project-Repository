import pandas as pd


# Load the dataset
df = pd.read_csv("final_flights_weather_merged.csv")

print(df['CANCELLATION_CODE'].unique())
# directly comapre delays due to weather w climate change chart 
# taking a specific severe weather event and show what will happen delay wise w this occuracne 