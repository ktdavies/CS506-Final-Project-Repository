import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Import cleaned data
    data = pd.read_csv('sampled147.csv')
    
    # Group by airline and average departure and arrival delays
    airDelay = data.groupby('AIRLINE_CODE')[['DEP_DELAY', 'ARR_DELAY']].mean()

    # Sort by average departure delay
    airDelay = airDelay.sort_values(by='DEP_DELAY', ascending=False)

    # Plot average departure delay by airline
    plt.figure(figsize=(15, 9))
    sns.barplot(x='DEP_DELAY', y='AIRLINE_CODE', data=airDelay, palette='YlGn')
    plt.title('Departure Delay by Commercial Carriers')
    plt.xlabel('Average Departure Delay (mins)')
    plt.ylabel('Airline Code')
    plt.show()

    # Plot average arrival delay by airline
    plt.figure(figsize=(15, 9))
    sns.barplot(x='ARR_DELAY', y='AIRLINE_CODE', data=airDelay, palette='BuGn')
    plt.title('Arrival Delay by Commercial Carriers')
    plt.xlabel('Average Arrival Delay (mins)')
    plt.ylabel('Airline Code')
    plt.show()

if __name__ == "__main__":
    main()
