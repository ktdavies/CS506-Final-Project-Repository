import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Import final cleaned version of data
    data = pd.read_csv('sampled147.csv')

    # Group and average by origin
    odelays = data.groupby('ORIGIN_CITY')[['DEP_DELAY', 'ARR_DELAY']].mean()
    # Group and average by destination
    ddelays = data.groupby('DEST_CITY')[['DEP_DELAY', 'ARR_DELAY']].mean().reset_index()

    # 15 cities with greatest departure delays
    originCities = odelays.sort_values(by='DEP_DELAY', ascending=False).head(15)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='DEP_DELAY', y='ORIGIN_CITY', data=originCities, palette='Purples', hue='ORIGIN_CITY')
    plt.title('Top 15 American Cities with Greatest Departure Delays')
    plt.xlabel('Average Delay (mins)')
    plt.ylabel('City of Origin')
    # Uncomment the line below to save the plot
    # plt.savefig('/Users/kaitlyndavies/Desktop/506Project/Top_15_Departure_Delays.png', bbox_inches='tight')
    plt.show()

    # 15 cities with greatest arrival delays
    destCities = ddelays.sort_values(by='ARR_DELAY', ascending=False).head(15)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='ARR_DELAY', y='DEST_CITY', data=destCities, palette='Blues', hue='DEST_CITY')
    plt.title('Top 15 American Cities with Greatest Arrival Delays')
    plt.xlabel('Average Delay (mins)')
    plt.ylabel('Destination City')
    # Uncomment the line below to save the plot
    # plt.savefig('/Users/kaitlyndavies/Desktop/506Project/Top_15_Arrival_Delays.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
