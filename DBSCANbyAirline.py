import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def main():
    # Reading cleaned data
    data = pd.read_csv('sampled147.csv')
    print("Preview of dataset:")
    print(data.head(5))

    # Selecting features + airline
    features = ['AIRLINE_CODE', 'DEP_DELAY', 'ARR_DELAY', 'DISTANCE',
                'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
                'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']

    # Getting rid of rows that are NaN 
    dataNoNa = data[features].dropna()

    # Clip negative values to zero for delay-related features only
    delay_cols = ['DEP_DELAY', 'ARR_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
                  'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']

    dataNoNa[delay_cols] = dataNoNa[delay_cols].clip(lower=0)

    # Grouping by airline: take average delays, average distance, etc.
    airlineAgg = dataNoNa.groupby('AIRLINE_CODE').mean()

    # Apply log transformation
    logAirlineAgg = np.log1p(airlineAgg)

    # Standardize
    scaler = StandardScaler()
    scaledAirlines = scaler.fit_transform(logAirlineAgg)

    # DBSCAN on airlines
    applyscan = DBSCAN(eps=0.8, min_samples=1)  # min_samples=1 so even single airlines can be a cluster
    airlineAgg['cluster'] = applyscan.fit_predict(scaledAirlines)

    print("\nClusters formed among airlines:")
    print(airlineAgg[['cluster']])

    # Visualization
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(airlineAgg['DEP_DELAY'], airlineAgg['ARR_DELAY'],
                          c=airlineAgg['cluster'], cmap='viridis', s=100)

    for i, txt in enumerate(airlineAgg.index):
        plt.annotate(txt, (airlineAgg['DEP_DELAY'].iloc[i], airlineAgg['ARR_DELAY'].iloc[i]), fontsize=8)

    plt.title('DBSCAN Clustering of Airlines by Average Flight Delays')
    plt.xlabel('Average Departure Delay (minutes)')
    plt.ylabel('Average Arrival Delay (minutes)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    main()
