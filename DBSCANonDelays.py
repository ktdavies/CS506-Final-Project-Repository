import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt  

#reading cleaned data 
data = pd.read_csv('cleaned_FandWFinal.csv')

#selecting features 
features = ['DEP_DELAY', 'ARR_DELAY', 'DISTANCE', 
            'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 
            'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']

# getting rid of rows that are NaN 
dataNoNa = data[features].dropna()

# look at cleaned data to make sure its okay 
print("Cleaned data shape: " )
print(dataNoNa.shape)
print("cleaned data:")
print(dataNoNa.head())

# taking sample from dataset 
dataSample = dataNoNa.sample(n=10000, random_state=42)

# standardizing the data 
scaler = StandardScaler()
scaledSample = scaler.fit_transform(dataSample)

# applying dbscan 
applyscan = DBSCAN(eps=2, min_samples=15)  # Adjust eps and min_samples as needed
dataSample['cluster'] = applyscan.fit_predict(scaledSample)

print("cluster number with number of points in the cluster:")

#counting the number of data points in each cluster
print(dataSample['cluster'].value_counts())  

# Optionally, you can check the number of noise points (-1)
noisePs = dataSample[dataSample['cluster'] == -1]
print("outliars:")
print(noisePs) 

# clustering sample preview
print(dataSample[['DEP_DELAY', 'ARR_DELAY', 'DISTANCE', 'cluster']].head())

# visualization of the clusters 
plt.scatter(dataSample['DEP_DELAY'], dataSample['ARR_DELAY'], c=dataSample['cluster'], cmap='viridis')
plt.title('DBSCAN Clustering of Flight Delays via Sampling')
plt.xlabel('Departure Delay')
plt.ylabel('Arrival Delay')
plt.colorbar(label='Cluster')
plt.show()