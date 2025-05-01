# CS506-Final-Project-Repository
# Group Members: 
   Kaitlyn Davies, Michael Ahrens, Mehmet Sarioglu

#  Building and Running our Code  


# Project Description 

For our final project we took an interest in examining flight and weather data. With the support of data from Kaggle  and _____[insert climate data]_____ we express how weather type, severity, and location affect delay frequency and duration and address how worsening weather could impact the American aviation industry making air travel more difficult as the world adapts to the current climate crisis. 

For our final project, we explored  flight performance and weather conditions using data sourced from Kaggle and [insert climate data]. Our analysis focuses on how differing weather types, their severity, and geographic locations influence delay frequency and duration. We took great interest in examining how increasingly extreme weather patterns driven by the climate crisis could challenge the resilience of the American aviation industry, potentially making air travel more unpredictable and difficult in the years to come.

# Preprocessing Data 
A critical part of our initial efforts focused on constructing a clean dataset by merging airport, weather, and flight records. This process involved extensive cleaning to resolve inconsistencies in time formats, standardize airport codes, and align disparate data schemas, ensuring accurate and meaningful analysis across all dimensions.
Airports Dataset:
We began by importing the airports dataset directly from a CSV file, carefully removing any unnecessary header lines. From this dataset, we then extracted the ICAO and IATA codes—crucial identifiers for linking weather and flight data. To ensure consistency during merging, we standardized these codes by trimming whitespace and converting all entries to uppercase for consistency.

## Weather Dataset:
For the weather data, we converted the StartTime(UTC) column to a datetime format and extracted the date to align with the flight schedule data. The airport codes in this dataset were ICAO format, so we standardized them in the same way and then merged the weather dataset with the cleaned airports dataset to map each ICAO code to its corresponding IATA code. This step was essential, as FAA flight records use IATA codes exclusively.


## Flights Dataset:
Flight data was loaded with careful attention to formatting and structure. The FL_DATE column was explicitly converted to datetime to align it better with the weather data set. The ORIGIN and DEST airport columns were uppercased and stripped of any trailing or leading whitespace to ensure clean joins.

We then executed a two-stage merge process:

## Origin Airport Merge: The weather dataset was merged with the flight dataset using the origin airport IATA code and the extracted date as keys. This allowed us to append origin-weather features to each flight record.

### Destination Airport Merge: 
We repeated the process to append destination-weather features using the destination airport IATA code and the same date field. After these merges, we dropped any redundant or duplicate columns (e.g., multiple date or airport code fields), ensuring the final dataset was tidy and efficient for modeling.

# Visualizations Before the Midterm Report 

Leading up to the midterm report, our primary focus was on exploring and familiarizing ourselves with the dataset through preliminary visualizations. This initial phase of data exploration helped us recognize patterns and trends providing valuable insights and raising important questions about the American Aviation industry. These early findings allowed us the groundwork for shifting our project toward a deeper investigation of how climate change may impact air travel in the future.  


# delay by departure city delay by arrival city 
The plots below highlight the cities with the highest average arrival and departure delays. We chose to visualize delays by city rather than by individual airport due to our project's emphasis on weather patterns. Since weather impacts a broader geographic area then just an airport, we assumed a city a more appropriate unit for understanding flight delay influences. 


![Image](https://github.com/user-attachments/assets/4b040e8e-9667-4c27-b0e4-31c687498c7d)
![Image](https://github.com/user-attachments/assets/e1974522-d2e2-4a46-9f4b-ff1bb7770698)




# departure and arrival delay by commercial carriers 
In this visualization, we explored the more lighthearted question: “Which airline should I choose if I want to arrive on time?” The graphs below shows the average departure and arrival delays for each airline, based on their carrier codes. Interestingly, the two graphs are nearly identical. This offered insight into not only airline performance but also served as a reassuring validation of our data processing. Since departure and arrival delays are typically correlated, this consistency suggested that our preprocessing and joins were functioning correctly.


![Image](https://github.com/user-attachments/assets/c3e841db-f6a8-4491-9965-881dae94bc58) 
![Image](https://github.com/user-attachments/assets/5a5e2fd6-49f3-42ae-88c9-736bd08c8ca3)

Additionally we compiled a DBSCAN with the same goal in mind and achieved these results. 
![Image](https://github.com/user-attachments/assets/86331c7d-1800-4932-a617-a533392fdcda)

### Here is a key to easily identify the airlines! 
| Airline Code | Airline            |
|--------------|--------------------|
| EV           | Envoy Air          |
| B6           | JetBlue Airways    |
| YV           | Mesa Air Group     |
| OO           | SkyWest Airlines   |
| F9           | Frontier Airlines  |
| UA           | United Airlines    |
| G4           | Allegiant Air      |
| AA           | American Airlines  |
| OH           | PSA Airlines       |
| MQ           | Envoy Air          |
| NK           | Spirit Airlines    |
| WN           | Southwest Airlines |
| DL           | Delta Air Lines    |
| 9E           | Endeavor Air       |
| YX           | Air Wisconsin      |
| QX           | Horizon Air        |
| AS           | Alaska Airlines    |
| HA           | Hawaiian Airlines  |




# Establishing our argument within our data set  
# correlation 
To support our hypothesis that weather severity impacts flight delays, we first analyzed our dataset to explore potential correlations between increasing weather severity and delays. With further evaluation of the data and the decision to incorporate  weather-related cancellations into our delays, we were able to calculate a moderately positive correlation coefficient of 0.3199. While this value may not appear highly significant on its own, it has meaningful implications given the dataset's large size and relatively short time span. The strength of this correlation, especially in the presence of various confounding factors, provides promising support for our hypothesis. This was a promising development within our project before even considering the potential future increase in weather severity. 

[insert correlation pic weathe sev. Vs avg del] 

# Multivariable Linear Regression 
To further establish our claims and strengthen our argument, we ran multivariable linear regression analysis on our data set with delays as the dependent variable and weather-related factors such as severity and precipitation as the independent variables. The results, presented in the table below, provide insight into how these weather conditions influence flight delays.

# Multivariable Linear Regression Coefficients and Interpretation

| **Variable**                 | **Coefficient** | **Interpretation**                                                                 |
|-----------------------------|----------------|-------------------------------------------------------------------------------------|
| `Severity_Origin_numerical` | +19.97          | Each increase in origin weather severity adds **19.97 minutes** to the arrival delay. |
| `Severity_Dest_numerical`   | +17.81          | Each increase in destination weather severity adds **17.81 minutes** to the departure delay. |
| `Precipitation_Origin`      | +8.92           | Each inch of precipitation at the origin adds **8.92 minutes** to the delay.         |
| `Precipitation_Dest`        | +8.25           | Each inch of precipitation at the destination adds **8.25 minutes** to the delay.     |

From these findings, we can quantify the degree by which severity and precipitation increase delay times. The results show a notable effect, with delays increasing by approximately 40 minutes across the severity scale, as weather patterns currently stand. This finding is particularly significant given the scale used for severity, and it suggests that an increase in weather severity could lead to even more substantial delays, raising concerns about the future resilience of the aviation industry. 



# Hypothesis testing 
To formalize our argument, we took the results from our multivariable linear regression and framed them within a hypothesis testing context. Starting with the null hypothesis that the predictor variables (weather severity and precipitation) had no effect on delay times we obtained a very small p-value, essentially close to zero. This provided a solid foundation for rejecting the null hypothesis at a significance level of 0.05. As a result, we were able to confidently conclude that weather severity does indeed impact flight delays in our dataset, laying the groundwork for the development of our prediction model.

# Climate Data Supported Prediction Modeling 
Our goal within this section of our project is to simulate how climate change will affect flight delays by adjusting our data set to reflect future climate predictions and building and running a prediction model on these new paramaters considering both total precipitation amounts and the frequency/intensity of heavy precipitation events.

# The Climate Data 

## Climate Change Analysis

We start by loading the weather dataset and checking how many events and airports are included. Then, we take a quick look at missing values. To keep things clean, we decide to drop any rows that have missing values because of their low impact. This gets the dataset ready for analysis.

We begin by calculating how long each weather event lasted, in hours. To keep things reasonable, we filter out events that lasted less than an hour or more than 30 days.

Then we summarize how many events happened at each airport, grouped by event type. We also compute a second duration column to get yearly percentages for how long different types of events occurred, broken down by location and severity.

Before clustering, we clean the data by removing non-numeric columns and focusing on relevant event types.

To understand which weather types last longest and are most severe, we group the data again by event type and severity level, calculate average durations, and plot a grouped bar chart. This gives us a clear picture of which weather types cause the most prolonged disruptions across the country.

Then we use the Elbow Method to find the optimal number of clusters by plotting how the KMeans distortion (inertia) changes as we increase the number of clusters from 1 to 19. After that, we look for the "elbow" in the curve to decide the best value for k is 4.

![Elbow-Method Plot](https://github.com/ktdavies/CS506-Final-Project-Repository/blob/main/Visualizations/newplot.png)

This part uses the Elbow Method to find the optimal number of clusters by plotting how the KMeans distortion (inertia) changes as we increase the number of clusters from 1 to 19. After that, we look for the "elbow" in the curve to decide the best value for k.

We apply KMeans clustering with 4 clusters on the cleaned weather data, then assign each airport to a cluster. After that, we merge these cluster labels back with the full weather and location data.

Next, we create an interactive U.S. map showing the spatial distribution of airports by cluster. Each airport is color-coded based on its assigned cluster.

Then we analyzed the behavior of the clusters by looking at how different weather types are distributed within each one. We calculated the proportion of time each event type occurred and visualized the typical weather profile for each cluster using bar charts.

![Cluster Behaviour](https://github.com/ktdavies/CS506-Final-Project-Repository/blob/main/Visualizations/clusters_behaviour.png)

For every cluster, we plot yearly trends for both total duration and average severity.

Each weather type is shown with a different color, and we include linear regression lines regression to highlight any upward or downward trends over the years. The goal is to see how weather patterns have evolved within each cluster, whether certain event types are becoming more frequent or severe over time.

![Yearly Trends](https://github.com/ktdavies/CS506-Final-Project-Repository/blob/main/Visualizations/yearly_trends.png)

And then the last part measures how much weather duration and severity are changing over time within each cluster. It compares regression slopes to baseline values from the first year to calculate the yearly percent change, then displays these trends along with statistical metrics like R² and p-value. Finally, it filters and highlights only statistically significant trends (p < 0.05) to pinpoint meaningful changes in weather patterns across clusters.


## Purpose
To simulate future impacts of climate change on flight delays, we adjust historical precipitation data both in overall amount and in frequency/intensity of heavy precipitation events.

## Adjustments

### 1. Future Scenario Adjustment
- **Adjustment**: +10% to historical precipitation values
- **Source**: [NCA4 - Fourth National Climate Assessment, Chapter 2](https://nca2018.globalchange.gov/chapter/2/)
- **Reason**: Under high emissions scenarios (RCP8.5), models project up to a 10% increase in precipitation across much of the United States by late 21st century, particularly in winter and spring.

### 2. Heavy Precipitation Events Adjustment
- **Adjustment**:
  - Increase **frequency** of heavy precipitation days (2% increase per decade)
  - Increase **intensity** (amount of rain per extreme event)
- **Source**:
  - [NCA4 - Fourth National Climate Assessment, Chapter 2](https://nca2018.globalchange.gov/chapter/2/)
  - [EPA Climate Indicators - Heavy Precipitation](https://www.epa.gov/climate-indicators/climate-change-indicators-heavy-precipitation)
  - https://journals.ametsoc.org/view/journals/apme/61/11/JAMC-D-22-0008.1.xml?utm_source=chatgpt.com
- **Reason**: The intensity and frequency of extreme precipitation events have increased significantly since the mid-20th century, and this trend is expected to continue under future warming scenarios.

## Implementation Summary
- Apply a **1.10 multiplication factor** to baseline precipitation data for future projections.
- Increase the modeled **probability and severity of heavy precipitation events** in simulations.

## Additional Notes
- While a national-level adjustment is used for simplicity, future work could apply **regional adjustment factors** for greater accuracy.

---

### References
- [NCA4 - Fourth National Climate Assessment, Chapter 2](https://nca2018.globalchange.gov/chapter/2/)
- [EPA Climate Indicators - Heavy Precipitation](https://www.epa.gov/climate-indicators/climate-change-indicators-heavy-precipitation)


# Prediction Model 

# Result Summary 

## Project Description  

For our final project we took an interest in examining flight and weather data. With the support of data from Kaggle and _____[insert climate data]_____ we express how weather type, severity, and location affect delay frequency and duration and address how worsening weather could impact the American aviation industry, making air travel more difficult as the world adapts to the current climate crisis.  

For our final project, we explored flight performance and weather conditions using data sourced from Kaggle and **[insert climate data]**. Our analysis focuses on how differing weather types, their severity, and geographic locations influence delay frequency and duration. We took great interest in examining how increasingly extreme weather patterns driven by the climate crisis could challenge the resilience of the American aviation industry, potentially making air travel more unpredictable and difficult in the years to come.  

## Preprocessing Data  

A critical part of our initial efforts focused on constructing a clean dataset by merging airport, weather, and flight records. This process involved extensive cleaning to resolve inconsistencies in time formats, standardize airport codes, and align disparate data schemas, ensuring accurate and meaningful analysis across all dimensions.  

**Airports Dataset:**  
We began by importing the airports dataset directly from a CSV file, carefully removing any unnecessary header lines. From this dataset, we then extracted the ICAO and IATA codes—crucial identifiers for linking weather and flight data. To ensure consistency during merging, we standardized these codes by trimming whitespace and converting all entries to uppercase for consistency.  

**Weather Dataset:**  
For the weather data, we converted the **StartTime (UTC)** column to a datetime format and extracted the date to align with the flight schedule data. The airport codes in this dataset were in ICAO format, so we standardized them in the same way and then merged the weather dataset with the cleaned airports dataset to map each ICAO code to its corresponding IATA code. This step was essential, as FAA flight records use IATA codes exclusively. 

**Flights Dataset**
For the flight data, we loaded the Kaggle Flight Delay and Cancellation Dataset (2019-2023), converted the FL_DATE field to a proper datetime, and trimmed away non-essential columns (e.g., taxi times, wheel-on/off, redundant DOT codes) to keep only schedule, delay-cause, and distance features. Origin and destination IATA codes were upper-cased and merged with the airport lookup, after which we joined daily weather summaries for both airports by matching [IATA, Date]. Key delay-reason and precipitation fields were zero-filled, categorical columns (e.g., cancellation code, weather type/severity) were set to “None” when missing, and timing metrics lacking data were flagged with -1. Finally, we engineered a binary DELAYED label (1 if ARR_DELAY ≥ 15 min) and saved the tidy result as final_flights_weather_merged.csv for downstream modeling.
