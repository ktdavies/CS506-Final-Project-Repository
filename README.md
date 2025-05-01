# CS506-Final-Project-Repository
# Group Members: 
   Kaitlyn Davies, Michael Ahrens, Mehmet Sarioglu

# Welcome 
Welcome to the CS506 Final Project repository! We’re excited to share our work with you. Before you dive in we’d like to provide some important details about our GitHub repository.

#  Building and Running our Code  

Due to the large size of our original dataset, we were unable to upload it in its entirety. To address this, we’ve created a runOurCode.py file that works with smaller datasets representative of our larger dataset. While the results you generate using this file may not be as extreme, they will follow the same overall trend as our results from the original data presented throughout our project.

To explore further, simply download the runOurCode.py file and run it. This script will clone the repository and automatically import all necessary dependencies. We’ve also included a test file as part of a GitHub workflow to validate our prediction model.
We hope you enjoy exploring our project!

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

## Origin Airport Merge: 
The weather dataset was merged with the flight dataset using the origin airport IATA code and the extracted date as keys. This allowed us to append origin-weather features to each flight record.

## Destination Airport Merge: 
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
The plot preserves the expected near-1-to-1 link between average departure and arrival delays, validating our data joins, yet small offsets remain: several carriers land marginally earlier or later than their take-off delay would suggest. These nuances appear consistently within the airline-level clusters identified by our models, indicating that en-route practices and schedule padding vary across groups as well as individual airlines.
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

![Image](https://github.com/user-attachments/assets/beefe3c5-eeaa-4d59-8cd5-5e48605284ed)

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


# Climate Change Analysis

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


# Climate Data Supported Prediction Modeling 
Our goal within this section of our project is to simulate how climate change will affect flight delays by adjusting our data set to reflect future climate predictions and building and running a prediction model on these new paramaters considering both total precipitation amounts and the frequency/intensity of heavy precipitation events.


## Data Cleaning & Encoding Workflow

The raw flight–weather merge (`cleaned.csv`) arrives with more than two dozen columns, many of which
are irrelevant to delay‑due‑weather modelling.  We retain just the fields needed for our analysis
(listed explicitly in the load statement) and parse all timestamp columns as true datetimes.

### 1  Filter out the COVID anomaly
Traffic patterns during Mar‑2020 → Dec‑2021 diverge sharply from the long‑term trend.  
All rows whose `flDate` falls inside that window are removed before any other processing, leaving
a 2019 and 2022‑2023 core sample.

### 2  Normalise the target delay column
* Missing `delayDueWeather` values are set to 0.  
* Flights cancelled specifically for weather (`cancelled == 1` **and** `cancellationCode == 'B'`)
  are re‑expressed as a synthetic delay of **400 minutes** so the downstream models can treat
  cancellations and long delays on a common numeric scale.

### 3  Isolate weather‑affected flights
Rows with `delayDueWeather == 0` are dropped, yielding a compact `df_weather`
DataFrame that contains only flights where weather was reported to matter.

### 4  Remove unusable weather descriptors
Any residual string code `"UNK"` (or the sentinel –1 that replaced it earlier) in the
origin or destination weather descriptors is eliminated.  
Rows where *both* origin and destination weather blocks are completely missing are also discarded.

### 5  Block‑level missing‑value imputation
For flights where **one side** (origin *or* destination) has no weather measurements at all,
we fill that block with neutral placeholders:

```
startTime = -1   endTime = -1
weatherType = -1   severity = -1   precipitation = 0
```

This preserves row count while signalling “no weather event recorded” to the models.

### 6  Numerical re‑encoding
* **Precipitation**: NaNs → 0 mm.  
* **Event times**: each datetime is mapped to *minutes from midnight* (float) with –1 for missing.  
* **Severity**: `"Light"→0.0`, `"Moderate"→1.0`, `"Heavy"→2.0`, –1.0 for missing.  
* **Weather type**: every unique string token is assigned the next integer ID, cast to
  `float32`; the missing sentinel is –1.0.



## Predictive Modeling with Calendar‑Aware Features

The notebook finishes the data‑engineering pipeline by training two production‑ready
models – one to estimate **weather‑driven departure delay minutes** and another to
flag **full cancellations** – while explicitly incorporating calendar context
(year / month / day) alongside the meteorological signals that were engineered
earlier in the workflow.

### 1 Delay‑Severity Regressor (XGBoost, GPU)

* **Sample** All flights that experienced *some* weather impact
  (`delayDueWeather > 0`) *but were not cancelled*.  
* **Severity weights** Each encoded `weatherType` is mapped to a continuous
  multiplier (e.g. Snow = 3, Rain = 2, Fog = 1) and combined with measured
  precipitation to form quantitative **impact scores** for origin and
  destination stations.  
* **Calendar enrichment** `flDate` is decomposed into `year`, `month`, and
  `day`, giving the learner seasonality awareness without leaking exact dates.  
* **Pipeline**  
  • Numerical columns → z‑score scaling  
  • Categorical columns (airline, airports, weatherType) → sparse one‑hot  
  • XGBoost 2.0 histogram tree booster (`device="cuda"`) with 200 trees, depth 6
    and 0.10 learning‑rate.  
* **Performance** Model quality is logged with RMSE and R² on a 20 % hold‑out
  split, and the fitted object is persisted as
  `cuda_xgb_severity_model.joblib`.

### 2 Cancellation Classifier (Logistic, Balanced)

* **Target** Binary flag where a weather cancellation (“B” code) is encoded as
  `is_canceled = 1`.  
* **Feature space** Same operational, meteorological and calendar attributes
  used for regression, but with *severity* and *weatherType* already in numeric
  form.  
* **Pre‑processing**  
  • Flight/airport identifiers and weather codes are treated as categories and
    one‑hot encoded.  
  • All other columns – including the calendar trio – are standardized.  
* **Model** ℓ₂‑regularised logistic regression (`C = 0.01`) with
  `class_weight="balanced"` to offset the rarity of cancellations.  
* **Evaluation & artefact** A full classification report is printed for the
  hold‑out set, and the trained pipeline is stored as
  `logreg_balanced_01.joblib`.

Both models are therefore **calendar‑aware**, **fully encoded**, and saved to
disk, ready to be plugged into the forward‑simulation notebooks that explore
ten‑year climate scenarios.


### 3 Model Performance Snapshot _(validation split)_

| Model | Metric | Value |
|-------|--------|-------|
| **Delay‑Severity XGBoost** | RMSE | **74.21 min** |
|  | R² | **0.699** |
| **Cancellation Logistic Reg.** | Accuracy | **0.70** |
|  | Precision (cancel=1) | 0.79 |
|  | Recall (cancel=1) | 0.70 |
|  | Macro F1 | 0.70 |

These scores provide a quantitative baseline for future scenario testing.

# The Climate Data 
### Climate‑change scenario forecasting

To estimate how shifting weather patterns could shape operational performance, we took the hold‑out portion of our weather‑enhanced flight dataset and **synthetically advanced it ten years into the future**.  
Cluster‑level trend multipliers – extracted from our historical NOAA analysis – were applied to each record’s precipitation, severity and event‑window length, and the calendar year was rolled forward by +10 while preserving month and day.  Fresh composite features were then recalculated so the modified inputs remain physically consistent.

Running the trained models on this future dataset yields:

* **Cancellation rate:** 53.67 % → **53.96 %** ( +0.28 %)
* **Mean delay:** 78.3 min → **82.7 min** ( +4.5 min)

These figures provide an early quantitative view of the operational risk posed by continuing weather intensification.

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

To quantify and forecast weather-induced disruption, we constructed a two-part predictive system: a delay severity estimator and a cancellation classifier. Together, they allow flight-level projections under both observed and simulated climate scenarios.

## Delay Severity Model

The delay model employs XGBoost regression with GPU acceleration to manage large datasets and complex feature interactions. Training was limited to flights affected by weather but not cancelled, ensuring the model learned only from operational delay cases.  

Feature engineering played a central role. Impact scores were created by combining precipitation amounts, severity ratings, and weather types to numerically encode meteorological conditions at both origin and destination airports. Calendar features (year, month, day) were included to introduce seasonality into the learning process. Numerical variables were standardized and categorical variables, such as airline and airport codes, were one-hot encoded.  

The model used 200 trees with a depth of 6 and a learning rate of 0.10 to strike a balance between flexibility and generalization. On the validation set, it achieved an RMSE of 74.21 minutes and an R² of 0.699, demonstrating reasonable predictive strength in a domain where delays are highly stochastic. The trained model was stored for forward-looking scenario testing.

## Cancellation Classifier

Complementing the delay regressor, a logistic regression classifier was developed to predict binary weather-related cancellations. Flights with a cancellation code "B" (weather) were flagged as positive cases. The classifier used the same weather and calendar features as the delay model, retaining consistency across the pipeline.

Due to the imbalance in cancellation occurrences, ℓ₂ regularization and class weighting were applied. On validation, the classifier returned an accuracy of 0.70, with precision and recall for cancellation events both near 0.70. These results reflect effective calibration between false positives and false negatives in a domain with uneven target distribution.

## Future Scenario Simulation

Both models were applied to a synthetically modified dataset reflecting projected climate conditions ten years ahead. Precipitation was increased by 10%, the frequency and intensity of heavy precipitation events were augmented, and the calendar was advanced while maintaining seasonal structure.

Results showed a modest but meaningful rise in disruption indicators. The cancellation rate edged up from 53.67% to 53.96%, while the mean delay increased from 78.3 minutes to 82.7 minutes, an average escalation of 4.5 minutes per flight. These shifts underline the operational vulnerabilities posed by a warming climate.


# Result Summary 

