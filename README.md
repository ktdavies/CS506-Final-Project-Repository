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

Weather Dataset:
For the weather data, we converted the StartTime(UTC) column to a datetime format and extracted the date to align with the flight schedule data. The airport codes in this dataset were ICAO format, so we standardized them in the same way and then merged the weather dataset with the cleaned airports dataset to map each ICAO code to its corresponding IATA code. This step was essential, as FAA flight records use IATA codes exclusively.


Flights Dataset:
Flight data was loaded with careful attention to formatting and structure. The FL_DATE column was explicitly converted to datetime to align it better with the weather data set. The ORIGIN and DEST airport columns were uppercased and stripped of any trailing or leading whitespace to ensure clean joins.

We then executed a two-stage merge process:

Origin Airport Merge: The weather dataset was merged with the flight dataset using the origin airport IATA code and the extracted date as keys. This allowed us to append origin-weather features to each flight record.

Destination Airport Merge: We repeated the process to append destination-weather features using the destination airport IATA code and the same date field.

After these merges, we dropped any redundant or duplicate columns (e.g., multiple date or airport code fields), ensuring the final dataset was tidy and efficient for modeling.

# Visualizations Before the Midterm Report 

Leading up to the midterm report, our primary focus was on exploring and familiarizing ourselves with the dataset through preliminary visualizations. This initial phase of data exploration helped us recognize patterns and trends providing valuable insights and raising important questions about the American Aviation industry. These early findings allowed us the groundwork for shifting our project toward a deeper investigation of how climate change may impact air travel in the future.  


# delay by departure city delay by arrival city 
The plots below highlight the cities with the highest average arrival and departure delays. We chose to visualize delays by city rather than by individual airport due to our project's emphasis on weather patterns. Since weather impacts a broader geographic area then just an airport, we assumed a city a more appropriate unit for understanding flight delay influences. 


[insert pic] 

[insert pic] 


# departure and arrival delay by commercial carriers 
In this visualization, we explored the more lighthearted question: “Which airline should I choose if I want to arrive on time?” The graphs below shows the average departure and arrival delays for each airline, based on their carrier codes. Interestingly, the two graphs are nearly identical. This offered insight into not only airline performance but also served as a reassuring validation of our data processing. Since departure and arrival delays are typically correlated, this consistency suggested that our preprocessing and joins were functioning correctly.


[inset pic] 

[insert pic] 

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


# Result Summary 
![image](https://github.com/user-attachments/assets/463e7a19-555a-4cd4-8186-f153ac9a523c)
