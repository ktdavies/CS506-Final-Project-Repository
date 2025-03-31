# CS506 — Midterm Report

**YouTube Presentation Link:  https://youtu.be/rgaosmBN_Nc?feature=shared

## Group Members
Kaitlyn Davies  
Michael Ahrens  
Mehmet Sarioglu


## Project Description

In this project, we explore how weather affects flight delays, with a focus on understanding the broader implications of climate change. Using historical flight data from the FAA and weather data from NOAA, we are developing models to predict delays based on meteorological conditions. While we have not yet integrated emissions data from the EPA, this remains part of our intended scope.

Our ultimate goal is to quantify how climate-driven weather patterns are impacting air travel reliability, and to visualize delay likelihoods using an interactive interface.


## Preliminary Data Processing

So far, we have collected and begun cleaning datasets from the FAA and NOAA. The FAA data includes flight-level delay records with timestamps and delay causes, while NOAA provides weather observations from major U.S. airports.

A significant portion of our early work focused on building a reliable, integrated dataset by merging airport, weather, and flight data. This required substantial data cleaning to ensure consistency across time formats, airport codes, and schemas.

Airports Dataset:
We began by loading the airports dataset explicitly from a CSV file, omitting unnecessary header lines. From this, we extracted the ICAO and IATA codes—the key identifiers needed to match weather and flight records. These codes were standardized by trimming whitespace and converting them to uppercase for consistency.

Weather Dataset:
For the weather data, we converted the StartTime(UTC) column to a datetime format and extracted the date to align with the flight schedule data. The airport codes in this dataset were ICAO format, so we standardized them in the same way and then merged the weather dataset with the cleaned airports dataset to map each ICAO code to its corresponding IATA code. This step was essential, as FAA flight records use IATA codes exclusively.

Flights Dataset:
Flight data was also loaded with careful attention to structure. The FL_DATE column was explicitly converted to datetime, and we extracted the date to match weather data. The ORIGIN and DEST airport columns were uppercased and stripped of any trailing or leading whitespace to ensure clean joins.


We then executed a two-stage merge process:

Origin Airport Merge: The weather dataset was merged with the flight dataset using the origin airport IATA code and the extracted date as keys. This allowed us to append origin-weather features to each flight record.

Destination Airport Merge: We repeated the process to append destination-weather features using the destination airport IATA code and the same date field.

After these merges, we dropped any redundant or duplicate columns (e.g., multiple date or airport code fields), ensuring the final dataset was tidy and efficient for modeling.


## Modeling Work to Date

We have mostly been focusing on understanding what our dataset looks like but have done some prelimilary visualisations. We used DBscan clustering to iunterpret arrival and departure delays and have found them to be well correlated and clustered as we expected them to be. We have also created bar charts and some time series plots showing delays by airline, airport, and date, the frequency of weather-related delays over time, and several other factors.

We intend to develop geospatial maps of airport delay distributions over the coming weeks to help us interpret our data and guide further modeling work.


## Next Steps

For the second half of the project, we plan to:

- Complete integration of weather and flight datasets for all major U.S. airports
- Train and evaluate  models (e.g. random forest, SVM, KNN)
- Begin incorporating emissions data from the EPA to explore long-term climate trends
- Build out visualizations for delay trends and model insights
- Design an  interface for users to explore delay predictions by location and weather condition
