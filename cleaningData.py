
import pandas as pd

#import uncleaned version of data 
dataCleaned= pd.read_csv('final_flights_weather_merged.csv')


# drop columns were not curently interested in 

dataCleaned.drop(['AIRLINE', 'AIRLINE_DOT', 'DOT_CODE','FL_NUMBER','CRS_DEP_TIME','DEP_TIME',
                   'TAXI_OUT','WHEELS_ON','WHEELS_OFF', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME',
                   'CANCELLATION_CODE','DIVERTED','CRS_ELAPSED_TIME','ELAPSED_TIME','AIR_TIME',
                  'Date', 'EventId','TimeZone', 'LocationLat','LocationLng', 'County','ZipCode',
                  'EventId_dest_weather','TimeZone_dest_weather','LocationLat_dest_weather',
                  'LocationLng_dest_weather','County_dest_weather','ZipCode_dest_weather','City',
                  'State','State_dest_weather','City_dest_weather'], axis=1 , inplace=True) 

# Save the final cleaned merged dataset
dataCleaned.to_csv('/Users/kaitlyndavies/Desktop/506Project/cleaned_FandWFinal.csv', index=False)

# Save a smaller sample for quick verification (first 50 rows)
dataCleaned.head(50).to_csv('/Users/kaitlyndavies/Desktop/506Project/small_sample_cleanedFinal.csv', index=False)
