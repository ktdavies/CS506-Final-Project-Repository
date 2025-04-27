import pandas as pd
import numpy as np

# === Load Dataset ===
file_path = '/Users/michael/Desktop/CS506 Project/Data Processing/sampled_cleamed_flights_weather.csv'  # Update if needed
df = pd.read_csv(file_path)

# === Define Parameters ===
future_precip_increase = 1.10  # +10% precipitation overall
heavy_precip_threshold = 0.3   # Threshold for heavy precipitation in inches
heavy_event_intensity_increase = 1.10  # +10% more rain on heavy days
frequency_boost_percent = 16  # +16% more rainy days
new_precip_range = (0.05, 0.1)  # Inches for new light rain days

def remap_severity(df, precip_col, severity_col):
    """
    Remaps severity based on precipitation values.
    """
    conditions = [
        (df[precip_col].isna()) | (df[precip_col] == 0),
        (df[precip_col] > 0) & (df[precip_col] <= 0.1),
        (df[precip_col] > 0.1) & (df[precip_col] <= 0.5),
        (df[precip_col] > 0.5) & (df[precip_col] <= 1.0),
        (df[precip_col] > 1.0)
    ]
    choices = [np.nan, 'Light', 'Moderate', 'Heavy', 'Severe']
    
    df[severity_col] = np.select(conditions, choices, default=np.nan)

# === Adjust Precipitation Intensity ===
adjusted_df = df.copy()

# Apply +10% to all precipitation values
for col in ['Precipitation_Origin', 'Precipitation_Dest']:
    if col in adjusted_df.columns:
        adjusted_df[col] = adjusted_df[col] * future_precip_increase

# Further increase heavy precipitation events
for col in ['Precipitation_Origin', 'Precipitation_Dest']:
    if col in adjusted_df.columns:
        heavy_mask = adjusted_df[col] > heavy_precip_threshold
        adjusted_df.loc[heavy_mask, col] *= heavy_event_intensity_increase

# === Adjust Precipitation Frequency ===
for col in ['Precipitation_Origin', 'Precipitation_Dest']:
    if col in adjusted_df.columns:
        dry_mask = adjusted_df[col] == 0.0
        dry_indices = adjusted_df[dry_mask].index

        # How many dry days to convert to light rain?
        num_to_convert = int(len(dry_indices) * (frequency_boost_percent / 100))

        # Randomly select dry days
        selected_indices = np.random.choice(dry_indices, size=num_to_convert, replace=False)

        # Assign small random precipitation values
        adjusted_df.loc[selected_indices, col] = np.random.uniform(new_precip_range[0], new_precip_range[1], size=num_to_convert)

# Apply remapping
remap_severity(adjusted_df, 'Precipitation_Origin', 'Severity_Origin')
remap_severity(adjusted_df, 'Precipitation_Dest', 'Severity_Dest')

# === Save Adjusted Dataset ===
adjusted_file_path = 'adjusted_flights_weather_future_climate.csv'
adjusted_df.to_csv(adjusted_file_path, index=False)

print(f"Adjustment complete! Adjusted file saved to {adjusted_file_path}")

# === (Optional) Quick Check ===
adjusted_df.head()
