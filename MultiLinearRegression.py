import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the dataset
    df = pd.read_csv('sampled147.csv')

    # Step 1: Handle missing values
    df['Precipitation_Dest'] = df['Precipitation_Dest'].fillna(0)
    df['Precipitation_Origin'] = df['Precipitation_Origin'].fillna(0)

    # Map weather severity
    df['Severity_Origin'] = df['WeatherType_Origin'].map({
        'Clear': 'Mild', 'Fog': 'Mild', 'Precipitation': 'Mild', 'Rain': 'Moderate',
        'Snow': 'Severe', 'Storm': 'Severe', 'Cold': 'Severe', 'Hail': 'Severe',
    }).fillna('Mild')

    df['Severity_Dest'] = df['WeatherType_Dest'].map({
        'Clear': 'Mild', 'Fog': 'Mild', 'Precipitation': 'Mild', 'Rain': 'Moderate',
        'Snow': 'Severe', 'Storm': 'Severe', 'Cold': 'Severe', 'Hail': 'Severe',
    }).fillna('Mild')

    # Convert severity to numerical
    severity_map = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
    df['Severity_Origin_numerical'] = df['Severity_Origin'].map(severity_map)
    df['Severity_Dest_numerical'] = df['Severity_Dest'].map(severity_map)

    # Avoid FutureWarning by not using inplace=True
    df['Severity_Origin_numerical'] = df['Severity_Origin_numerical'].fillna(df['Severity_Origin_numerical'].mode()[0])
    df['ARR_DELAY'] = df['ARR_DELAY'].fillna(df['ARR_DELAY'].mean())
    df['DEP_DELAY'] = df['DEP_DELAY'].fillna(df['DEP_DELAY'].mean())

    # Treat weather-related cancellations as extreme delays
    if 'CANCELLATION_CODE' in df.columns:
        df.loc[df['CANCELLATION_CODE'] == 'B', 'ARR_DELAY'] = 400
        df.loc[df['CANCELLATION_CODE'] == 'B', 'DEP_DELAY'] = 400
    
    # Convert FL_DATE to datetime
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

    # Step 2: Run multivariable linear regression
    X = df[['Severity_Origin_numerical', 'Severity_Dest_numerical', 'Precipitation_Origin', 'Precipitation_Dest']]
    y = df['ARR_DELAY']
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Step 3: Visualization of daily trends
    daily_severity = df.groupby('FL_DATE')['Severity_Origin_numerical'].mean().reset_index()
    daily_delay = df.groupby('FL_DATE')['ARR_DELAY'].mean().reset_index()

    merged = pd.merge(daily_severity, daily_delay, on='FL_DATE')

    # Step 4: Correlation matrix
    corr = df[['ARR_DELAY', 'Severity_Origin_numerical', 'Severity_Dest_numerical',
               'Precipitation_Origin', 'Precipitation_Dest']].corr()

    # Plot the correlation matrix (optional)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    main()
