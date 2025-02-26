import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.api as sm

#Read file section

#file for Halifax (uncomment line 12)
df = pd.read_csv('path/to/Halifax.csv')

#File for Montreal(uncomment line 15)
# df = pd.read_csv('/path/to/Montreal.csv')

#File for Vancouver (uncomment line 18-21)
# vancouver_1_df = pd.read_csv('path/to/Vancouver-1.csv')
# vancouver_2_df = pd.read_csv('path/to/Vancouver-2.csv')
# # Combine the Vancouver data
# df = pd.concat([vancouver_1_df, vancouver_2_df])

df['DATE'] = pd.to_datetime(df['DATE'])
selected_columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE', 'PRCP', 'SNWD', 'TAVG', 'TMAX', 'TMIN']
df_selected = df[selected_columns]

pre_pandemic = df_selected[df_selected['DATE'] < '2020-03-01']
pandemic = df_selected[df_selected['DATE'] >= '2020-03-01']


# print(pre_pandemic.head())
# print(pandemic.head())


def visualising_data(pre_pandemic, pandemic):
    # Convert DATE to ordinal for regression analysis
    pre_pandemic.loc[:, 'DATE_ORD'] = pre_pandemic['DATE'].map(pd.Timestamp.toordinal)
    pandemic.loc[:, 'DATE_ORD'] = pandemic['DATE'].map(pd.Timestamp.toordinal)

    # Drop rows with NaN values in TAVG for regression analysis
    pre_pandemic_clean = pre_pandemic.dropna(subset=['TAVG']).copy()
    pandemic_clean = pandemic.dropna(subset=['TAVG']).copy()

    # Linear regression for pre-pandemic period
    X_pre = sm.add_constant(pre_pandemic_clean['DATE_ORD'])  # Adding a constant for the intercept
    model_pre = sm.OLS(pre_pandemic_clean['TAVG'], X_pre).fit()

    # Linear regression for pandemic period
    X_pandemic = sm.add_constant(pandemic_clean['DATE_ORD'])  # Adding a constant for the intercept
    model_pandemic = sm.OLS(pandemic_clean['TAVG'], X_pandemic).fit()

    # Predicted values
    pre_pandemic_clean.loc[:, 'TAVG_PRED'] = model_pre.predict(X_pre)
    pandemic_clean.loc[:, 'TAVG_PRED'] = model_pandemic.predict(X_pandemic)

    # Plotting the data and the regression lines
    plt.figure(figsize=(14, 7))

    # Plot pre-pandemic data and regression line
    plt.scatter(pre_pandemic_clean['DATE'], pre_pandemic_clean['TAVG'], label='Pre-Pandemic Data', color='blue', alpha=0.5)
    plt.plot(pre_pandemic_clean['DATE'], pre_pandemic_clean['TAVG_PRED'], label='Pre-Pandemic Trend', color='blue')

    # Plot pandemic data and regression line
    plt.scatter(pandemic_clean['DATE'], pandemic_clean['TAVG'], label='Pandemic Data', color='red', alpha=0.5)
    plt.plot(pandemic_clean['DATE'], pandemic_clean['TAVG_PRED'], label='Pandemic Trend', color='red')

    # Adding title and labels
    plt.title('Average Temperature (TAVG) Over Time with Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Average Temperature (TAVG)')
    plt.legend()

    # Display the plot
    plt.show()


def extreme_weather(pre_pandemic_x, pandemic_x):
    #Chi-square test about extreme events during pandemic
    pre_pandemic = pre_pandemic_x.dropna(subset=['TAVG']).copy()
    pandemic = pandemic_x.dropna(subset=['TAVG']).copy()

    threshold = df_selected['TMAX'].quantile(0.90)

    pre_pandemic_extreme_events = pre_pandemic[pre_pandemic['TMAX'] > threshold]
    pandemic_extreme_events = pandemic[pandemic['TMAX'] > threshold]

    # Frequency of extreme events
    pre_pandemic_extreme_count = pre_pandemic_extreme_events.shape[0]
    pandemic_extreme_count = pandemic_extreme_events.shape[0]

    # Print counts
    print(f"Pre-Pandemic Extreme Events: {pre_pandemic_extreme_count}")
    print(f"Pandemic Extreme Events: {pandemic_extreme_count}")

    # Hypothesis testing: Compare proportions of extreme events using Chi-square test
    contingency_table = pd.DataFrame({
        'Extreme Events': [pre_pandemic_extreme_count, pandemic_extreme_count],
        'Total Days': [pre_pandemic.shape[0], pandemic.shape[0]]
    })

    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

    print(f"Chi-square Test: chi2 = {chi2}, p-value = {p_value}")

    # Conclusion based on p-value
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in the frequency of extreme weather events between the pre-pandemic and pandemic periods.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in the frequency of extreme weather events between the pre-pandemic and pandemic periods.")




def calculate_correlation(data, columns):
    
    # Select specified columns and drop rows with any NaN values
    df_selected = data[columns].dropna()
    
    # Calculate correlation matrix
    corr_matrix = df_selected.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Matrix of Weather Variables')
    plt.show()
    
    return corr_matrix

columns_to_analyze = ['PRCP', 'SNWD', 'TAVG', 'TMAX', 'TMIN']
correlation_matrix = calculate_correlation(df_selected, columns_to_analyze)
print(correlation_matrix)

extreme_weather(pre_pandemic, pandemic)
