import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.nonparametric.smoothers_lowess import lowess



def read_process_data(d, fs, cols):
    all_data = []

    # Process each file in the directory
    for file in fs:
        curr_data = pd.read_csv(os.path.join(d, file))

        curr_data['DATE'] = pd.to_datetime(curr_data['DATE'])
        curr_data = curr_data[cols]

        # Adding the day of the year and the year columns
        curr_data['day_of_year'] = curr_data['DATE'].dt.dayofyear
        curr_data['year'] = curr_data['DATE'].dt.year

        # Averaging the data for each day of each year
        curr_data = curr_data.groupby(['year', 'day_of_year']).mean().reset_index()

        # Changing the nan values to the mean of the column
        for col in cols[1:]:
            curr_data[col] = curr_data[col].fillna(curr_data[col].mean())

        all_data.append(curr_data)
    
    return all_data



def lowess_and_plot(data, cols):
    smoothed_data = data.copy()
    smoothed_data['day_of_year'] = data['day_of_year']
    
    for col in cols[1:]:
        smoothed = lowess(data[col], data['day_of_year'], frac=0.1)
        smoothed_data[col] = smoothed[:, 1]
        
        plt.figure(figsize=(14, 8))
        plt.plot(data['day_of_year'], data[col], 'b.', label='Original data')
        plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', label='Lowess smoothing')
        plt.xlabel('Day of the year')
        plt.ylabel(col)
        plt.title(f'{col} with Lowess smoothing')
        plt.legend()
        plt.savefig(f'Anova/{col}_smoothed.png')
        
    return smoothed_data



# Extracting the data for the lockdown period
def extract_lockdown_data(data):
    ld_start = pd.to_datetime('2020-03-21').dayofyear
    ld_end = pd.to_datetime('2020-06-01').dayofyear
    
    return data[(data['day_of_year'] >= ld_start) & (data['day_of_year'] <= ld_end)].reset_index(drop=True)



def extract_by_year_partitions(data):
    partitions = []
    years = data['year'].unique()
    
    for year in years:
        partitions.append(data[data['year'] == year].reset_index(drop=True))
    
    return partitions


def perfom_anova(data, cols, mode):
    # preparing the lockdown data for the post hoc analysis
    lockdown_data = extract_lockdown_data(data)

    by_year_partitions = extract_by_year_partitions(lockdown_data)

    # Doing ANOVA + Post-hoc analysis for each column
    for column in cols[1:]: 
        anova = stats.f_oneway(*[partition[column] for partition in by_year_partitions])
        print(f'ANOVA for {column}: F-statistic = {anova.statistic}, p-value = {anova.pvalue}')
        
        posthoc = pairwise_tukeyhsd(lockdown_data[column], lockdown_data['year'], alpha=0.05)
        
        print(posthoc)
        fig = posthoc.plot_simultaneous()
        plt.title(f'Tukey HSD post-hoc test for {column} of {mode} data')
        # plt.show()
        plt.savefig(f'Anova/{mode}_{column}_tukey.png')



def main():

    if not os.path.exists('Anova'):
        os.makedirs('Anova')

    # deciding the important columns of study
    crucial_columns = ['DATE', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']

    dir = sys.argv[1]
    files = [f for f in os.listdir(dir) if f.endswith('.csv')]

    processed_data = read_process_data(dir, files, crucial_columns)

    # Combine all the data into a single DataFrame
    data = pd.concat(processed_data, ignore_index=True)

    
    smoothed_data = lowess_and_plot(data, crucial_columns)    

    # Perform ANOVA for the lockdown period of the crucial columns for both the filtered and unfiltered data
    perfom_anova(smoothed_data, crucial_columns, 'filtered')
    perfom_anova(data, crucial_columns, 'unfiltered')



if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python lockdown_period_analysis.py <data_dir>')
        sys.exit(1)

    main()
    
