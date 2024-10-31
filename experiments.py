import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('sub-01/ses-T1/func/sub-01_ses-T1_task-SLD_events.tsv', sep='\t')

print(data.head())
print(data.isnull().sum())

# Handling missing responses
# Optionally, fill NaN response times with zeros
# data['response_time'] = data['response_time'].fillna(0)

import pandas as pd

# Assuming your data is saved as 'sub-01/ses-T1/func/sub-01_ses-T1_task-SLD_events.tsv'
data = pd.read_csv('sub-01/ses-T1/func/sub-01_ses-T1_task-SLD_events.tsv', sep='\t')

# Convert 'response_time' to numeric, coerce errors to NaN
data['response_time'] = pd.to_numeric(data['response_time'], errors='coerce')

# Filter out '1-back' and '2-back' trials
one_back = data[data['trial_type'] == '1-back']
two_back = data[data['trial_type'] == '2-back']

def mean_response_time(df):
    correct_trials = df[df['accuracy'] == 1.0]
    return correct_trials['response_time'].mean()

one_back_rt = mean_response_time(one_back)
two_back_rt = mean_response_time(two_back)

print(f"1-back Mean Response Time: {one_back_rt:.3f} seconds")
print(f"2-back Mean Response Time: {two_back_rt:.3f} seconds")


import pandas as pd

# Load the data
data = pd.read_csv('sub-01/ses-T1/func/sub-01_ses-T1_task-SLD_events.tsv', sep='\t')
data['response_time'] = pd.to_numeric(data['response_time'], errors='coerce')

# Separate the data by trial type
one_back = data[data['trial_type'] == '1-back']
two_back = data[data['trial_type'] == '2-back']

# Filter correct trials and extract response times
one_back_correct = one_back[one_back['accuracy'] == 1.0]
one_back_correct_rt = one_back_correct['response_time'].dropna()

two_back_correct = two_back[two_back['accuracy'] == 1.0]
two_back_correct_rt = two_back_correct['response_time'].dropna()

# Function to display detailed calculations
def detailed_mean_calculation(rt_series, condition_name):
    print(f"Detailed Calculation for {condition_name} Correct Trials:")
    print("-" * 50)
    # List all response times
    print("Response Times (seconds):")
    for idx, rt in enumerate(rt_series, 1):
        print(f"Trial {idx}: {rt:.3f}")
    print("-" * 50)
    # Calculate sum and count
    total_rt = rt_series.sum()
    count_rt = rt_series.count()
    print(f"Sum of Response Times: {total_rt:.3f} seconds")
    print(f"Number of Correct Trials: {count_rt}")
    # Calculate mean
    mean_rt = total_rt / count_rt if count_rt != 0 else float('nan')
    print(f"Mean Response Time: {mean_rt:.3f} seconds")
    print("\n")
    return mean_rt

# Perform calculations for each condition
one_back_mean_rt = detailed_mean_calculation(one_back_correct_rt, '1-back')
two_back_mean_rt = detailed_mean_calculation(two_back_correct_rt, '2-back')

# Print final results
print(f"1-back Mean Response Time: {one_back_mean_rt:.3f} seconds")
print(f"2-back Mean Response Time: {two_back_mean_rt:.3f} seconds")



