import pandas as pd
import numpy as np

# Creating a time index for some random synthetic EEG Dataset 
time= pd.date_range(start='2024-01-11', periods=1000, freq= 'ms')

# Generating synthetic EEG data for 4 channels

np.random.seed(0)
data=np.random.randn(1000,4)

#Creating a data frame
df= pd.DataFrame(data, columns=['Fp1','Fp2','Fp3','Fp4'], index=time)

# Inspecting first few rows
print(df.head())

# For checking data types and summary statistics
print(df.info())
print(df.describe())
