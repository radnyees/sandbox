import pandas as pd
import numpy as np

# Creating a time index for some random synthetic EEG Dataset 
time= pd.date_range(start='2024-01-11', periods=1000, freq= 'ms')

# Generating synthetic EEG data for 4 channels

np.random.seed(0)
data=np.random.randn(1000,4)

#Creating a data frame
df= pd.DataFrame(data, columns=['Fp1','Fp2','F3','F4'], index=time)

# Inspecting first few rows
print(df.head())

# For checking data types and summary statistics
print(df.info())
print(df.describe())

def handling_missing_values():
    import pandas as pd
    import numpy as np
    df_missing = data.copy()
    df_missing.iloc[::50] = np.nan

    # Identifying missing values
    print("Number of missing values per column:")
    print(df_missing.isnull().sum())

    df_dropped = df_missing.dropna()
    df_interpolated = df_missing.interpolate()
    print("Original data shape:", df_missing.shape)
    print("Data after dropping missing values:", df_dropped.shape)
    print("Data after interpolation:")
    print(df_interpolated.head(60))

handling_missing_values

# Filtering EEG Signals

def filtering_signals():

    start_time = '2024-01-11 00:00:00.200'
    end_time = '2024-01-11 00:00:00.400'
    df_time_filtered = df.loc[start_time:end_time]

    #filtering the EEG signals keeping Fp1 and F3
    channels_to_keep = ['Fp1', 'F3']
    df_channel_filtered = df_time_filtered[channels_to_keep]

    # Condition applied (Amplitude 1.5)
    df_condition_filtered = df_channel_filtered[df_channel_filtered['Fp1'] > 1.5]

    print("Filtered Data:")
    print(df_condition_filtered)
filtering_signals

# Spectral Analysis of EEG Data
def analyse_EEG():
    import numpy as np
    import pandas as pd
    

    # General Synthetic Data Created

    sampling_rate = 1000  # in Hz (samples per second)
    duration = 2  # in seconds
    n = sampling_rate * duration  # Total number of samples
    timestep = 1 / sampling_rate  # Time between samples

    time = np.linspace(0, duration, n, endpoint=False)

    delta = np.sin(2 * np.pi * 2 * time)  # 2 Hz
    theta = np.sin(2 * np.pi * 6 * time)  # 6 Hz
    alpha = np.sin(2 * np.pi * 10 * time)  # 10 Hz
    beta = np.sin(2 * np.pi * 20 * time)  # 20 Hz
    gamma = np.sin(2 * np.pi * 40 * time)  # 40 Hz

    # Adding random noise
    np.random.seed(0)  
    noise = np.random.normal(0, 0.5, n)
    signal = delta + theta + alpha + beta + gamma + noise

    df = pd.DataFrame({'Time': time, 'Fp1': signal})
    df.set_index('Time', inplace=True)
    signal = df['Fp1'].values
    n = len(signal)
    freq = np.fft.fftfreq(n, d=timestep)
    fft_values = np.fft.fft(signal)
    fft_values = np.abs(fft_values)
    freq = freq[:n // 2]
    fft_values = fft_values[:n // 2]

    # Plotting Spectral Density
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq, fft_values)
    plt.title('Frequency Components of Synthetic EEG Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 60) 
    plt.grid(True)
    plt.savefig('eeg_plot.png')
    plt.show()
    
analyse_EEG()





