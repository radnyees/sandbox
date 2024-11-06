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
    


def EEG_signals_Fp2():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt 
    import seaborn as sns
    sns.set_theme(style='whitegrid')
    sampling_rate = 1000
    duration = 2
    n = sampling_rate * duration
    time = np.linspace(0, duration, n, endpoint=False)
    
    # Generated new
    delta = np.sin(2 * np.pi * 4 * time)
    theta = np.sin(2 * np.pi * 8 * time)
    alpha = np.sin(2 * np.pi * 12 * time)
    beta = np.sin(2 * np.pi * 24 * time)
    gamma = np.sin(2 * np.pi * 40 * time)
    noise = np.random.normal(0, 0.5, n)
    signal = delta + theta + alpha + beta + gamma + noise
    
    df = pd.DataFrame({'Time': time, 'Fp1': signal})
    channel= 'Fp1'
    data_to_plot=df[channel]
    plt.figure(figsize=(12, 4))
    plt.plot(data_to_plot.index, data_to_plot.values)
    plt.title(f'EEG Signal from {channel}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig('eeg_analysis_Fp2')
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def differ_channels():
    # Step 1: Generate synthetic data
    channels = ['Fp1', 'Fp2', 'F3', 'F4']
    time_points = np.arange(1000)
    data = {}  # Initialize 'data' as an empty dictionary

    for channel in channels:
        frequency = np.random.uniform(1, 10)
        data[channel] = np.sin(2 * np.pi * frequency * time_points / 100) \
                        + np.random.normal(0, 0.1, len(time_points))

    df = pd.DataFrame(data, index=time_points)

    # Step 2: Normalize signals
    data_normalized = (df - df.mean()) / df.std()

    # Step 3: Plot signals with offsets
    plt.figure(figsize=(12, 6))
    offset = 5
    for i, channel in enumerate(channels):
        plt.plot(data_normalized.index, data_normalized[channel] + i * offset, label=channel)

    plt.title('Multichannel EEG Signals')
    plt.xlabel('Time')
    plt.ylabel('Normalized Amplitude with Offset')
    plt.yticks([])
    plt.legend(loc='upper right')
    plt.savefig('EEG_Multichannels')
    plt.show()


differ_channels()


