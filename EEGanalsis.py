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




