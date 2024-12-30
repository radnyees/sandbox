def simulate_basic_processing():   # Stimulated basic processing with high pass and low pass filters in MNE. Created a data with drift to mimic slow fluctuations.
    import numpy as np
    import pandas as pd
    import mne
    import matplotlib.pyplot as plt

    # Simulate EEG with slow drift across all channels
    np.random.seed(42)

    n_samples = 1000
    n_channels = 4
    sfreq = 200
    time = np.linspace(0, 5, n_samples, endpoint=False)  # 5 seconds of data

    # Creating a slow ~0.1 Hz sinusoidal drift
    drift = 0.5 * np.sin(2 * np.pi * 0.1 * time)  # shape: (1000,)

    # Converting drift to (1000,4) so each channel gets the same drift
    drift_2d = drift[:, np.newaxis]  # shape: (1000, 1)
    drift_2d = np.tile(drift_2d, (1, n_channels))  # shape: (1000, 4)

    # added random noise
    noise = np.random.randn(n_samples, n_channels)  # shape: (1000, 4)

    # Final EEG data: slow drift + random noise
    eeg_data = drift_2d + noise  # shape: (1000, 4)

    # Created a dataframe to save
    columns = ['time'] + [f'ch_{i+1}' for i in range(n_channels)]
    df_eeg = pd.DataFrame(
        np.column_stack([time, eeg_data]),
        columns=columns
    )
    df_eeg.to_csv('simulated_eeg_filter.csv', index=False)
    print("Saved 'simulated_eeg_filter.csv' with 4-channel drift + noise.")

    # Step 2: Load into MNE
    df_loaded = pd.read_csv('simulated_eeg_filter.csv')
    time_vals = df_loaded['time'].to_numpy()
    data_vals = df_loaded.iloc[:, 1:].to_numpy().T  # (channels, samples)

    info = mne.create_info(ch_names=[f'ch_{i+1}' for i in range(n_channels)],
                        sfreq=sfreq,
                        ch_types=['eeg']*n_channels)

    raw = mne.io.RawArray(data_vals, info)

    # Step 3: Filter (e.g., 1-40 Hz)
    raw.filter(l_freq=1.0, h_freq=40.0)

    # Step 4: Plot & Save
    fig_raw = raw.plot(show=False, title='Filtered EEG (1-40 Hz)')
    fig_raw.savefig('filtered_eeg_plot.png')
    plt.close(fig_raw)

    fig_psd = raw.plot_psd(show=False, fmax=50)
    fig_psd.savefig('filtered_eeg_psd.png')
    plt.close(fig_psd)

    print("Plotting complete. See 'filtered_eeg_plot.png' and 'filtered_eeg_psd.png'.")

def average_referencing(): 
    import numpy as np
    import pandas as pd
    import mne
    import matplotlib.pyplot as plt

    # CREATING SIMULATED EEG WITH EVENTS

    sfreq = 250
    n_channels = 6
    duration_sec = 4
    n_samples = int(sfreq * duration_sec)

    np.random.seed(100)

    time = np.linspace(0, duration_sec, n_samples, endpoint=False)
    eeg_data = np.random.randn(n_samples, n_channels)

    # Creating a DataFrame
    columns = ['time'] + [f'ch_{i+1}' for i in range(n_channels)]
    df_eeg = pd.DataFrame(np.column_stack([time, eeg_data]), columns=columns)
    df_eeg.to_csv('simulated_eeg_ref.csv', index=False)

    # Creating a TSV with 3 events
    events_dict = {
        'onset': [0.5, 1.7, 3.0],  # seconds
        'duration': [0.2, 0.1, 0.2],
        'trial_type': ['Stimulus', 'Response', 'Feedback']
    }
    df_events = pd.DataFrame(events_dict)
    df_events.to_csv('simulated_events_ref.tsv', sep='\t', index=False)

    print("Simulated EEG saved as 'simulated_eeg_ref.csv'")
    print("Simulated events saved as 'simulated_events_ref.tsv'")


    # LOADING DATA INTO MNE AND REFERENCE

    df_loaded = pd.read_csv('simulated_eeg_ref.csv')
    data_vals = df_loaded.iloc[:, 1:].to_numpy().T  # shape = (channels, samples)

    info = mne.create_info(
        ch_names=[f'ch_{i+1}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )

    raw = mne.io.RawArray(data_vals, info)

    # Applying average reference
    raw.set_eeg_reference(ref_channels='average')  

    # LOADING EVENTS AS ANNOTATIONS

    events_df = pd.read_csv('simulated_events_ref.tsv', sep='\t')
    annotations = mne.Annotations(
        onset=events_df['onset'].to_numpy(),
        duration=events_df['duration'].to_numpy(),
        description=events_df['trial_type'].to_numpy()
    )
    raw.set_annotations(annotations)

    # Converting to MNE events
    events, event_id = mne.events_from_annotations(raw)

    # EPOCH THE DATA

    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=-0.2, tmax=0.5,  # 200 ms before and 500 ms after each event
        baseline=(None, 0),
        preload=True
    )

    fig_raw = raw.plot(show=False)
    fig_raw.savefig('ref_raw.png')
    plt.close(fig_raw)

    fig_epochs = epochs.plot(show=False)
    fig_epochs.savefig('ref_epochs.png')
    plt.close(fig_epochs)

    evoked = epochs.average()
    fig_evoked = evoked.plot(show=False, titles='Average Referenced Evoked')
    fig_evoked.savefig('ref_evoked.png')
    plt.close(fig_evoked)

    print("Example with average referencing complete. Plots saved: 'ref_raw.png', 'ref_epochs.png', 'ref_evoked.png'")
average_referencing()

def Time_Frequency_Representation():
    import numpy as np
    import pandas as pd
    import mne
    from mne.time_frequency import tfr_morlet
    import matplotlib.pyplot as plt

   
    # SIMULATING EEG & SINGLE EVENT
    
    np.random.seed(123)

    sfreq = 128
    duration_sec = 5
    n_samples = int(sfreq * duration_sec)
    n_channels = 4

    time = np.linspace(0, duration_sec, n_samples, endpoint=False)
    # Adding a 10 Hz oscillation in ch_1 for demonstration
    eeg_data = np.random.randn(n_samples, n_channels)
    eeg_data[:, 0] += 0.5 * np.sin(2 * np.pi * 10 * time)  # 10 Hz in channel 1

    # Making a DataFrame
    columns = ['time'] + [f'ch_{i+1}' for i in range(n_channels)]
    df_eeg = pd.DataFrame(np.column_stack([time, eeg_data]), columns=columns)
    df_eeg.to_csv('simulated_eeg_tfr.csv', index=False)
    print("Simulated EEG with 10 Hz oscillation saved as 'simulated_eeg_tfr.csv'")

    # Creating a single event near the middle
    events_dict = {
        'onset': [2.0],  # 2 seconds into the recording
        'duration': [0.2],
        'trial_type': ['Stimulus']
    }
    df_events = pd.DataFrame(events_dict)
    df_events.to_csv('simulated_events_tfr.tsv', sep='\t', index=False)
    print("Simulated event saved as 'simulated_events_tfr.tsv'")

   
    # LOADING INTO MNE & EPOCH
    
    df_loaded = pd.read_csv('simulated_eeg_tfr.csv')
    data_vals = df_loaded.iloc[:, 1:].to_numpy().T

    info = mne.create_info(
        ch_names=[f'ch_{i+1}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )

    raw = mne.io.RawArray(data_vals, info)

    # Loading events
    events_df = pd.read_csv('simulated_events_tfr.tsv', sep='\t')
    annotations = mne.Annotations(
        onset=events_df['onset'].to_numpy(),
        duration=events_df['duration'].to_numpy(),
        description=events_df['trial_type'].to_numpy()
    )
    raw.set_annotations(annotations)

    # Converting to events
    events, event_id = mne.events_from_annotations(raw)

    # Create epochs: 0.5 s before event to 1.0 s after
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=1.0, preload=True)

    
    # TIME-FREQUENCY ANALYSIS
    
    freqs = np.arange(2, 30, 2)  # 2 to 28 Hz in steps of 2
    n_cycles = freqs / 2.0       # Adaptive number of cycles
    power = tfr_morlet(epochs, picks='all', freqs=freqs, n_cycles=n_cycles, return_itc=False)

    
    fig_power = power.plot_topo(show=False)
    fig_power.savefig('tfr_topo.png')
    plt.close(fig_power)

    # incase we want to print a single channel
    fig_channel = power.plot([0], baseline=(-0.5, 0), mode='logratio', show=False)
    fig_channel.savefig('tfr_channel0.png')
    plt.close(fig_channel)

    print("Time-frequency analysis complete. Plots saved: 'tfr_topo.png', 'tfr_channel0.png'")

def ICA_artifact_removal():
    import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

def simulate_eeg_with_blink():
    """Simulate 6-channel EEG data with a blink artifact in the first channel."""
    np.random.seed(2024)
    sfreq = 250
    n_channels = 6
    duration_sec = 5
    n_samples = int(sfreq * duration_sec)

    time = np.linspace(0, duration_sec, n_samples, endpoint=False)
    eeg_data = np.random.randn(n_samples, n_channels)

    # Inject a blink artifact in channel 1 between t=1.0s and t=1.2s
    blink_start = int(1.0 * sfreq)
    blink_end = int(1.2 * sfreq)
    eeg_data[blink_start:blink_end, 0] += 5.0  # Large amplitude blink

    # Create a DataFrame (optional, for demonstration)
    columns = ['time'] + [f'ch_{i+1}' for i in range(n_channels)]
    df_eeg = pd.DataFrame(np.column_stack([time, eeg_data]), columns=columns)
    df_eeg.to_csv('simulated_eeg_blink.csv', index=False)
    print("Simulated EEG with blink artifact saved as 'simulated_eeg_blink.csv'")

def load_and_prepare_eeg():
    """Load the simulated EEG, rename channels, set montage, filter, and return Raw object."""
    # Read the CSV
    df_loaded = pd.read_csv('simulated_eeg_blink.csv')
    data_vals = df_loaded.iloc[:, 1:].to_numpy().T  # shape: (n_channels, n_samples)

    sfreq = 250
    n_channels = data_vals.shape[0]

    # Create MNE Info
    info = mne.create_info(
        ch_names=[f'ch_{i+1}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )
    raw = mne.io.RawArray(data_vals, info)

    # Rename channels to standard 10-20 system equivalents
    rename_dict = {
        'ch_1': 'Fp1',
        'ch_2': 'Fp2',
        'ch_3': 'C3',
        'ch_4': 'C4',
        'ch_5': 'O1',
        'ch_6': 'O2'
    }
    raw.rename_channels(rename_dict)

    # Set montage (no error because channels match standard_1020)
    raw.set_montage('standard_1020')

    # High-pass filter the data (e.g., 1 Hz) for better ICA performance
    raw.filter(l_freq=1.0, h_freq=None)

    return raw

