import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


# SIMULATE EEG DATA (CSV)

# Set random seed for reproducibility
np.random.seed(42)

# Define sampling frequency and duration
sfreq = 256         # Hz
duration_sec = 2.0  # Total recording duration in seconds
n_samples = int(sfreq * duration_sec)

# Simulate 5 EEG channels
n_channels = 5

# Create a time vector and random data for each channel
time = np.linspace(0, duration_sec, n_samples, endpoint=False)
eeg_data = np.random.randn(n_samples, n_channels)

# Build a DataFrame: First column = time, remaining columns = EEG channels
columns = ['time'] + [f'ch_{i+1}' for i in range(n_channels)]
df_eeg = pd.DataFrame(np.column_stack([time, eeg_data]), columns=columns)

# Save to CSV
df_eeg.to_csv('simulated_eeg.csv', index=False)
print("Simulated EEG data saved as 'simulated_eeg.csv'")

###############################################################################
#  SIMULATE EVENTS (TSV)

# We'll create two events, both within the 2-second window, no overlap
events_dict = {
    'onset':     [0.4, 1.2],       # Times in seconds
    'duration':  [0.1, 0.1],       # Each event lasts 0.1s
    'trial_type': ['Stimulus', 'Response']
}
df_events = pd.DataFrame(events_dict)

# Save to TSV
df_events.to_csv('simulated_events.tsv', sep='\t', index=False)
print("Simulated events saved as 'simulated_events.tsv'")


#  LOAD EEG DATA FROM CSV INTO MNE

# Read the CSV
df_loaded = pd.read_csv('simulated_eeg.csv')
time_vals = df_loaded['time'].to_numpy()
eeg_vals = df_loaded.iloc[:, 1:].to_numpy()  # all columns except 'time'

# Create MNE Info object
ch_names = list(df_loaded.columns[1:])  # channel names (ch_1, ch_2, etc.)
ch_types = ['eeg'] * len(ch_names)      # all channels are EEG

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create MNE RawArray (channels x samples => transpose the data)
raw = mne.io.RawArray(eeg_vals.T, info)

# Optional: Plot raw data (without annotations) and save
fig_raw = raw.plot(show=False)
fig_raw.savefig('raw_eeg_plot.png')
plt.close(fig_raw)
print("Raw EEG plot saved as 'raw_eeg_plot.png'")


#  LOAD EVENT MARKERS FROM TSV AND SET AS ANNOTATIONS

events_df = pd.read_csv('simulated_events.tsv', sep='\t')

# Create MNE Annotations
annotations = mne.Annotations(
    onset=events_df['onset'].to_numpy(),
    duration=events_df['duration'].to_numpy(),
    description=events_df['trial_type'].to_numpy()
)

# Attach annotations to the raw data
raw.set_annotations(annotations)

# Plot annotated raw data and save
fig_annotated = raw.plot(show=False)
fig_annotated.savefig('annotated_eeg_plot.png')
plt.close(fig_annotated)
print("Annotated EEG plot saved as 'annotated_eeg_plot.png'")


#  CONVERT ANNOTATIONS TO EVENTS AND EPOCH

events, event_id = mne.events_from_annotations(raw)

# We have two events: "Stimulus" and "Response"
print("Events found:\n", events)
print("Event IDs:\n", event_id)

# Create epochs around each event
# tmin/tmax define the time window around each event (in seconds)
epochs = mne.Epochs(
    raw, 
    events, 
    event_id=event_id, 
    tmin=-0.1,  # 100ms before event
    tmax=0.4,   # 400ms after event
    preload=True,
    event_repeated='error'  # We won't have repeated events, so no error
)

# Plot epochs and save
fig_epochs = epochs.plot(show=False)
fig_epochs.savefig('epochs_plot.png')
plt.close(fig_epochs)
print("Epochs plot saved as 'epochs_plot.png'")


# AVERAGE (CREATE AN EVOKED) AND PLOT

evoked = epochs.average()

fig_evoked = evoked.plot(show=False)
fig_evoked.savefig('evoked_plot.png')
plt.close(fig_evoked)
print("Evoked plot saved as 'evoked_plot.png'")


