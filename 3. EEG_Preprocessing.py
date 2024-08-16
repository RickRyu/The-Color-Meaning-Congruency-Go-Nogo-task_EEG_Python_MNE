import mne
import pandas as pd
from autoreject import AutoReject
from tkinter import Tk, filedialog
from scipy.signal import savgol_filter
import numpy as np

# Initialize Tkinter window
root = Tk()
root.withdraw()

# Open dialog for loading EDF file
input_file_path = filedialog.askopenfilename(title="Select an EDF file")
print(f"Selected EDF file: {input_file_path}")
if not input_file_path:
    print("No file selected.")
    exit()

# Load the raw EDF file
raw = mne.io.read_raw_edf(input_file_path, preload=True)

# Pick only the matching channels
matching_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10',
                     'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
                     'PO9', 'PO10', 'O1', 'Oz', 'O2']

try:
    raw.pick_channels(matching_channels)
except ValueError as e:
    print(f"Error: {e}")
    exit()

# Set the montage (channel locations)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# 1. Apply band-pass filter
raw.filter(l_freq=0.1, h_freq=40, method='iir', verbose=False)

# 2. Apply ICA
ica = mne.preprocessing.ICA(n_components=32, random_state=97, max_iter=800)
ica.fit(raw)

# EOG related ICA components
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2', 'C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'T7', 'T8'])
print("Automatically detected EOG indices:", eog_indices)

# ECG related ICA components
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='FT9', method='correlation', threshold='auto')
print("Automatically detected ECG indices:", ecg_indices)

# Muscle related ICA components
muscle_indices, muscle_scores = ica.find_bads_muscle(raw)
print("Automatically detected Muscle indices:", muscle_indices)

# Combine all detected indices
auto_excluded_indices = list(set(eog_indices + ecg_indices + muscle_indices))
print("Combined indices to be excluded:", auto_excluded_indices)

# Set ICA exclusions
ica.exclude = auto_excluded_indices

# Apply ICA correction
raw_corrected = ica.apply(raw.copy(), exclude=ica.exclude)

# Load the marker CSV file
csv_file_path = filedialog.askopenfilename(title="Select a CSV file")
print(f"Selected CSV file: {csv_file_path}")
marker_df = pd.read_csv(csv_file_path)

# Convert time from seconds to samples
sampling_frequency = raw.info['sfreq']
marker_df['latency_samples'] = (marker_df['latency'] * sampling_frequency).astype(int)

# Check if 'type_mapped' column exists in marker_df
if 'type_mapped' in marker_df.columns:
    events = marker_df[['latency_samples', 'type_mapped', 'marker_id']].astype(int).to_numpy()
else:
    marker_df['type_mapped'] = 0
    events = marker_df[['latency_samples', 'type_mapped', 'marker_id']].astype(int).to_numpy()

# Define the time range for baseline epochs
tmin_baseline = 0.0
tmax_baseline = 15.0

# Define the time range for task epochs
tmin_task = -0.2
tmax_task = 1.5

# Create epochs for baseline and tasks combined
epochs = mne.Epochs(raw_corrected, events, event_id={str(marker_id): marker_id for marker_id in np.unique(events[:, 2])},
                    tmin=min(tmin_baseline, tmin_task), tmax=max(tmax_baseline, tmax_task), baseline=None, preload=True)

# Apply AutoReject to all epochs
ar_all = AutoReject(random_state=42, n_jobs=-1)
epochs_ar = ar_all.fit_transform(epochs)

# Get the indices of the remaining epochs after bad epoch removal
remaining_epoch_indices_all = np.where(np.isin(epochs.events[:, 0], epochs_ar.events[:, 0]))[0]

# Create epochs for baseline (eyes open)
if 1 in events[:, 2]:
    baseline_open_epochs = epochs[epochs.events[:, 2] == 1]
    # Filter baseline_open_epochs to include only the remaining epochs
    baseline_open_epochs = baseline_open_epochs[np.isin(baseline_open_epochs.events[:, 0], epochs_ar.events[:, 0])]
    baseline_open_epochs.crop(tmin=tmin_baseline, tmax=tmax_baseline)
    baseline_open_epochs.baseline = (0, 15)
else:
    print("No events found for baseline (eyes open)")
    baseline_open_epochs = None

# Create epochs for baseline (eyes closed)
if 2 in events[:, 2]:
    baseline_closed_epochs = epochs[epochs.events[:, 2] == 2]
    # Filter baseline_closed_epochs to include only the remaining epochs
    baseline_closed_epochs = baseline_closed_epochs[np.isin(baseline_closed_epochs.events[:, 0], epochs_ar.events[:, 0])]
    baseline_closed_epochs.crop(tmin=tmin_baseline, tmax=tmax_baseline)
    baseline_closed_epochs.baseline = (0, 15)
else:
    print("No events found for baseline (eyes closed)")
    baseline_closed_epochs = None

# Create epochs for tasks
task_epochs = epochs[np.isin(epochs.events[:, 2], [19, 20, 21, 22, 23, 24, 25, 26])]
# Filter task_epochs to include only the remaining epochs
task_epochs = task_epochs[np.isin(task_epochs.events[:, 0], epochs_ar.events[:, 0])]
task_epochs.crop(tmin=tmin_task, tmax=tmax_task)
task_epochs.baseline = (-0.2, 0)

# Apply AutoReject to task epochs with cross-validation
ar_task = AutoReject(random_state=100, cv=10)
ar_task.fit(task_epochs)
task_epochs_ar, _ = ar_task.transform(task_epochs, return_log=True)

# Get the indices of the remaining task epochs after bad epoch removal
remaining_task_epoch_indices = np.where(np.isin(task_epochs.events[:, 0], task_epochs_ar.events[:, 0]))[0]

# Apply Savitzky-Golay filter to epochs
def apply_savgol_filter(epoch):
    return savgol_filter(epoch, window_length=11, polyorder=3, axis=-1)

# Apply Savitzky-Golay filter to baseline (eyes open) epochs
if baseline_open_epochs is not None and baseline_open_epochs._data.shape[0] > 0:
    baseline_open_epochs_filtered = baseline_open_epochs.copy().apply_function(apply_savgol_filter)
else:
    print("baseline_open_epochs is empty or None. Skipping Savitzky-Golay filtering.")
    baseline_open_epochs_filtered = None

# Apply Savitzky-Golay filter to baseline (eyes closed) epochs
if baseline_closed_epochs is not None and baseline_closed_epochs._data.shape[0] > 0:
    baseline_closed_epochs_filtered = baseline_closed_epochs.copy().apply_function(apply_savgol_filter)
else:
    print("baseline_closed_epochs is empty or None. Skipping Savitzky-Golay filtering.")
    baseline_closed_epochs_filtered = None

# Apply Savitzky-Golay filter to task epochs
if task_epochs_ar is not None and task_epochs_ar._data.shape[0] > 0:
    task_epochs_filtered = task_epochs_ar.copy().apply_function(apply_savgol_filter)
else:
    print("task_epochs_ar is empty or None. Skipping Savitzky-Golay filtering.")
    task_epochs_filtered = None

# Calculate the average of the filtered baseline (eyes open) epochs
if baseline_open_epochs_filtered is not None:
    average_baseline_open = baseline_open_epochs_filtered.average()
else:
    average_baseline_open = None

# Calculate the average of the filtered baseline (eyes closed) epochs
if baseline_closed_epochs_filtered is not None:
    average_baseline_closed = baseline_closed_epochs_filtered.average()
else:
    average_baseline_closed = None

# Apply baseline correction to task epochs using the combination of time-interval and polynomial fitting baseline correction
if task_epochs_filtered is not None and (average_baseline_open is not None or average_baseline_closed is not None):
    combined_baseline_data = []
    if average_baseline_open is not None:
        combined_baseline_data.append(average_baseline_open.data)
    if average_baseline_closed is not None:
        combined_baseline_data.append(average_baseline_closed.data)
    
    if combined_baseline_data:
        combined_baseline_data = np.concatenate(combined_baseline_data, axis=1)
        
        # Define the time points for the baseline data
        baseline_times = np.linspace(0, 15, combined_baseline_data.shape[1])
        
        # Define the time points for the task epochs
        task_times = np.linspace(-0.2, 1, task_epochs_filtered.times.shape[0])
        
        # Perform time-interval baseline correction
        interval_duration = 1  # Duration of each interval in seconds
        interval_size = int(interval_duration * sampling_frequency)
        num_intervals = combined_baseline_data.shape[1] // interval_size
        
        interval_baselines = []
        interval_times = []
        for i in range(num_intervals):
            start_idx = i * interval_size
            end_idx = (i + 1) * interval_size
            interval_data = combined_baseline_data[:, start_idx:end_idx]
            interval_mean = np.mean(interval_data, axis=1, keepdims=True)
            interval_baselines.append(interval_mean)
            interval_times.append((start_idx + end_idx) / 2 / sampling_frequency)
        
        interval_baselines = np.concatenate(interval_baselines, axis=1)
        interval_times = np.array(interval_times)
        
        # Interpolate the interval baselines to match the task epoch times
        interpolated_interval_baselines = np.zeros((interval_baselines.shape[0], task_times.shape[0]))
        for i in range(interval_baselines.shape[0]):
            interpolated_interval_baselines[i, :] = np.interp(task_times, interval_times, interval_baselines[i, :])
        
        # Perform polynomial fitting baseline correction
        poly_degree = 3
        poly_baselines = np.apply_along_axis(lambda x: np.polyval(np.polyfit(baseline_times, x, poly_degree), task_times), axis=1, arr=combined_baseline_data)
        
        # Calculate the average of interval baselines and polynomial fitting baselines
        combined_baselines = (interpolated_interval_baselines + poly_baselines) / 2
        
        # Reshape combined_baselines to match the shape of task_epochs_filtered._data
        combined_baselines = np.transpose(np.repeat(combined_baselines[:, np.newaxis, :], task_epochs_filtered._data.shape[0], axis=1), (1, 0, 2))
        
        # Subtract the combined baselines from task_epochs_filtered
        task_epochs_filtered._data -= combined_baselines
    else:
        print("Baseline correction not applied due to missing baseline data.")
else:
    print("Task epochs or baseline data not available. Skipping baseline correction.")


####################################fif#######################################################################################

# Calculate the average of filtered task epochs for each marker_id
average_tasks = {}
if task_epochs_filtered is not None:
    for marker_id in events[np.isin(events[:, 2], [19, 20, 21, 22, 23, 24, 25, 26]), 2]:
        epoch_indices = np.where(task_epochs_filtered.events[:, 2] == marker_id)[0]
        selected_epochs = task_epochs_filtered[epoch_indices]
        
        if len(selected_epochs) > 0:
            average_task = selected_epochs.average()
            average_tasks[str(marker_id)] = average_task
        else:
            print(f"No epochs found for marker_id {marker_id}. Skipping averaging.")
else:
    print("Task epochs not available. Skipping averaging.")

# Save the average epochs
save_path = filedialog.asksaveasfilename(title="Save the average epochs", defaultextension=".fif", filetypes=[("FIF files", "*.fif")])
if save_path:
    # Ensure the filename conforms to MNE's naming conventions for evoked data
    if not save_path.endswith('-ave.fif'):
        save_path = save_path.replace('.fif', '-ave.fif')
    
    if average_tasks:
        # Convert the average epochs dictionary to a list of Evoked objects
        evoked_list = list(average_tasks.values())
        
        # Save all Evoked objects in one file
        mne.write_evokeds(save_path, evoked_list)
    else:
        print("No average tasks available. File not saved.")
else:
    print("File not saved.")