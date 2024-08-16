from tkinter import Tk, filedialog
from mne import read_evokeds
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# Initialize Tkinter window
root = Tk()
root.withdraw()

# Open dialog for loading FIF files
fif_file_paths = filedialog.askopenfilenames(title="Select FIF files")
print(f"Selected FIF files: {fif_file_paths}")

# Define the channels of interest
channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'PO10', 'O1', 'Oz', 'O2']

# Function to smooth the signal
def smooth_signal(signal, window_size=20):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')
    
# Define time windows for specific ERP components
n1_window = (0.100, 0.200)  # N1 component
p2_window = (0.150, 0.300)  # P2 component
n2_window = (0.200, 0.350)  # N2 component
p3_window = (0.300, 0.600)  # P3 component

# Define the function to find ERP peaks
def find_erp_peak(data, times, time_window, peak_type="pos"):
    start_idx = np.where(times >= time_window[0])[0][0]
    end_idx = np.where(times <= time_window[1])[0][-1]
    sliced_data = data[start_idx:end_idx]
    sliced_times = times[start_idx:end_idx]
    
    if peak_type == "pos":
        peak_idx = np.argmax(sliced_data)
    elif peak_type == "neg":
        peak_idx = np.argmin(sliced_data)
    else:
        raise ValueError("Invalid peak_type. Expected 'pos' or 'neg'.")
    
    peak_latency = sliced_times[peak_idx]
    
    return peak_latency

# Define the function to calculate mean amplitude
def calculate_mean_amplitude(data, times, time_window):
    start_idx = np.where(times >= time_window[0])[0][0]
    end_idx = np.where(times <= time_window[1])[0][-1]
    sliced_data = data[start_idx:end_idx]
    mean_amplitude = np.mean(sliced_data)
    
    return mean_amplitude

# Define the mapping of Marker IDs to conditions
marker_id_to_condition = {
    19: 'Condition1_GreenGo',
    20: 'Condition1_RedStop',
    21: 'Condition1_RedGo',
    22: 'Condition1_GreenStop',
    23: 'Condition2_GreenGo',
    24: 'Condition2_RedStop',
    25: 'Condition2_RedGo',
    26: 'Condition2_GreenStop'
}
    
# Initialize a list to store all ERP results
all_erp_results = []

# Process each FIF file
for fif_file_path in fif_file_paths:
    # Read the evoked responses from the FIF file
    evokeds = mne.read_evokeds(fif_file_path, verbose=False)
    
    # Loop through each evoked response
    for evoked in evokeds:
        # Apply band-pass filter (1-30 Hz)
        evoked.filter(l_freq=1, h_freq=30, picks=channels_of_interest, method='iir', verbose=False)
        
        # Extract the marker ID from the comment using regular expression
        match = re.search(r'\d+', evoked.comment)
        if match:
            marker_id = int(match.group())
        else:
            print(f"No marker ID found in comment: {evoked.comment}")
            continue
        
        # Loop through each channel of interest
        for channel in channels_of_interest:
            # Get the data and times for the current channel
            data = evoked.data[evoked.ch_names.index(channel)]
            times = evoked.times
            
            # Apply smoothing to the data
            smoothed_data = smooth_signal(data)
            
            # Find peak latencies for each ERP component
            n1_peak_latency = find_erp_peak(smoothed_data, times, n1_window, peak_type="neg")
            p2_peak_latency = find_erp_peak(smoothed_data, times, p2_window, peak_type="pos")
            n2_peak_latency = find_erp_peak(smoothed_data, times, n2_window, peak_type="neg")
            p3_peak_latency = find_erp_peak(smoothed_data, times, p3_window, peak_type="pos")
            
            # Calculate mean amplitudes for each ERP component
            n1_mean_amp = calculate_mean_amplitude(smoothed_data, times, (n1_peak_latency - 0.050, n1_peak_latency + 0.050))
            p2_mean_amp = calculate_mean_amplitude(smoothed_data, times, (p2_peak_latency - 0.050, p2_peak_latency + 0.050))
            n2_mean_amp = calculate_mean_amplitude(smoothed_data, times, (n2_peak_latency - 0.050, n2_peak_latency + 0.050))
            
            if p3_peak_latency == 0.6:
                p3_mean_amp = calculate_mean_amplitude(smoothed_data, times, (0.550, 0.650))
            else:
                p3_mean_amp = calculate_mean_amplitude(smoothed_data, times, (p3_peak_latency - 0.050, p3_peak_latency + 0.050))
            
            # Store results in the all_erp_results list
            all_erp_results.append([marker_id, marker_id_to_condition.get(marker_id, ''), channel, n1_mean_amp, n1_peak_latency, p2_mean_amp, p2_peak_latency, n2_mean_amp, n2_peak_latency, p3_mean_amp, p3_peak_latency])

# Specify the output folder path
output_folder = filedialog.askdirectory(title="Select the output folder")

# Create a DataFrame from the all_erp_results list
df = pd.DataFrame(all_erp_results, columns=['Marker ID', 'Condition', 'Channel', 'N1 Mean Amplitude', 'N1 Latency', 'P2 Mean Amplitude', 'P2 Latency', 'N2 Mean Amplitude', 'N2 Latency', 'P3 Mean Amplitude', 'P3 Latency'])

# Save the DataFrame to an Excel file
output_file = os.path.join(output_folder, "ERP_Results.xlsx")
df.to_excel(output_file, index=False)
print(f"Saved all ERP results to {output_file}")