from tkinter import Tk, filedialog
import mne
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 원하시는 글씨 크기로 설정하세요

# Initialize Tkinter window
root = Tk()
root.withdraw()

# Select multiple FIF files
fif_file_paths = filedialog.askopenfilenames(title="Select multiple FIF files")
print(f"Selected FIF files: {fif_file_paths}")

# Select directory to save topomaps
output_dir = filedialog.askdirectory(title="Select directory to save topomaps")
if not output_dir:  # If user cancels directory selection
    print("No directory selected. Exiting.")
    exit()

print(f"Selected output directory: {output_dir}")

# 사전에 정의된 이벤트 이름 매핑
event_name_mapping = {
    '19': 'Condition 1 - Green Go',
    '20': 'Condition 1 - Red Stop',
    '21': 'Condition 1 - Red Go',
    '22': 'Condition 1 - Green Stop',
    '23': 'Condition 2 - Green Go',
    '24': 'Condition 2 - Red Stop',
    '25': 'Condition 2 - Red Go',
    '26': 'Condition 2 - Green Stop'
}

# Initialize a dictionary to collect the Evoked objects for each event
all_evokeds = {}

# Loop through the selected files and read the evoked data
for fif_file_path in fif_file_paths:
    evokeds = mne.read_evokeds(fif_file_path)
    for evoked in evokeds:
        event_name = event_name_mapping.get(evoked.comment, evoked.comment)
        if event_name not in all_evokeds:
            all_evokeds[event_name] = []
        all_evokeds[event_name].append(evoked)

# Define time windows for specific ERP components
n1_window = (0.100, 0.200)  # N1 component
p2_window = (0.150, 0.300)  # P2 component
n2_window = (0.200, 0.350)  # N2 component
p3_window = (0.300, 0.600)  # P3 component

# Define the function to find ERP peaks
def find_erp_peak(data, times, time_window, peak_type):
    start_idx = np.where(times >= time_window[0])[0][0]
    end_idx = np.where(times <= time_window[1])[0][-1]
    sliced_data = data[start_idx:end_idx]
    sliced_times = times[start_idx:end_idx]

    if peak_type == "pos":
        peak_amplitude = np.max(sliced_data)
        peak_time = sliced_times[np.argmax(sliced_data)]
    elif peak_type == "neg":
        peak_amplitude = np.min(sliced_data)
        peak_time = sliced_times[np.argmin(sliced_data)]
    else:
        return np.nan, np.nan
    return peak_amplitude, peak_time

# 특정 채널 선택 (예: 'Fz')
channel_of_interest = 'Fz'

# Compute and plot the average for each selected event
for event_name, evokeds in all_evokeds.items():
    # Combine the evoked objects to get the average
    average_evoked = mne.combine_evoked(evokeds, weights='equal')

    # 특정 채널의 데이터 가져오기
    channel_index = average_evoked.ch_names.index(channel_of_interest)
    data = average_evoked.data[channel_index, :]  # 특정 채널 데이터
    times = average_evoked.times

    # Find peak times for each component
    n1_peak_amp, n1_peak_time = find_erp_peak(data, times, n1_window, "neg")
    n2_peak_amp, n2_peak_time = find_erp_peak(data, times, n2_window, "neg")
    p3_peak_amp, p3_peak_time = find_erp_peak(data, times, p3_window, "pos")

    specific_times = [n1_peak_time, n2_peak_time, p3_peak_time]

    # Plot topomaps without specifying ch_type
    fig = average_evoked.plot_topomap(times=specific_times, show=False)
    fig.suptitle(f"{event_name}")
    
    # Save the figure
    filename = f"{event_name.replace(' ', '_')}_topomap.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)  # Close the figure to free up memory

print(f"Topomaps have been saved in the selected directory: '{output_dir}'")

# If you still want to show all plots at the end, you can use:
# plt.show()
