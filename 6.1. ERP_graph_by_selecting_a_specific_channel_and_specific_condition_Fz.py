import numpy as np
import mne
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25  # 원하시는 글씨 크기로 설정하세요

# Initialize Tkinter window for multiple FIF file selection
root = Tk()
root.withdraw()  # Hide the root window
fif_file_paths = filedialog.askopenfilenames(title="Select multiple FIF files")
root.destroy()  # Destroy the root window after selection

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

# Function to smooth the signal
def smooth_signal(signal, window_size=20):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

# Store evoked data for each event type
evoked_dict = {event: [] for event in event_name_mapping.values()}

# Process each FIF file
for fif_file_path in fif_file_paths:
    # Read the evoked responses from the FIF file
    evokeds = mne.read_evokeds(fif_file_path, verbose=False)

    # Loop through each evoked response
    for evoked in evokeds:
        # Update the event name using the predefined mapping
        event_name = event_name_mapping.get(evoked.comment)

        # If the event name is not found in the mapping, skip this evoked response
        if event_name is None:
            continue

        evoked.comment = event_name

        # Pick specific channels (e.g., 'Fz')
        evoked.pick_channels(['Fz'])

        # Apply band-pass filter (1-30 Hz)
        evoked.filter(l_freq=1, h_freq=30, picks=['Fz'], method='iir', verbose=False)


        # Append the evoked data to the corresponding event in the dictionary
        evoked_dict[evoked.comment].append(evoked)

# Calculate the grand average for each event type
grand_averages = {event: mne.grand_average(evokeds) for event, evokeds in evoked_dict.items() if evokeds}

# Add empty data for missing event types
for event in event_name_mapping.values():
    if event not in grand_averages:
        grand_averages[event] = mne.EvokedArray(np.zeros((1, len(evokeds[0].times))), evokeds[0].info, tmin=evokeds[0].times[0])

# Define colors for each event type
colors = {
    event: color
    for event, color in {
            
            'Condition 1 - Green Go': 'lightgreen',
            'Condition 1 - Red Stop': 'salmon',
            'Condition 1 - Red Go': 'orange',
            'Condition 1 - Green Stop': 'lightblue',
            'Condition 2 - Green Go': 'green',
            'Condition 2 - Red Stop': 'red',
            'Condition 2 - Red Go': 'darkorange',
            'Condition 2 - Green Stop': 'blue'
                
    }.items()
    if event in grand_averages
}

# Plot comparison of evokeds
fig, ax = plt.subplots()

# Create a list of event types defined in the colors dictionary
event_types = list(colors.keys())

# Filter the grand_averages dictionary to include only the desired event types
filtered_grand_averages = {event: grand_averages[event] for event in event_types}

mne.viz.plot_compare_evokeds(filtered_grand_averages, picks='Fz', axes=ax, colors=colors, show=False)

# Adding the head plot for channel location visualization
# Usually, this requires knowing the layout of the channels.
# For simplicity, if you have the layout info, you can plot it using:
# mne.viz.plot_sensors(evokeds[0].info, show_names=['Fz'])

# Show the plot
plt.show()