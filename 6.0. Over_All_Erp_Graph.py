import numpy as np
import mne
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15  # 원하시는 글씨 크기로 설정하세요

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
        
        # Apply band-pass filter (1-30 Hz)
        evoked.filter(l_freq=1, h_freq=30, method='iir', verbose=False)
        
        # Smooth the signal
        for ch_idx in range(len(evoked.data)):
            evoked.data[ch_idx] = smooth_signal(evoked.data[ch_idx])
        
        # Append the evoked data to the corresponding event in the dictionary
        evoked_dict[evoked.comment].append(evoked)

# Calculate the grand average for each event type
grand_averages = {event: mne.grand_average(evokeds) for event, evokeds in evoked_dict.items() if evokeds}

# Define colors for each event type
colors = {
    'Condition 1 - Green Go': 'lightgreen',
    'Condition 1 - Red Stop': 'salmon',
    'Condition 1 - Red Go': 'orange',
    'Condition 1 - Green Stop': 'lightblue',
    'Condition 2 - Green Go': 'green',
    'Condition 2 - Red Stop': 'red',
    'Condition 2 - Red Go': 'darkorange',
    'Condition 2 - Green Stop': 'blue'
}

# Calculate the grand average for each event type, but only include specific events for topomap
grand_averages_topomap = {
    event: mne.grand_average(evokeds)
    for event, evokeds in evoked_dict.items()
    if evokeds and event in ['Condition 1 - Green Go', 'Condition 1 - Red Stop', 'Condition 1 - Red Go', 'Condition 1 - Green Stop', 'Condition 2 - Green Go', 'Condition 2 - Red Stop', 'Condition 2 - Red Go', 'Condition 2 - Green Stop']
}
# 'figures' 폴더가 없으면 생성
if not os.path.exists('figures'):
    os.makedirs('figures')

# Plot topo
mne.viz.plot_compare_evokeds(grand_averages_topomap, picks='eeg', axes='topo')
plt.tight_layout()
plt.savefig("figures/fig1.png")
plt.clf()
