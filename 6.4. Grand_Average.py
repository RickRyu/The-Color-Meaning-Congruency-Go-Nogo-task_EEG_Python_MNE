from tkinter import Tk, filedialog
import mne
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 원하시는 글씨 크기로 설정하세요

# Initialize Tkinter window for multiple FIF file selection
root = Tk()
root.withdraw()
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

def smooth_signal(signal, window_size=20):
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    padding = (len(smoothed_signal) - len(signal)) // 2
    if padding > 0:
        return smoothed_signal[padding:-padding]
    else:
        return smoothed_signal

evoked_dict = {event: [] for event in event_name_mapping.values()}

for fif_file_path in fif_file_paths:
    evokeds = mne.read_evokeds(fif_file_path, verbose=False)
    for evoked in evokeds:
        evoked.comment = event_name_mapping.get(evoked.comment, evoked.comment)
        evoked.filter(l_freq=1, h_freq=30, method='iir', verbose=False)
        for ch_idx in range(len(evoked.data)):
            evoked.data[ch_idx] = smooth_signal(evoked.data[ch_idx])
        evoked_dict[evoked.comment].append(evoked)

grand_averages = {event: mne.grand_average(evokeds) for event, evokeds in evoked_dict.items()}

# Define time windows for specific ERP components
n1_window = (0.100, 0.200)  # N1 component
n2_window = (0.200, 0.350)  # N2 component
p3_window = (0.300, 0.600)  # P3 component

def find_erp_peak(data, times, time_window, peak_type):
    start_idx = np.where(times >= time_window[0])[0][0]
    end_idx = np.where(times <= time_window[1])[0][-1]
    sliced_data = data[start_idx:end_idx]
    sliced_times = times[start_idx:end_idx]

def find_erp_peak(data, times, time_window, peak_type="neg"):
    start_idx = np.where(times >= time_window[0])[0][0]
    end_idx = np.where(times <= time_window[-1])[0][-1]
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

for event_name, grand_average in grand_averages.items():
    n1_peak_amps = []
    n1_peak_times = []
    n2_peak_amps = []
    n2_peak_times = []
    p3_peak_amps = []
    p3_peak_times = []

    for channel_index in range(len(grand_average.ch_names)):
        data = grand_average.data[channel_index, :]
        times = grand_average.times

        n1_peak_amp, n1_peak_time = find_erp_peak(data, times, n1_window, "neg")
        n2_peak_amp, n2_peak_time = find_erp_peak(data, times, n2_window, "neg")
        p3_peak_amp, p3_peak_time = find_erp_peak(data, times, p3_window, "pos")

        n1_peak_amps.append(n1_peak_amp)
        n1_peak_times.append(n1_peak_time)
        n2_peak_amps.append(n2_peak_amp)
        n2_peak_times.append(n2_peak_time)
        p3_peak_amps.append(p3_peak_amp)
        p3_peak_times.append(p3_peak_time)

    n1_peak_time = np.mean(n1_peak_times)
    n2_peak_time = np.mean(n2_peak_times)
    p3_peak_time = np.mean(p3_peak_times)

    specific_times = [n1_peak_time, n2_peak_time, p3_peak_time]
    grand_average.plot_joint(times=specific_times, title=f"Grand average for event {event_name}", show=False)
    plt.show()
