from tkinter import Tk, filedialog
from mne import read_evokeds
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import seaborn as sns
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 원하시는 글씨 크기로 설정하세요

def smooth_signal(signal, window_size=20):
    smoothed_signal = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        smoothed_signal[:, i] = np.convolve(signal[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed_signal

# 베타 대역 정의
beta_band = (13, 30)

# Initialize a list to store all ERS/ERD results
all_ers_erd_results = []

# Tkinter 창 초기화 및 FIF 파일 선택
root = Tk()
root.withdraw()
fif_file_paths = filedialog.askopenfilenames(title="Select multiple FIF files")
root.destroy()

# 선택한 FIF 파일에서 evoked 데이터 읽기
evokeds = [read_evokeds(fif_file, condition=None, baseline=(None, 0), proj=True) for fif_file in fif_file_paths]

# 마커 ID와 이벤트 이름 매핑 정의
marker_event_mapping = {
    19: 'Condition1_GreenGo',
    20: 'Condition1_RedStop',
    21: 'Condition1_RedGo',
    22: 'Condition1_GreenStop',
    23: 'Condition2_GreenGo',
    24: 'Condition2_RedStop',
    25: 'Condition2_RedGo',
    26: 'Condition2_GreenStop'
}

# 베타 대역 정의
beta_band = (13, 30)

# ERS/ERD 계산 함수 정의
def calculate_ers_erd(evoked_list, baseline_interval, event_interval, band, l_freq, h_freq):
    ers_erd_results = []
    times = evoked_list[0].times
    for evoked in evoked_list:
        filtered_evoked = evoked.copy().filter(l_freq=l_freq, h_freq=h_freq, method='iir') # 대역 통과 필터 적용
        smoothed_evoked = smooth_signal(filtered_evoked.data) # 스무딩 적용
        baseline_power = smoothed_evoked[:, (times >= baseline_interval[0]) & (times <= baseline_interval[1])] ** 2
        event_power = smoothed_evoked[:, (times >= event_interval[0]) & (times <= event_interval[1])] ** 2
        ers_erd = 100 * (np.mean(event_power, axis=1) - np.mean(baseline_power, axis=1)) / np.mean(baseline_power, axis=1)
        ers_erd_results.append(ers_erd)
    grand_average_ers_erd = np.mean(ers_erd_results, axis=0) # grand average 계산
    return grand_average_ers_erd, times

# 시간 구간 정의
baseline_interval = (-0.2, 0)
event_interval = (0, 1.0)

condition_stimulus_ers_erd = {
    'Condition1': {'GreenGo': [], 'RedStop': [], 'RedGo': [], 'GreenStop': []},
    'Condition2': {'GreenGo': [], 'RedStop': [], 'RedGo': [], 'GreenStop': []}
}

l_freq = 13 # 하한 주파수 (예: 8 Hz)
h_freq = 30 # 상한 주파수 (예: 30 Hz)

# 조건과 자극 정보 추출 함수 정의
for epoch in evokeds:
    for evoked in epoch:
        marker_id = int(evoked.comment) # 마커 ID 추출
        if marker_id in marker_event_mapping:
            condition_stimulus = marker_event_mapping[marker_id]
            condition, stimulus = condition_stimulus.split('_')
            ers_erd, times = calculate_ers_erd([evoked], baseline_interval, event_interval, beta_band, l_freq, h_freq)
            condition_stimulus_ers_erd[condition][stimulus].extend([ers_erd])
        else:
            print(f"Skipping epoch due to unrecognized marker ID: {marker_id}")
            continue

# DataFrame 생성
data = []
for condition in condition_stimulus_ers_erd:
    for stimulus in condition_stimulus_ers_erd[condition]:
        ers_erd_values = condition_stimulus_ers_erd[condition][stimulus]
        if ers_erd_values:
            for channel_ers_erd in ers_erd_values:
                for ers_erd_value in channel_ers_erd:
                    data.append({'Condition': condition, 'Stimulus': stimulus, 'ERS/ERD': ers_erd_value})

df = pd.DataFrame(data)

# 여기에 수정된 코드를 추가하세요
epochs_list = []
for evoked_list in evokeds:
    for evoked in evoked_list:
        marker_id = int(evoked.comment)
        epoch = mne.EpochsArray(evoked.data[np.newaxis, :, :], evoked.info, tmin=evoked.times[0], events=np.array([[0, 0, marker_id]]))
        epochs_list.append(epoch)

epochs = mne.concatenate_epochs(epochs_list)

# tmin과 tmax 정의
tmin, tmax = -0.2, 1.0

# baseline 정의
baseline = (-0.2, 0)

# compute ERDS maps ###########################################################
freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

# 원하는 채널 선택
selected_channels = ['Fz']  # 원하는 채널 이름을 리스트로 지정

# 선택한 채널의 인덱스 획득
selected_channel_indices = [epochs.ch_names.index(ch) for ch in selected_channels]

# 에포크 정보 출력
print("Epoch info:")
print(epochs)

# 고유한 마커 ID 출력
print("Unique marker IDs:")
print(np.unique(epochs.events[:, 2]))

# 결과 저장 디렉토리 선택
output_dir = filedialog.askdirectory(title="Select output directory")

# 선택한 디렉토리가 유효한 경우에만 진행
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

    for condition in condition_stimulus_ers_erd:
        for stimulus in condition_stimulus_ers_erd[condition]:
            event_label = f"{condition}_{stimulus}"
            event_id = list(marker_event_mapping.keys())[list(marker_event_mapping.values()).index(event_label)]
            
            epochs_ev = epochs[epochs.events[:, 2] == event_id]
            
            if len(epochs_ev) > 0:
                tfr = tfr_multitaper(epochs_ev, freqs=freqs, n_cycles=n_cycles,
                                     use_fft=True, return_itc=False, average=False,
                                     decim=2)
                tfr.crop(tmin, tmax)
                tfr.apply_baseline(baseline, mode="percent")
                
                fig, axes = plt.subplots(1, len(selected_channels) + 1, figsize=(12, 4),
                                         gridspec_kw={"width_ratios": [10] * len(selected_channels) + [1]})
                for i, ch_index in enumerate(selected_channel_indices):
                    _, c1, p1, _ = pcluster_test(tfr.data[:, ch_index, ...], tail=1, **kwargs)
                    _, c2, p2, _ = pcluster_test(tfr.data[:, ch_index, ...], tail=-1, **kwargs)

                    c = np.stack(c1 + c2, axis=2)
                    p = np.concatenate((p1, p2))
                    mask = c[..., p <= 0.05].any(axis=-1)

                    tfr.average().plot([ch_index], vmin=vmin, vmax=vmax, cmap='RdBu_r',
                                       axes=axes[i], colorbar=False, show=False, mask=mask,
                                       mask_style="mask")

                    axes[i].set_title(epochs.ch_names[ch_index], fontsize=10)
                    axes[i].axvline(0, linewidth=1, color="black", linestyle=":")
                    if i != 0:
                        axes[i].set_ylabel("")
                        axes[i].set_yticklabels("")
                fig.colorbar(axes[0].images[-1], cax=axes[-1])
                fig.suptitle(f"Beta ERDS - {event_label}")
                
                fig_filename = os.path.join(output_dir, f"erds_map_{event_label}_selected_channels.png")
                fig.savefig(fig_filename)
                print(f"ERDS map for {event_label} with selected channels saved to: {fig_filename}")
                plt.close(fig)
            else:
                print(f"No epochs found for {event_label}. Skipping ERDS map generation.")
else:
    print("No output directory selected. Aborting.")