from tkinter import Tk, filedialog
from mne import read_evokeds
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

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
       filtered_evoked = evoked.copy().filter(l_freq=l_freq, h_freq=h_freq, method='iir')  # 대역 통과 필터 적용
       smoothed_evoked = smooth_signal(filtered_evoked.data)  # 스무딩 적용
       baseline_power = smoothed_evoked[:, (times >= baseline_interval[0]) & (times <= baseline_interval[1])] ** 2
       event_power = smoothed_evoked[:, (times >= event_interval[0]) & (times <= event_interval[1])] ** 2
       
       ers_erd = 100 * (np.mean(event_power, axis=1) - np.mean(baseline_power, axis=1)) / np.mean(baseline_power, axis=1)
       ers_erd_results.append(ers_erd)
   ers_erd_results = np.array(ers_erd_results)
   return ers_erd_results, times

# 시간 구간 정의
baseline_interval = (-0.2, 0)
event_interval = (0, 1.5)  # 변경된 부분

condition_stimulus_ers_erd = {
   'Condition1': {'GreenGo': [], 'RedStop': [], 'RedGo': [], 'GreenStop': []},
   'Condition2': {'GreenGo': [], 'RedStop': [], 'RedGo': [], 'GreenStop': []}
}

l_freq = 13  # 하한 주파수 (예: 8 Hz)
h_freq = 30  # 상한 주파수 (예: 30 Hz)

# 조건과 자극 정보 추출 함수 정의
for epoch in evokeds:
   for evoked in epoch:
       marker_id = int(evoked.comment)  # 마커 ID 추출
       if marker_id in marker_event_mapping:
           condition_stimulus = marker_event_mapping[marker_id]
           condition, stimulus = condition_stimulus.split('_')
           ers_erd, times = calculate_ers_erd([evoked], baseline_interval, event_interval, beta_band, l_freq, h_freq)
           condition_stimulus_ers_erd[condition][stimulus].extend(ers_erd)
       else:
           print(f"Skipping epoch due to unrecognized marker ID: {marker_id}")
           continue

# 시각화 및 결과 저장
root = Tk()
root.withdraw()
file_path = filedialog.asksaveasfilename(title="Specify the base path for saving ERS/ERD results", defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
root.destroy()

if file_path:
    file_path_prefix = file_path.rsplit('.', 1)[0]
    for condition in condition_stimulus_ers_erd:
        for stimulus in condition_stimulus_ers_erd[condition]:
            ers_erd_results = condition_stimulus_ers_erd[condition][stimulus]
            if ers_erd_results:
                times = evokeds[0][0].times  # 시간 정보 추출
                mean_ers_erd = np.mean(ers_erd_results, axis=0)
                print(f"Mean ERS/ERD shape: {mean_ers_erd.shape}")  # 추가된 부분
                
                # 토포맵 그리기
                fig, ax = plt.subplots(figsize=(6, 6))
                im, _ = mne.viz.plot_topomap(mean_ers_erd, evokeds[0][0].info, axes=ax, show=False, cmap='RdBu_r')
                # 색상 범위를 데이터의 최소값과 최대값으로 설정
                min_value = np.min(mean_ers_erd)
                max_value = np.max(mean_ers_erd)
                im.set_clim(min_value, max_value)
                ax.set_title(f'Mean Beta ERS/ERD - {condition} - {stimulus}')
                fig.colorbar(im)
                
                # 토포맵 저장
                topomap_save_path = f"{file_path_prefix}-{condition}-{stimulus}-topomap.png"
                fig.savefig(topomap_save_path)
                plt.close(fig)
                print(f"Saved mean beta ERS/ERD topomap for {condition} - {stimulus} to {topomap_save_path}")
            
                
                # ERS/ERD 결과 엑셀 파일로 저장
                save_path = f"{file_path_prefix}-{condition}-{stimulus}.xlsx"
                ers_erd_df = pd.DataFrame(ers_erd_results, columns=evokeds[0][0].info['ch_names']).T
                ers_erd_df.columns = [f'Epoch {i+1}' for i in range(len(ers_erd_results))]
                ers_erd_df.to_excel(save_path, index=True)
                print(f"Saved ERS/ERD results for {condition} - {stimulus} to {save_path}")