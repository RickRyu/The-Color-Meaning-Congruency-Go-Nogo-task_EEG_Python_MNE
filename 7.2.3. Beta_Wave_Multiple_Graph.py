from tkinter import Tk, filedialog
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11  # 원하시는 글씨 크기로 설정하세요

# Initialize Tkinter window for multiple FIF file selection
root = Tk()
root.withdraw()
fif_file_paths = filedialog.askopenfilenames(title="Select multiple FIF files")
print(f"Selected FIF files: {fif_file_paths}")

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

# Pick only the selected channels and apply the 10-20 montage
selected_channels = ['FC2', 'FC6']
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

# Calculate time-frequency representations for each condition
tfrs = {}
for event_name, evokeds in all_evokeds.items():
    evoked = mne.combine_evoked(evokeds, weights='equal')
    evoked.pick_channels(selected_channels, ordered=True)
    evoked.set_eeg_reference(ref_channels='average', projection=True)
    evoked.set_montage(ten_twenty_montage)
    freqs = np.arange(13, 31, 1)
    n_cycles = freqs / 2.
    tfr = mne.time_frequency.tfr_morlet(evoked, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=3, n_jobs=1)
    tfr.apply_baseline(baseline=(None, 0), mode='logratio')
    tfrs[event_name] = tfr

# Combine time-frequency data into a single DataFrame
df = pd.concat([tfr.to_data_frame(time_format=None, long_format=True).assign(condition=condition) for condition, tfr in tfrs.items()], ignore_index=True)

# Filter to retain only beta frequency band
df = df[df.freq.between(13, 30, inclusive='both')]

# Get unique channels from the DataFrame
unique_channels = df["channel"].unique()
print("Unique channels:", unique_channels)

# Order channels for plotting using the unique channels
df["channel"] = df["channel"].cat.reorder_categories(unique_channels.tolist(), ordered=True)

# 범례 순서 지정
legend_order = ['Condition 1 - Green Go', 'Condition 1 - Red Stop', 'Condition 1 - Red Go', 'Condition 1 - Green Stop',
                'Condition 2 - Green Go', 'Condition 2 - Red Stop', 'Condition 2 - Red Go', 'Condition 2 - Green Stop']

# 범례 순서대로 condition 열 재정의
df['condition'] = pd.Categorical(df['condition'], categories=legend_order, ordered=True)

g = sns.FacetGrid(df, col="channel", margin_titles=True)
g.map(sns.lineplot, "time", "value", "condition")
axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
g.map(plt.axhline, y=0, **axline_kw)
g.map(plt.axvline, x=0, **axline_kw)
g.set(ylim=(None, 1.5))
g.set_axis_labels("Time (s)", "ERDS")
g.set_titles(col_template="{col_name}")

# 범례 위치 조정
g.add_legend(ncol=1, loc="upper right", bbox_to_anchor=(0.95, 0.95), fontsize='small', frameon=False)

# 그래프 레이아웃 조정
g.fig.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.08, wspace=0.3, hspace=0.3)

# 그래프 크기 조정
g.fig.set_size_inches(12, 6)

plt.show()