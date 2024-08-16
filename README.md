# EEG Analysis Scripts README

This repository contains a collection of Python scripts for EEG (Electroencephalography) data analysis, visualization, and processing. The scripts are numbered to indicate the order of the workflow. Below is a brief description of each script in the correct numerical order:

1. **Task_Reverse_Color.py**
   - Implements a reverse color task for EEG experiments.
   - Presents visual stimuli and records participant responses.
   - Saves task performance data in CSV format.

2. **Task_Everyday_Color.py**
   - Similar to Task_Reverse_Color.py, but implements an everyday color task.
   - Presents different visual stimuli for comparison with the reverse color task.

3. **EEG_Preprocessing.py**
   - Preprocesses raw EEG data from EDF files.
   - Applies filters, ICA for artifact removal, and creates epochs.
   - Uses AutoReject for automated bad epoch removal.

4. **FIF_Amplitude_Result.py**
   - Analyzes amplitude results from FIF files.
   - Calculates and extracts ERP (Event-Related Potential) component information.
   - Saves results in Excel format.

5. **FIF_Alpha_Synchronization_Topomap_and_Result.py**
   - Calculates and visualizes alpha band synchronization using FIF files.
   - Creates topographic maps and time series plots of ERS/ERD (Event-Related Synchronization/Desynchronization).
   - Saves results in Excel format and as image files.

5.2. **FIF_Beta_Synchronization_Topomap_and_Result.py**
   - Similar to the alpha synchronization script, but focuses on the beta frequency band.
   - Calculates and visualizes beta band synchronization.

5.3. **FIF_Theta_Synchronization_Topomap_and_Result.py**
   - Analyzes theta band synchronization from FIF files.
   - Creates topographic maps and time series plots for theta band ERS/ERD.

6.0. **Over_All_Erp_Graph.py**
   - Generates an overall ERP graph combining data from multiple FIF files.
   - Creates a comprehensive visualization of ERP components across different conditions.

6.1. **ERP_graph_by_selecting_a_specific_channel_and_specific_condition_FC1.py**
   - Focuses on creating ERP graphs for a specific channel (FC1) and selected conditions.
   - Allows for detailed analysis of ERP components in specific experimental conditions.

6.2. **Evoked_Specipic_Time_(N1,N2,P3)_Based_On_Fz.py**
   - Focuses on specific ERP components (N1, N2, P3) based on the Fz electrode.
   - Creates topographic maps for these specific time points.

6.4. **Grand_Average.py**
   - Computes and visualizes the grand average of ERP data.
   - Identifies and marks specific ERP components (N1, N2, P3) in the grand average.

7.1. **Beta_Compute_ERDS_Maps.py**
   - Computes and visualizes ERDS (Event-Related Desynchronization/Synchronization) maps for beta frequency.
   - Uses cluster-based permutation tests for statistical analysis.

7.2.1. **Beta_Wave_Each_Channel_Graph_Everyday_Color.py**
   - Creates individual channel graphs for beta wave analysis in the everyday color task condition.
   - Focuses on specific channels like FC2, FC6, Fp2, and C4.

7.2.2. **Beta_Wave_Each_Channel_Graph_Reverse_Color.py**
   - Similar to the previous script, but for the reverse color task condition.
   - Allows comparison of beta wave activity between everyday and reverse color tasks.

7.2.3. **Beta_Wave_Multiple_Graph.py**
   - Creates multiple graphs for beta wave analysis.
   - Visualizes beta band activity across different channels and conditions.

## Usage

To use these scripts:

1. Ensure you have the required dependencies installed (mne, numpy, matplotlib, pandas, seaborn, etc.).
2. Run the scripts in the order indicated by their numbering.
3. Follow the prompts to select input files and specify output locations.

## Workflow

1. Start with the task scripts (1 and 2) to collect experimental data.
2. Preprocess the raw EEG data using script 3.
3. Analyze amplitude results and synchronization patterns (scripts 4 and 5).
4. Generate various ERP visualizations (scripts 6.0 to 6.4).
5. Perform detailed beta wave analysis (scripts 7.1 to 7.2.3).

## Note

These scripts are designed to work with specific EEG data formats and experimental paradigms. Modify as needed for your specific research requirements.
