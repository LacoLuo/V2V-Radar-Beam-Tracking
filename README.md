# Millimeter Wave V2V Beam Tracking using Radar: Algorithms and Real-World Demonstration
This is a python code package related to the following article: H.Luo, U. Demirhan and A. Alkhateeb, "[Millimeter Wave V2V Beam Tracking using Radar: Algorithms and Real-World Demonstration](https://wi-lab.net/research/v2v-radar-aided-beam-tracking/)", accepted to 2023 31st European Signal Processing Conference (EUSIPCO).

# Abstract of the Article
Utilizing radar sensing for assisting communication has attracted increasing interest thanks to its potential in dynamic environments. A particularly interesting problem for this approach appears in the vehicle-to-vehicle (V2V) millimeter wave and terahertz communication scenarios, where the narrow beams change with the movement of both vehicles. To address this problem, in this work, we develop a radar-aided beam-tracking framework, where a single initial beam and a set of radar measurements over a period of time are utilized to predict the future beams after this time duration. Within this framework, we develop two approaches with the combination of various degrees of radar signal processing and machine learning. To evaluate the feasibility of the solutions in a realistic scenario, we test their performance on a real-world V2V dataset. Our results indicated the importance of high angular resolution radar for this task and affirmed the potential of using radar for the V2V beam management problems.

# Code Package Content
The scripts for generating the results of the end-to-end ML solution in the paper. For the beam tracking with transmitter identification, please refer to this [branch](https://github.com/LacoLuo/V2V-Radar-Beam-Tracking/tree/Tx_Identification). This script adopts Scenario 36 of DeepSense 6G dataset.

**To implement our solution, please follow these steps:**

**Download Dataset and Code**
1. Download [Scenario 36 of DeepSense 6G dataset](https://www.deepsense6g.net/scenarios36-39/)
2. Download (or clone) the repository into a directory
3. Extract the dataset into the repository directory

**Generate Development Dataset**
1. Move the scripts in preprocess/ to the dataset directory
2. To focus on the scenarios interesting for tracking, we manually filter the data by keeping the sequences of samples with changing beam indices (e.g., the transmitter getting closer to the receiver in a different lane or the vehicles consecutively taking a turn). If you want to use other V2V scenarios (Scenario 37-39), the following scripts can help you check the beam indices and filter the dataset.
```
python plot_beam_vs_time.py
```
```
python generate_selected_dataset.py
```
3. Preprocess the radar data
```
python radar_preprocess.py \
  --out_dir unit1_selected_mat/ \
  --log
```
4. Generate the train/val/test sets
```
python sequence_generator.py \
  --csv_file scenario36_selected_preprocess.csv \
  --x_size 10 \
  --y_size 1
```

**ML Model Training**
```
python train.py \
  --trn_data_path dataset/scenario36_selected_preprocess_series_train.csv \
  --val_data_path dataset/scenario36_selected_preprocess_series_val.csv \
  --store_model_path ckpt/ \
  --normalize \
  --feature RD \
  --x_size 10
```

**ML Model Evaluation**
```
python inference.py \
  --test_data_path dataset/scenario36_selected_preprocess_series_test.csv \
  --load_model_path ckpt/ckpt_name \
  --normalize \
  --feature RD \
  --x_size 10
```

If you have any questions regarding the code and the used dataset, please write to [DeepSense 6G dataset forum](https://deepsense6g.net/forum/) or contact [Hao Luo](mailto:h.luo@asu.edu)

# License and Referencing
This code package is licensed under a[ Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> H. Luo, U. Demirhan and A. Alkhateeb, "Millimeter Wave V2V Beam Tracking using Radar: Algorithms and Real-World Demonstration," 2023 31st European Signal Processing Conference (EUSIPCO), Helsinki, Finland, 2023, pp. 740-744, doi: 10.23919/EUSIPCO58844.2023.10289752.

If you use the [DeepSense 6G dataset](https://deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb et al., "DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset," in IEEE Communications Magazine, vol. 61, no. 9, pp. 122-128, September 2023, doi: 10.1109/MCOM.006.2200730.
