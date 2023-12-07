import tqdm
import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    # Generate the list of selected data's absolute indices
    abs_indices = np.arange(2960, 3100)
    abs_indices = np.concatenate((abs_indices, np.arange(3680, 3780)))
    abs_indices = np.concatenate((abs_indices, np.arange(4700, 5400)))
    abs_indices = np.concatenate((abs_indices, np.arange(6350, 6500)))
    abs_indices = np.concatenate((abs_indices, np.arange(10000, 10200)))
    abs_indices = np.concatenate((abs_indices, np.arange(11000, 11200)))
    abs_indices = np.concatenate((abs_indices, np.arange(13650, 13800)))
    abs_indices = np.concatenate((abs_indices, np.arange(16400, 16500)))
    abs_indices = np.concatenate((abs_indices, np.arange(20800, 21200)))
    abs_indices = np.concatenate((abs_indices, np.arange(21700, 22132)))
    abs_indices = np.concatenate((abs_indices, np.arange(23800, 24050)))
    abs_indices = np.concatenate((abs_indices, np.arange(30270, 30550)))

    abs_indices = np.sort(abs_indices)

    # Read the csv file
    scenario_filename = 'scenario36.csv'
    df = pd.read_csv(scenario_filename)
    df_radar = pd.DataFrame(columns=df.columns)

    selected_seq_idx = 0
    last_abs_idx = -1

    for i, abs_idx in enumerate(tqdm.tqdm(abs_indices, total=len(abs_indices))):
        row_selected_data = df.copy().loc[df['abs_index']==abs_idx]

        # Update new sequence index 
        if abs_idx != last_abs_idx + 1:
            selected_seq_idx += 1
        last_abs_idx = abs_idx
        row_selected_data["selected_seq_index"] = selected_seq_idx

        # Update radar index
        unit1_pwr1 = np.loadtxt(row_selected_data.unit1_pwr1.values[0], dtype=float)
        unit1_pwr2 = np.loadtxt(row_selected_data.unit1_pwr2.values[0], dtype=float)
        unit1_pwr3 = np.loadtxt(row_selected_data.unit1_pwr3.values[0], dtype=float)
        unit1_pwr4 = np.loadtxt(row_selected_data.unit1_pwr4.values[0], dtype=float)
        unit1_pwr = np.concatenate((unit1_pwr1, unit1_pwr2, unit1_pwr3, unit1_pwr4))
        radar_idx = np.argmax(unit1_pwr) // 64 + 1
        row_selected_data["radar_index"] = radar_idx

        df_radar = pd.concat([df_radar, row_selected_data])

    df_radar.reset_index(drop=True, inplace=True)
    df_radar.to_csv(f'scenario36_selected.csv')
    print(df_radar)