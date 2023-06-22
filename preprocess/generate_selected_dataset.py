import tqdm
import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    # Generate the list of selected data's absolute indices
    abs_indices = np.arange(2968, 3100)
    abs_indices = np.concatenate((abs_indices, np.arange(3687, 3765)))
    abs_indices = np.concatenate((abs_indices, np.arange(4873, 5290)))
    abs_indices = np.concatenate((abs_indices, np.arange(6381, 6470)))
    abs_indices = np.concatenate((abs_indices, np.arange(7697, 8254)))
    abs_indices = np.concatenate((abs_indices, np.arange(10050, 10200)))
    abs_indices = np.concatenate((abs_indices, np.arange(10308, 10409)))
    abs_indices = np.concatenate((abs_indices, np.arange(10712, 10788)))
    abs_indices = np.concatenate((abs_indices, np.arange(11056, 11152)))
    abs_indices = np.concatenate((abs_indices, np.arange(11370, 11500)))
    abs_indices = np.concatenate((abs_indices, np.arange(11717, 11774)))
    abs_indices = np.concatenate((abs_indices, np.arange(13677, 14460)))
    abs_indices = np.concatenate((abs_indices, np.arange(16400, 16500)))
    abs_indices = np.concatenate((abs_indices, np.arange(20802, 21240)))
    abs_indices = np.concatenate((abs_indices, np.arange(21737, 22209)))
    abs_indices = np.concatenate((abs_indices, np.arange(30256, 30478)))

    abs_indices = np.sort(abs_indices)

    # Read the csv file
    scenario_filename = 'scenario38.csv'
    df = pd.read_csv(scenario_filename, index_col=0)
    df_radar = pd.DataFrame(columns=df.columns)
    sync_shift = 14

    selected_seq_idx = 0
    last_abs_idx = -1

    for i, abs_idx in enumerate(tqdm.tqdm(abs_indices, total=len(abs_indices))):
        row_selected_data = df.copy().loc[df['abs_index']==abs_idx]

        # Update new sequence index and shifted radar data
        if abs_idx != last_abs_idx + 1:
            selected_seq_idx += 1
        last_abs_idx = abs_idx
        row_selected_data["selected_seq_index"] = selected_seq_idx

        row_idx = row_selected_data.index
        row_selected_data[15:19] = df.copy().iloc[row_idx+sync_shift, 15:19] # apply shift on radar data

        # Update radar index
        unit1_pwr1 = np.loadtxt(row_selected_data.unit1_pwr1.values[0], dtype=float)
        unit1_pwr2 = np.loadtxt(row_selected_data.unit1_pwr2.values[0], dtype=float)
        unit1_pwr3 = np.loadtxt(row_selected_data.unit1_pwr3.values[0], dtype=float)
        unit1_pwr4 = np.loadtxt(row_selected_data.unit1_pwr4.values[0], dtype=float)
        unit1_pwr = np.concatenate((unit1_pwr1, unit1_pwr2, unit1_pwr3, unit1_pwr4))
        radar_idx = np.argmax(unit1_pwr) // 64 + 1
        row_selected_data["radar_index"] = radar_idx

        df_radar = df_radar.append(row_selected_data)

    df_radar.reset_index(drop=True, inplace=True)
    df_radar.to_csv(f'scenario38_selected.csv')
    print(df_radar)


