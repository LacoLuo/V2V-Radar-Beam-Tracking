import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read the csv file
    csv_file = 'scenario36.csv'
    df = pd.read_csv(csv_file)
    
    beam_indices = list()
    abs_indices = list()
    for i, row in enumerate(tqdm.tqdm(df.itertuples(index=False), total=len(df))):
        # Get the receive power of each direction
        unit1_pwr1 = np.loadtxt(row.unit1_pwr1, dtype=float)
        unit1_pwr2 = np.loadtxt(row.unit1_pwr2, dtype=float)
        unit1_pwr3 = np.loadtxt(row.unit1_pwr3, dtype=float)
        unit1_pwr4 = np.loadtxt(row.unit1_pwr4, dtype=float)

        # Estimate the optimal beam
        unit1_pwr = np.concatenate((unit1_pwr1, unit1_pwr2, unit1_pwr3, unit1_pwr4))
        beam_index = np.argmax(unit1_pwr) + 1
        beam_indices.append(beam_index)
        abs_indices.append(row.abs_index)

    picked_abs_indices = np.asarray([2960, 3100, 
                                 3680, 3780, 
                                 4700, 5400, 
                                 6350, 6500,  
                                 10000, 10200, 
                                 11000, 11200, 
                                 13650, 13800, 
                                 16400, 16500,
                                 20800, 21200,
                                 21700, 22132,
                                 23800, 24050,
                                 30270, 30550])
    fig, ax = plt.subplots()
    ax.plot(abs_indices, beam_indices, c='blue')
    abs_indices = np.asarray(abs_indices)
    beam_indices = np.asarray(beam_indices)
    
    picked_indices = []
    for index in picked_abs_indices:
        print(index)
        picked_indices.append(np.where(abs_indices==index)[0][0])
    print(picked_indices)
   
    for i in range(len(picked_indices)//2):
        indices = np.arange(picked_indices[2*i], picked_indices[2*i+1], dtype=int)
        ax.plot(abs_indices[indices], beam_indices[indices], c='red')
    
    plt.savefig('beam_vs_time.png', dpi=400)
    plt.show()
