"""
Each generation contains 8 simulations. 
For example, generation 1 consists of files ranging from 0 to 7.

This code aims at copying 1 vtu for the highest reward simulation of each generation.
It then groups them in a separate folder names "video", with .vtu files named simu_XXXXX.


"""
import os
def get_vtus(input_folder, ngen=8):
    
    rewards_file = input_folder + "/all_rewards.txt"
    # The all_reward file looks like this : 
    # ind_simu    reward
    # 6 -0.13188
    # 1 -0.0237
    # 0 -0.04872
    # 7 -0.08209
    # 3 -0.20675
    # 2 -0.15711
    # 4 -0.22998
    
    #First, we get the simulation of best reward for each generation

    best_simulations = {}
    with open(rewards_file, "r") as f:
        for line in f.readlines()[1:]:
            ind_simu, reward = line.split()
            gen = int(ind_simu) // ngen
            if gen not in best_simulations or float(reward) > best_simulations[gen][1]:
                best_simulations[gen] = (ind_simu, float(reward))

    # Now we have the best simulation for each generation
    # Next, we need to copy the corresponding .vtu files

    output_folder = input_folder + '/video'
    os.makedirs(output_folder, exist_ok=True)

    for gen, (ind_simu, reward) in best_simulations.items():
        vtu_file = input_folder + f'/0/{ind_simu}/vtu/simu_00100.vtu'
        cmd = f'cp {vtu_file} {output_folder}'
        os.system(cmd)
        os.system(f'mv {output_folder}/simu_00100.vtu {output_folder}/simu_{str(int(ind_simu)//ngen)}.vtu')

    return output_folder

if __name__ == "__main__":
    print(get_vtus("airfoil_17-31-12"))