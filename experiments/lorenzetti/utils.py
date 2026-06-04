import h5py
import numpy as np


def load_data(data_file, n_layers=17):
    data={}
    
    with h5py.File(data_file, "r") as full_file:
        for i in range(n_layers):
            layer_name = f"layer_{i}"
            
            layer_data = full_file[layer_name][:]
            
            data[layer_name] = np.log10(layer_data+1)
            
        energy = full_file["energy"][:]
        data["energy"] = np.log10(energy)
        
        # to integrate C
        if "eta" in full_file:
            data["eta"] = full_file["eta"][:]
        if "phi" in full_file:
            data["phi"] = full_file["phi"][:] # Integrate as an output from an MLP
            
        elif "truth_kinematics" in full_file:
            data["eta"] = full_file["truth_kinematics"][:][:, 0]
            data["phi"] = full_file["truth_kinematics"][:][:, 1]
    
    return data
