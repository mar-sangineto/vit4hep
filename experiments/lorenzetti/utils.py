import h5py
import numpy as np


def load_data(data_file, n_layers=17):
    # Layer energy is in MeV(1e6); truth energy is in GeV(1e9)
    # Electrons simulation was done from 2GeV to 7 TeV
    
    data={}
    shift=5000 # 5GeV for layers to avoid 0 and negative values
    
    with h5py.File(data_file, "r") as full_file:
        
        for i in range(n_layers): #assumes that the first items are the layers
            
            layer_name = list(full_file.keys())[i]
            
            layer_data = full_file[layer_name][:]
            
            data[f"layer_{i}"] = np.log10(layer_data+shift) # log is to deal with a large range of values
            
        energy = full_file["energy"][:]
        data["energy"] = np.log10(energy) # truth particle wont have negative energy
        
        ### Next step : implement eta and phi. Initial tests with just energy so that it works as it is working in calogan
#         # to integrate C
#         if "eta" in full_file:
#             data["eta"] = full_file["eta"][:]
#         if "phi" in full_file:
#             data["phi"] = full_file["phi"][:] # IDEA: Integrate as an output from an MLP
            
#         elif "truth_kinematics" in full_file:
#             data["eta"] = full_file["truth_kinematics"][:][:, 0]
#             data["phi"] = full_file["truth_kinematics"][:][:, 1]
    
    return data
