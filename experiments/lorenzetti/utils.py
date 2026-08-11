import h5py
import numpy as np


def load_data(data_file, n_layers=17):
    # Layer energy is stored in MeV, truth/incident energy in GeV. Converting
    # layers to GeV here makes GeV the single unit for the rest of the
    # pipeline (transforms, u_i ratios, evaluation) -- without this, ratios
    # like u_0 = layer_energy / incident_energy come out ~1000x too large.
    # See job_batchs/logs/lorenzetti_shape_training_registry.md.
    # Electrons simulation was done from 2GeV to 7 TeV

    data={}
    shift=5000 # 5GeV for layers to avoid 0 and negative values
    MEV_PER_GEV = 1000.0

    with h5py.File(data_file, "r") as full_file:

        for i in range(n_layers): #assumes that the first items are the layers

            layer_name = list(full_file.keys())[i]

            layer_data = full_file[layer_name][:] / MEV_PER_GEV

            #data[f"layer_{i}"] = np.log10(layer_data+shift) # log is to deal with a large range of values
            data[f"layer_{i}"] = layer_data
            
        energy = full_file["energy"][:]
        #data["energy"] = np.log10(energy) # truth particle wont have negative energy
        data["energy"] = energy

        # Per-event eta/phi, when present, for optional eta/phi conditioning (see
        # data.use_eta_phi_condition in the energy-model config / datasets.py /
        # experiment.py::sample_n()). Left as plain, untransformed dict entries --
        # no transform in transforms.py references the "eta"/"phi" keys, so they
        # pass through the forward/reverse transform chain unchanged.
        # reshaped to (N, 1) to match "energy"'s 2D convention -- datasets.py
        # calls .flatten(start_dim=1) on every data_dict entry.
        if "eta" in full_file:
            data["eta"] = full_file["eta"][:].reshape(-1, 1)
        if "phi" in full_file:
            data["phi"] = full_file["phi"][:].reshape(-1, 1)
        elif "truth_kinematics" in full_file:
            data["eta"] = full_file["truth_kinematics"][:][:, 0].reshape(-1, 1)
            data["phi"] = full_file["truth_kinematics"][:][:, 1].reshape(-1, 1)

    return data
