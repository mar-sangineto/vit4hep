import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.lorenzetti.utils import load_data
from experiments.logger import LOGGER


class LorenzettiDataset(Dataset):
    """
    """

    def __init__(
        self,
        hdf5_file,
        transform=None,
        return_us=False,
        dtype=torch.float32,
        rank=0,
        bin_edges=None,
    ):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            transform: list of transformations applied as preprocessing
            return_us: whether to return the extra layer energy conditions (used for the energy network)
            dtype: data type for the voxels and conditions
            rank: rank of the process
            bin_edges: cumulative voxel-count boundaries per layer (len = n_layers + 1);
                defaults to the original 17-layer edges if not given
        """

        if bin_edges is None:
            bin_edges = [0, 2048, 3072, 4352, 4792, 5208, 5464, 5496, 5528, 5560, 5816, 6072, 6136, 6200, 6240, 6272, 6304, 6320]
        self.bin_edges = np.array(bin_edges)
        n_layers = len(self.bin_edges) - 1

        self.data_dict = load_data(hdf5_file, n_layers=n_layers)

        for key in self.data_dict.keys():
            self.data_dict[key] = torch.tensor(self.data_dict[key]).flatten(start_dim=1)

        self.return_us = return_us
        self.transform = transform
        self.dtype = dtype

        # apply preprocessing
        if self.transform:
            for fn in self.transform:
                if fn.__class__.__name__ == "LorenzettiNormalizeLayerEnergy":
                    fn.bin_edges = self.bin_edges
                self.data_dict = fn(self.data_dict, rank=rank)

        if self.return_us:
            self.layers = self.data_dict["extra_dims"]
            self.energy = self.data_dict["energy"]
        else:
            # store data as 2d array with flattened layers: (dataset size, # of voxels)
            # geometric information is recovered from the bin edges
            # and the model config file
            self.layers = torch.hstack(
                (
                    [self.data_dict[f"layer_{i}"] for i in range(n_layers)]
                ),
            )
            self.layers = self.layers.unsqueeze(1)
            self.energy = torch.hstack(
                (self.data_dict["energy"], self.data_dict["extra_dims"]),
            )

        self.layers = self.layers.to(dtype)
        self.energy = self.energy.to(dtype)

        self.min_bounds = self.layers.min()
        self.max_bounds = self.layers.max()

        LOGGER.info(f"datasets: loaded data with shape {(*self.layers.shape,)}")
        LOGGER.info(f"datasets: boundaries of dataset are ({self.min_bounds}, {self.max_bounds})")

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        return self.layers[idx], self.energy[idx]
