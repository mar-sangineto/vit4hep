# standard python libraries
import math
import os
import time
import warnings

import h5py
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import experiments.lorenzetti.transforms as transforms
from experiments.base_experiment import BaseExperiment
from experiments.calo_utils.us_evaluation.classifier import eval_ui_dists
from experiments.calo_utils.us_evaluation.plots import plot_ui_dists
from experiments.lorenzetti.datasets import LorenzettiDataset
from experiments.lorenzetti.evaluate import eval_lorenzetti_lowlevel

# Other functions of project
from experiments.logger import LOGGER


class Lorenzetti(BaseExperiment):
    """
    Train a generative model on the Lorenzetti dataset.
    Structure:

    init_data()          : Read in data parameters and prepare the datasets
    _init_dataloader()   : Create the dataloaders for training and validation
    _batch_loss()        : Calls the model's batch_loss function
    sample_us()          : Sample energy ratios from the energy model
    sample_n()           : Generate n_samples from the trained model, either energy ratios or full normalized showers
    sample()             : First generate full shower, then make plots and evaluate
    eval_sample()        : Evaluate saved samples without re-generating them
    save_sample()        : Save generated samples in the correct format
    load_sample()        : Load generated samples from the saved format
    load_energy_model()  : Load an external energy model, used if sample_us==True
    """

    def init_data(self):
        self.hdf5_train = self.cfg.data.training_file
        self.hdf5_test = self.cfg.data.test_file
        self.return_us = self.cfg.data.return_us
        # Optional (energy model only): condition on each event's own (eta, phi)
        # in addition to incident energy, instead of training blind across
        # whatever eta/phi span the training file covers. See datasets.py and
        # sample_n() below. Defaults to False so every existing config is unaffected.
        self.use_eta_phi_condition = self.cfg.data.get("use_eta_phi_condition", False)
        self.transforms = []
        
        self.n_layers = self.cfg.data.n_layers
        if "list_shape" in self.cfg.model:
            self.bin_edges = [0]
            for shape in self.cfg.model.list_shape:
                self.bin_edges.append(self.bin_edges[-1] + math.prod(shape))
            assert len(self.cfg.model.list_shape) == self.n_layers, (
                f"model.list_shape has {len(self.cfg.model.list_shape)} layers but "
                f"data.n_layers={self.n_layers} -- keep them in sync"
            )
        else:
            # e.g. the energy model, whose model config has no list_shape
            self.bin_edges = list(self.cfg.data.bin_edges)
            assert len(self.bin_edges) - 1 == self.n_layers, (
                f"data.bin_edges has {len(self.bin_edges) - 1} layers but "
                f"data.n_layers={self.n_layers} -- keep them in sync"
            )

        LOGGER.info("init_data: preparing model training")
        for name, kwargs in self.cfg.data.transforms.items():
            if "FromFile" in name:
                kwargs["model_dir"] = self.cfg.run_dir
            self.transforms.append(getattr(transforms, name)(**kwargs))
        LOGGER.info("init_data: list of preprocessing steps:")
        for _, transform in enumerate(self.transforms):
            LOGGER.info(f"{transform.__class__.__name__}")

        self.train_dataset = LorenzettiDataset(
            self.hdf5_train,
            transform=self.transforms,
            return_us=self.return_us,
            dtype=self.dtype,
            rank=self.rank,
            bin_edges=self.bin_edges,
            use_eta_phi_condition=self.use_eta_phi_condition,
        )

        self.val_dataset = LorenzettiDataset(
            self.hdf5_train,
            transform=self.transforms,
            return_us=self.return_us,
            dtype=self.dtype,
            rank=self.rank,
            bin_edges=self.bin_edges,
            use_eta_phi_condition=self.use_eta_phi_condition,
        )

        self.layer_boundaries = self.train_dataset.bin_edges

    def _init_dataloader(self):
        self.batch_size = (
            self.cfg.training.batchsize // self.world_size
            if self.world_size > 1
            else self.cfg.training.batchsize
        )

        if self.world_size > 1:
            self.train_dist_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self.val_dist_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        else:
            self.train_dist_sampler = None
            self.val_dist_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_dist_sampler,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_dist_sampler,
            pin_memory=True,
        )

        LOGGER.info(
            f"init_dataloader: created training dataloader with {len(self.train_loader)} batches"
        )
        LOGGER.info(
            f"init_dataloader: created validation dataloader with {len(self.val_loader)} batches"
        )

    def _batch_loss(self, data):
        return self.model._batch_loss(data)

    @torch.inference_mode()
    def sample_n(self):
        self.model.eval()

        t_0 = time.time()

        # Sample incident energies from the *actual* training range, not a hardcoded
        # constant. This used to be `torch.rand(...) * 99 + 1` -- a leftover CaloGAN
        # 1-100 GeV assumption -- so a model trained on any other range (e.g. 150-200
        # GeV) was evaluated on conditioning energies 100% outside its training
        # distribution, causing wild extrapolation and near-total generative failure.
        # Read the same file LorenzettiScaleEnergyFromFile calibrates its bounds from
        # (self.hdf5_train, via the first-constructed train_dataset in init_data()).
        with h5py.File(self.hdf5_train, "r") as f:
            e_train = f["energy"][:]
        e_min_gev, e_max_gev = float(e_train.min()), float(e_train.max())
        Einc = torch.rand((self.cfg.n_samples, 1)) * (e_max_gev - e_min_gev) + e_min_gev
        Einc = Einc.to(device=self.device, dtype=self.dtype)

        samples_dict = {}
        samples_dict["energy"] = Einc
        # transform Einc to basis used in training
        for fn in self.transforms:
            if hasattr(fn, "cond_transform"):
                samples_dict = fn(samples_dict)

        transformed_cond = samples_dict["energy"]
        if self.use_eta_phi_condition:
            # Draw synthetic per-sample (eta, phi) uniformly within the training
            # file's own truth_kinematics range -- exactly the same "resample
            # within the training distribution" approach already used for Einc
            # above, just extended to the two extra conditioning dims. Left
            # untransformed to match datasets.py's forward-pass convention (no
            # LogEnergy/ScaleEnergy applied to eta/phi).
            with h5py.File(self.hdf5_train, "r") as f:
                truth_kinematics = f["truth_kinematics"][:]
            eta_min_val, eta_max_val = float(truth_kinematics[:, 0].min()), float(truth_kinematics[:, 0].max())
            phi_min_val, phi_max_val = float(truth_kinematics[:, 1].min()), float(truth_kinematics[:, 1].max())
            Eta = torch.rand((self.cfg.n_samples, 1)) * (eta_max_val - eta_min_val) + eta_min_val
            Phi = torch.rand((self.cfg.n_samples, 1)) * (phi_max_val - phi_min_val) + phi_min_val
            Eta = Eta.to(device=self.device, dtype=self.dtype)
            Phi = Phi.to(device=self.device, dtype=self.dtype)
            transformed_cond = torch.hstack([transformed_cond, Eta, Phi])

        batchsize_sample = self.cfg.training.batchsize_sample
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batchsize_sample, shuffle=False
        )

        # sample u_i's if self is a shape model
        if self.cfg.model_type == "shape":
            if self.cfg.sample_us:
                u_samples = self.sample_us(transformed_cond_loader)
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else:  # optionally use truth us
                transformed_cond = LorenzettiDataset(
                    self.hdf5_test,
                    transform=self.transforms,
                    return_us=self.return_us,
                    bin_edges=self.bin_edges,
                ).energy.to(self.device)

            # concatenate with Einc
            transformed_cond_loader = DataLoader(
                dataset=transformed_cond, batch_size=batchsize_sample, shuffle=False
            )

        sample = torch.vstack([self.model.sample_batch(c).cpu() for c in transformed_cond_loader])

        t_1 = time.time()
        sampling_time = t_1 - t_0
        LOGGER.info(f"sample_n: Finished generating {len(sample)} samples after {sampling_time} s.")
        return sample, transformed_cond.cpu()

    def sample_us(self, transformed_cond_loader):
        """Sample u_i's from the energy model"""
        # load energy model
        self.load_energy_model()

        # sample us
        t_0 = time.time()
        u_samples = torch.vstack(
            [self.energy_model.sample_batch(c) for c in transformed_cond_loader]
        )
        t_1 = time.time()
        LOGGER.info(
            f"sample_us: Finished generating {len(u_samples)} energy samples after {t_1 - t_0} s."
        )

        u_samples_dict = {}
        u_samples_dict["extra_dims"] = u_samples
        for fn in self.energy_model_transforms[::-1]:
            if hasattr(fn, "u_transform"):
                fn.layer_keys = ["extra_dims"]
                u_samples_dict = fn(u_samples_dict, rev=True)
        for fn in self.transforms:
            if hasattr(fn, "u_transform"):
                fn.layer_keys = ["extra_dims"]
                u_samples_dict = fn(u_samples_dict)

        return u_samples_dict["extra_dims"].to(self.dtype)

    def sample(self):
        LOGGER.info("sample: generating samples")
        samples, conditions = self.sample_n()

        if self.cfg.model_type == "energy":
            reference = LorenzettiDataset(
                self.hdf5_test,
                transform=self.transforms,  # TODO: Or, apply NormalizeEByLayer popped from model transforms
                return_us=self.return_us,
                bin_edges=self.bin_edges,
            )
            samples_dict = {}
            samples_dict["extra_dims"] = samples
            # `conditions` is the full C vector sample_n() fed the network -- just
            # [energy] normally, but [energy, eta, phi] when
            # data.use_eta_phi_condition=true (see sample_n()). The reverse-transform
            # chain below (LorenzettiNormalizeLayerEnergy etc.) only expects the
            # energy scalar in data_dict["energy"], so slice it out here regardless
            # of conditions' width -- mirrors how the model_type != "energy" (shape)
            # branch already does `conditions[:, 0]` / `conditions[:, 1:]` below.
            samples_dict["energy"] = conditions[:, :1]
            reference_dict = {}
            reference_dict["extra_dims"] = reference.layers
            reference_dict["energy"] = reference.energy
            # postprocess
            for fn in self.transforms[::-1]:
                if fn.__class__.__name__ == "NormalizeLayerEnergyGAN":
                    break  # this might break plotting
                fn.layer_keys = ["extra_dims"]
                samples_dict = fn(samples_dict, rev=True)
                reference_dict = fn(reference_dict, rev=True)

            samples = samples_dict["extra_dims"]
            reference = reference_dict["extra_dims"]
            # clip u_i's (except u_0) to [0,1]
            samples[:, 1:] = torch.clip(samples[:, 1:], min=0.0, max=1.0)
            reference[:, 1:] = torch.clip(reference[:, 1:], min=0.0, max=1.0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_ui_dists(
                    samples.detach().cpu().numpy(),
                    reference.detach().cpu().numpy(),
                    cfg=self.cfg,
                )
                eval_ui_dists(
                    samples.detach().cpu().numpy(),
                    reference.detach().cpu().numpy(),
                    cfg=self.cfg,
                )
        else:
            bin_edges = self.bin_edges
            samples = samples.reshape(samples.shape[0], -1)

            samples_dict = {}
            samples_dict["energy"] = conditions[:, 0]
            samples_dict["extra_dims"] = conditions[:, 1:]
            for i in range(self.n_layers):
                samples_dict[f"layer_{i}"] = samples[:, bin_edges[i] : bin_edges[i+1]]
            # postprocess
            for fn in self.transforms[::-1]:
                samples_dict = fn(samples_dict, rev=True)

            samples = (
                torch.hstack(
                    (
                        [samples_dict[f"layer_{i}"] for i in range(self.n_layers)]
                    ),
                )
                .detach()
                .cpu()
                .numpy()
            )

            if self.cfg.save:
                self.save_sample(samples, conditions, name=f"_{self.cfg.run_idx}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eval_lorenzetti_lowlevel(samples, self.cfg)

    def eval_sample(self, dirname=""):
        samples, energies = self.load_sample(dirname=dirname)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_lorenzetti_lowlevel(samples, self.cfg)

    def save_sample(self, sample, energies, name=""):
        """Save sample in the correct format"""
        save_file = h5py.File(self.cfg.run_dir + f"/samples{name}.hdf5", "w")
        save_file.create_dataset("incident_energies", data=energies, compression="gzip")
        save_file.create_dataset("showers", data=sample, compression="gzip")
        save_file.close()

    def load_sample(self, dirname=""):
        """Load sample from the correct format"""
        if dirname == "":
            dirname = self.cfg.run_dir + f"/samples_{self.cfg.run_idx}.hdf5"
        LOGGER.info(f"load_sample: loading samples from {dirname}")
        load_file = h5py.File(dirname, "r")
        energies = load_file["incident_energies"][:]
        sample = load_file["showers"][:]
        load_file.close()
        return sample, energies

    def load_energy_model(self):
        # initialize model
        energy_model_cfg = OmegaConf.load(self.cfg.energy_model + "config.yaml")
        # get transforms
        self.energy_model_transforms = []
        for name, kwargs in energy_model_cfg.data.transforms.items():
            if "FromFile" in name:
                kwargs["model_dir"] = energy_model_cfg.run_dir
            self.energy_model_transforms.append(getattr(transforms, name)(**kwargs))

        self.energy_model = instantiate(energy_model_cfg.model)
        num_parameters = sum(p.numel() for p in self.energy_model.parameters() if p.requires_grad)
        LOGGER.info(
            f"Instantiated energy model {type(self.energy_model.net).__name__} with {num_parameters} learnable parameters"
        )
        model_path = os.path.join(energy_model_cfg.run_dir, "models", "model_run0.pt")
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["model"]
            LOGGER.info(f"Loading energy model from {model_path}")
            self.energy_model.load_state_dict(state_dict)
        except FileNotFoundError as err:
            raise ValueError(f"Cannot load model from {model_path}") from err

        self.energy_model.to(self.device, dtype=self.dtype)
