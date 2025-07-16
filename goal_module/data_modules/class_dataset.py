import os
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .goal_dataset import DummyGoalDataset
from .load_dataset import load_dataset_train, load_dataset_test


class ClassificationDataModule(LightningDataModule):
    def __init__(self, dataset, data_dir, train_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = dataset
        self.data_dir = data_dir
        self.train_config = train_config
        self.data_config = data_config

        # Data-specific variables - fill with setup function;
        self.transform = None
        self.train_data, self.val_data, self.test_data = None, None, None

    def setup(self, stage=None):
        if self.dataset == "dummy_goal":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                dummy_full = DummyGoalDataset(train=True, n_samples=1000)
                self.train_data, self.val_data = random_split(dummy_full,
                                                              [int(0.9*len(dummy_full)),
                                                               len(dummy_full) - int(0.9*len(dummy_full))])

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_data = DummyGoalDataset(self.data_dir, train=False, n_samples=200)
        elif self.dataset == "pogany":
            if stage == "fit" or stage is None:
                dataset = load_dataset_train(self.data_config)
                self.train_data, self.val_data = random_split(dataset,
                                                              [int(0.9*len(dataset)),
                                                               len(dataset) - int(0.9*len(dataset))])
            if stage == "test" or stage is None:
                self.test_data = load_dataset_test(self.data_config)
            
        else:
            raise ValueError(
                "[Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=self.train_config["num_workers"],
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            persistent_workers=True
        )


class DCADataModule(LightningDataModule):
    def __init__(self, dataset, data_dir, train_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = dataset
        self.data_dir = data_dir
        self.train_config = train_config
        self.data_config = data_config

        # Data-specific variables - fill with setup function;
        self.transform = None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.test_sampler = None
        self.dca_partial_eval_indices = None

    def set_dca_eval_sample_indices(self):
        if self.dca_partial_eval_indices is None:
            self.dca_partial_eval_indices = np.random.choice(
                list(range(len(self.test_data))),
                self.train_config["n_dca_samples"],
                replace=False,
            )


    def setup(self, stage=None):

        if self.dataset == "dummy_goal":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                dummy_full = DummyGoalDataset(self.data_dir, train=True, n_samples=1000)
                self.train_data, self.val_data = random_split(dummy_full,
                                                              [int(0.9*len(dummy_full)),
                                                               len(dummy_full) - int(0.9*len(dummy_full))])

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_data = DummyGoalDataset(self.data_dir, train=False, n_samples=200)

                self.set_dca_eval_sample_indices()
                self.partial_test_sampler = torch.utils.data.SubsetRandomSampler(
                    self.dca_partial_eval_indices
                )

        else:
            raise ValueError(
                "[DCA Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")



    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=self.train_config["num_workers"],
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            sampler=self.partial_test_sampler,
            drop_last=False,
            persistent_workers=True
        )

