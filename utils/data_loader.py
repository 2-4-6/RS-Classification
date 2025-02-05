"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines Data Loader for training, validation and test partitions using torch and numpy
"""

import numpy as np
import torch


class DataLoader():
    """
    THIS CLASS INITIALIZES THE TRAINING, VALIDATION, AND TEST DATA LOADERS
    """
    def __init__(self, train_val_reader=None, test_reader=None, validation_split=0.2):
        '''
        THIS FUNCTION INITIALIZES THE DATA LOADER.
        :param train_val_reader: data reader inherited from torch.utils.data.Dataset for train/validation partitions
        :param test_reader: data reader inherited from torch.utils.data.Dataset for test partition
        :param validation_split: the rate of validation data (between 0.0-1.0) to split data into 2 partitions for training and validation
        :return: None
        '''

        if train_val_reader is not None:
            indices = list(range(len(train_val_reader)))
            np.random.RandomState(420).shuffle(indices)
            split = int(np.floor(validation_split * len(train_val_reader)))
            train_indices, val_indices = indices[split:], indices[:split]
            # Splitting the data into training and validation partitions
            self.train_dataset = torch.utils.data.Subset(train_val_reader, train_indices)
            self.val_dataset = torch.utils.data.Subset(train_val_reader, val_indices)

        if test_reader is not None:
            self.test_dataset=test_reader

        if test_reader is None and train_val_reader is None:
            raise

    def get_train_loader(self, batch_size=4, num_workers=2):
        '''
        THIS FUNCTION RETURNS THE TRAINING DATA LOADER.
        :param batch_size: number of batches while loading the training data
        :param num_workers: number of workers to operating in parallel
        :return: torch.utils.data.DataLoader
        '''
        print('INFO: Training data loader initialized.')
        return  torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers)


    def get_validation_loader(self, batch_size, num_workers):
        '''
        THIS FUNCTION RETURNS THE VALIDATION DATA LOADER.
        :param batch_size: number of batches while loading the validation data
        :param num_workers: number of workers to operating in parallel
        :return: torch.utils.data.DataLoader
        '''
        print('INFO: Validation data loader initialized.')
        return  torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers)

    def get_test_loader(self, batch_size, num_workers):
        '''
        THIS FUNCTION RETURNS THE TEST DATA LOADER.
        :param batch_size: number of batches while loading the test data
        :param num_workers: number of workers to operating in parallel
        :return: torch.utils.data.DataLoader
        '''
        print('INFO: Test data loader initialized.')
        return  torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)

