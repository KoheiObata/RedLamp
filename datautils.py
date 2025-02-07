from loaders.load import load_data
from loaders.loader_aug import Loader_aug, Loader_aug_batch


import numpy as np



def load_dataloader_aug(dataparams, anomaly_types=None, anomaly_types_for_dict=None, group='train'):
    dataset_path = './dataset'
    if group=='train':
        train_dataset, val_dataset = load_data(dataset=dataparams.dataset,
                    group='train',
                    entities=dataparams.entities,
                    downsampling=dataparams.downsampling,
                    min_length=None,
                    root_dir=dataset_path,
                    verbose=True,
                    validation=True)

        train_dataloader = Loader_aug(dataset=train_dataset,
                                    batch_size=dataparams.batch_size,
                                    window_size=dataparams.window_size,
                                    window_step=dataparams.window_step,
                                    anomaly_types=dataparams.anomaly_types,
                                    anomaly_types_for_dict=anomaly_types_for_dict,
                                    min_range=dataparams.min_range,
                                    min_features=dataparams.min_features,
                                    max_features=dataparams.max_features,
                                    fast_sampling=False,
                                    shuffle=True,
                                    verbose=True,)

        val_dataloader = Loader_aug(dataset=val_dataset,
                                    batch_size=dataparams.batch_size,
                                    window_size=dataparams.window_size,
                                    window_step=dataparams.window_step,
                                    anomaly_types=dataparams.anomaly_types,
                                    anomaly_types_for_dict=anomaly_types_for_dict,
                                    min_range=dataparams.min_range,
                                    min_features=dataparams.min_features,
                                    max_features=dataparams.max_features,
                                    fast_sampling=False,
                                    shuffle=True,
                                    verbose=True,)
        return train_dataloader, val_dataloader
    elif group=='test':
        test_dataset = load_data(dataset=dataparams.dataset,
                    group='test',
                    entities=dataparams.entities,
                    downsampling=dataparams.downsampling,
                    min_length=None,
                    root_dir=dataset_path,
                    verbose=True,
                    validation=False)

        test_dataloader = Loader_aug(dataset=test_dataset,
                                    batch_size=dataparams.batch_size,
                                    window_size=dataparams.window_size,
                                    window_step=dataparams.window_size,
                                    anomaly_types=anomaly_types,
                                    anomaly_types_for_dict=anomaly_types_for_dict,
                                    min_range=dataparams.min_range,
                                    min_features=dataparams.min_features,
                                    max_features=dataparams.max_features,
                                    fast_sampling=False,
                                    shuffle=False,
                                    verbose=True,)
        return test_dataloader
    elif group=='test_all':
        test_dataset = load_data(dataset=dataparams.dataset,
                    group='test',
                    entities=dataparams.entities,
                    downsampling=dataparams.downsampling,
                    min_length=None,
                    root_dir=dataset_path,
                    verbose=True,
                    validation=False)

        test_dataloader = Loader_aug(dataset=test_dataset,
                                    batch_size=dataparams.batch_size,
                                    window_size=dataparams.window_size,
                                    window_step=1,
                                    anomaly_types=anomaly_types,
                                    anomaly_types_for_dict=anomaly_types_for_dict,
                                    min_range=dataparams.min_range,
                                    min_features=dataparams.min_features,
                                    max_features=dataparams.max_features,
                                    fast_sampling=False,
                                    shuffle=False,
                                    verbose=True,)
        return test_dataloader
