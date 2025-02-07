import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import requests
import zipfile

from .dataset import Entity, Dataset


MACHINES = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
            'machine-2-1', 'machine-2-2','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7','machine-2-8','machine-2-9',
            'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4','machine-3-5','machine-3-6','machine-3-7','machine-3-8', 'machine-3-9',
            'machine-3-10', 'machine-3-11']
smap_data_set_number = ['A-1', 'A-2', 'A-3', 'A-4', 'A-7', 'B-1', 'D-1', 'D-11', 'D-13', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3']
msl_data_set_number = ['C-1', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15', 'T-12', 'T-13', 'T-4', 'T-5']

# Data URIs
SMD_URL = 'https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset'
ANOMALY_ARCHIVE_URI = r'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip'
VALID_DATASETS = ['msl', 'smap', 'smd', 'anomaly_archive', 'iops']

def download_file(filename:str, directory: str, source_url: str, decompress: bool = False) -> None:
    """Download data from source_ulr inside directory.
    Parameters
    ----------
    filename: str
        Name of file
    directory: str, Path
        Custom directory where data will be downloaded.
    source_url: str
        URL where data is hosted.
    decompress: bool
        Wheter decompress downloaded file. Default False.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    print('directory', directory)
    directory.mkdir(parents=True, exist_ok=True)

    filepath = Path(f'{directory}/{filename}')

    # Streaming, so we can iterate over the response.
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(source_url, stream=True, headers=headers)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
            f.flush()
    t.close()

    size = filepath.stat().st_size

    if decompress:
        if '.zip' in filepath.suffix:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(directory)
        else:
            from patoolib import extract_archive
            extract_archive(str(filepath), outdir=directory)

def load_data(dataset: str, group: str, entities: Union[str, List[str]], downsampling: float=None, min_length: float=None, root_dir:str='./data', normalize:bool=True, verbose:bool=True, validation:bool=False):
    """Function to load TS anomaly detection datasets.
    Parameters
    ----------
    dataset: str
        Name of the dataset.
    group: str
        The train or test split.
    entities: Union[str, List[str]]
        Entities to load from the dataset.
    downsampling: Optional[float]
        Whether and the extent to downsample the data.
    root_dir: str
        Path to the directory where the datasets are stored.
    normalize: bool
        Whether to normalize Y.
    verbose: bool
        Controls verbosity
    """
    if dataset == 'smd':
        return load_smd(group=group, machines=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'msl':
        if entities=='msl':
            entities=msl_data_set_number
        return load_msl(group=group, channels=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'smap':
        if entities=='smap':
            entities=smap_data_set_number
        return load_smap(group=group, channels=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'anomaly_archive':
        return load_anomaly_archive(group=group, datasets=entities, downsampling=downsampling, min_length=min_length, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'iops':
        return load_iops(group=group, filename=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    else:
        raise ValueError(f'Dataset must be one of {VALID_DATASETS}, but {dataset} was passed!')

def load_smd(group, machines=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    # NOTE: The SMD dataset is normalized and therefore we do not need normalize it further. The normalize parameter is for input compatibility.
    if machines is None:
        machines = MACHINES

    if isinstance(machines, str):
        machines = [machines]

    root_dir = f'{root_dir}/ServerMachineDataset'

    # Download data
    for machine in machines:

        if not os.path.exists(f'{root_dir}/train/{machine}.txt'):

            print('downloading SMD train')
            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/train',
                          source_url=f'{SMD_URL}/train/{machine}.txt')

            print('downloading SMD test')
            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/test',
                          source_url=f'{SMD_URL}/test/{machine}.txt')

            print('downloading SMD test label')
            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/test_label',
                          source_url=f'{SMD_URL}/test_label/{machine}.txt')

    # Load train data
    if group=='train':
        entities, entities_val = [], []
        for machine in machines:
            name = 'smd-train'
            name_val = 'smd-val'
            train_file = f'{root_dir}/train/{machine}.txt'
            Y = np.loadtxt(train_file, delimiter=',').T

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape

                right_padding = downsampling - n_t%downsampling
                Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)

            if validation:
                train_length = int(Y.shape[1]*0.9)
                entity = Entity(Y=Y[:, :train_length], name=machine, verbose=verbose)
                entities.append(entity)
                entity_val = Entity(Y=Y[:, train_length:], name=machine, verbose=verbose)
                entities_val.append(entity_val)
            else:
                entity = Entity(Y=Y, name=machine, verbose=verbose)
                entities.append(entity)

        if validation:
            smd = Dataset(entities=entities, name=name, verbose=verbose)
            smd_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
            return smd, smd_val
        else:
            smd = Dataset(entities=entities, name=name, verbose=verbose)
            return smd

    # Load test data
    elif group=='test':
        entities = []
        for machine in machines:
            name = 'smd-test'
            test_file = f'{root_dir}/test/{machine}.txt'
            label_file = f'{root_dir}/test_label/{machine}.txt'

            Y = np.loadtxt(test_file, delimiter=',').T
            labels = np.loadtxt(label_file, delimiter=',')

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape
                right_padding = downsampling - n_t%downsampling

                Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
                labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

            labels = labels[None, :]
            entity = Entity(Y=Y, name=machine, labels=labels, verbose=verbose)
            entities.append(entity)

        smd = Dataset(entities=entities, name=name, verbose=verbose)
        return smd


def load_msl(group, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    return _load_nasa(group=group, spacecraft='MSL', channels=channels, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)

def load_smap(group, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    return _load_nasa(group=group, spacecraft='SMAP', channels=channels, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)

def _load_nasa(group, spacecraft, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    root_dir = f'{root_dir}/NASA'
    meta_data = pd.read_csv(f'{root_dir}/labeled_anomalies.csv')

    CHANNEL_IDS =  list(meta_data.loc[meta_data['spacecraft'] == spacecraft]['chan_id'].values)
    if verbose:
        print(f'Number of Entities: {len(CHANNEL_IDS)}')

    print('channels',channels)
    print('CHANNELS',sorted(CHANNEL_IDS))
    if channels is None: channels = CHANNEL_IDS

    if isinstance(channels, str):
        channels = [channels]

    if group == 'train':
        entities, entities_val = [], []
        for channel_id in channels:
            if normalize:
                with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f:
                    Y = np.load(f) # Transpose dataset
                scaler = MinMaxScaler()
                scaler.fit(Y)

            name = f'{spacecraft}-train'
            name_val = f'{spacecraft}-val'
            with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f:
                Y = np.load(f).T # Transpose dataset

            if normalize:
                Y = scaler.transform(Y.T).T

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape

                right_padding = downsampling - n_t%downsampling
                Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)


            if validation:
                train_length = int(Y.shape[1]*0.9)
                entity = Entity(Y=Y[:, :train_length], name=channel_id, verbose=verbose)
                entities.append(entity)
                entity_val = Entity(Y=Y[:, train_length:], name=channel_id, verbose=verbose)
                entities_val.append(entity_val)
            else:
                # print('Y', Y.shape)
                # entity = Entity(Y=Y[0, :].reshape((1, -1)), X=Y[1:, :], name=channel_id, verbose=verbose)
                entity = Entity(Y=Y, name=channel_id, verbose=verbose)
                entities.append(entity)

        if validation:
            data = Dataset(entities=entities, name=name, verbose=verbose)
            data_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
            return data, data_val
        else:
            data = Dataset(entities=entities, name=name, verbose=verbose)
            return data


    elif group == 'test':
        entities = []
        for channel_id in channels:
            if normalize:
                with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f:
                    Y = np.load(f) # Transpose dataset
                scaler = MinMaxScaler()
                scaler.fit(Y)

            name = f'{spacecraft}-test'
            with open(f'{root_dir}/test/{channel_id}.npy', 'rb') as f:
                Y = np.load(f).T # Transpose dataset

            if normalize:
                Y = scaler.transform(Y.T).T

            # Label the data
            labels = np.zeros(Y.shape[1])
            anomalous_sequences = eval(meta_data.loc[meta_data['chan_id'] == channel_id]['anomaly_sequences'].values[0])
            if verbose: print('Anomalous sequences:', anomalous_sequences)

            for interval in anomalous_sequences:
                labels[interval[0]:interval[1]] = 1

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape
                right_padding = downsampling - n_t%downsampling

                Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
                labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

            labels = labels[None, :]
            # entity = Entity(Y=Y[0, :].reshape((1, -1)), X=Y[1:, :], name=channel_id, labels=labels, verbose=verbose)
            entity = Entity(Y=Y, name=channel_id, labels=labels, verbose=verbose)
            entities.append(entity)

        data = Dataset(entities=entities, name=name, verbose=verbose)
        return data

def download_anomaly_archive(root_dir='./data'):
    """Convenience function to download the Timeseries Anomaly Archive datasets
    """
    # Download the data
    download_file(filename=f'AnomalyArchive',
                directory=root_dir,
                source_url=ANOMALY_ARCHIVE_URI,
                decompress=True)

    # Reorganising the data
    shutil.move(src=f'{root_dir}/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData',
                dst=root_dir)
    os.remove(os.path.join(root_dir, 'AnomalyArchive'))
    shutil.rmtree(os.path.join(root_dir, 'AnomalyDatasets_2021'))
    shutil.move(src=f'{root_dir}/UCR_Anomaly_FullData',
                dst=f'{root_dir}/AnomalyArchive')

def load_anomaly_archive(group, datasets=None, downsampling=None, min_length=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    if not os.path.exists(f'{root_dir}/AnomalyArchive/'): download_anomaly_archive(root_dir=root_dir)

    ANOMALY_ARCHIVE_ENTITIES = ['_'.join(e.split('_')[:4]) for e in os.listdir(os.path.join(root_dir, 'AnomalyArchive'))]
    ANOMALY_ARCHIVE_ENTITIES = sorted(ANOMALY_ARCHIVE_ENTITIES)

    if datasets is None: datasets = ANOMALY_ARCHIVE_ENTITIES
    if verbose: print(f'Number of datasets: {len(datasets)}')

    entities, entities_val = [], []
    for file in os.listdir(os.path.join(root_dir, 'AnomalyArchive')):

        downsampling_entity = downsampling
        if '_'.join(file.split('_')[:4]) in datasets or file.split('_')[0]==datasets or file.split('_')[2]==datasets:
            with open(os.path.join(root_dir, 'AnomalyArchive',  file)) as f:
                Y = f.readlines()
                if len(Y) == 1:
                    Y = Y[0].strip()
                    Y = np.array([eval(y) for y in Y.split(" ") if len(y) > 1]).reshape((1, -1))
                elif len(Y) > 1:
                    Y = np.array([eval(y.strip()) for y in Y]).reshape((1, -1))

            fields = file.split('_')
            meta_data = {
                    'name': '_'.join(fields[:4]),
                    'train_end': int(fields[4]),
                    'anomaly_start_in_test': int(fields[5])-int(fields[4]),
                    'anomaly_end_in_test': int(fields[6][:-4])-int(fields[4]),
                }
            if verbose:
                print(f'Entity meta-data: {meta_data}')

            if normalize:
                Y_train = Y[0, 0:meta_data['train_end']].reshape((-1, 1))
                scaler = MinMaxScaler()
                scaler.fit(Y_train)
                Y = scaler.transform(Y.T).T

            n_time = Y.shape[-1]
            len_train = meta_data['train_end']
            len_test = n_time - len_train

            # No downsampling if n_time < min_length
            if (downsampling_entity is not None) and (min_length is not None):

                if (len_train//downsampling_entity < min_length) or (len_test//downsampling_entity < min_length):
                    downsampling_entity = None

            if group == 'train':
                name = f"{meta_data['name']}-train"
                name_val = f"{meta_data['name']}-val"
                Y = Y[0, 0:meta_data['train_end']].reshape((1, -1))

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_t = Y.shape

                    right_padding = downsampling_entity - n_t%downsampling_entity
                    Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                    Y = Y.reshape(n_features, Y.shape[-1]//downsampling_entity, downsampling_entity).max(axis=2)

                if validation:
                    train_length = int(Y.shape[1]*0.9)
                    entity = Entity(Y=Y.reshape((1, -1))[:, :train_length], name=meta_data['name'], verbose=verbose)
                    entities.append(entity)
                    entity_val = Entity(Y=Y.reshape((1, -1))[:, train_length:], name=meta_data['name'], verbose=verbose)
                    entities_val.append(entity_val)
                else:
                    entity = Entity(Y=Y.reshape((1, -1)), name=meta_data['name'], verbose=verbose)
                    entities.append(entity)


            elif group == 'test':
                name = f"{meta_data['name']}-test"
                Y = Y[0, meta_data['train_end']+1:].reshape((1, -1))

                # Label the data
                labels = np.zeros(Y.shape[1])
                labels[meta_data['anomaly_start_in_test']:meta_data['anomaly_end_in_test']] = 1

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_t = Y.shape
                    right_padding = downsampling_entity - n_t%downsampling_entity

                    Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                    labels = np.pad(labels, (right_padding, 0))

                    Y = Y.reshape(n_features, Y.shape[-1]//downsampling_entity, downsampling_entity).max(axis=2)
                    labels = labels.reshape(labels.shape[0]//downsampling_entity, downsampling_entity).max(axis=1)

                labels = labels[None, :]
                entity = Entity(Y=Y.reshape((1, -1)), name=meta_data['name'], labels=labels, verbose=verbose)
                entities.append(entity)


    if validation:
        data = Dataset(entities=entities, name=name, verbose=verbose)
        data_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
        return data, data_val
    else:
        data = Dataset(entities=entities, name=name, verbose=verbose)
        return data


def load_iops(group, filename, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    root_dir = f'{root_dir}/IOPS/{filename}'

    if group == 'train':
        df = pd.read_csv(f'{root_dir}.train.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)

        name = f'{filename}-train'
        name_val = f'{filename}-val'
        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(Y.T)
            Y = scaler.transform(Y.T).T

        # Downsampling
        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling
            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)


        if validation:
            train_length = int(Y.shape[1]*0.9)
            entity = Entity(Y=Y[:, :train_length], name=name, verbose=verbose)
            entity_val = Entity(Y=Y[:, train_length:], name=name_val, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            data_val = Dataset(entities=[entity_val], name=name_val, verbose=verbose)
            return data, data_val
        else:
            entity = Entity(Y=Y, name=name, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            return data


    elif group == 'test':
        df = pd.read_csv(f'{root_dir}.test.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)
        if normalize:
            df_train = pd.read_csv(f'{root_dir}.train.out', header=None, names=['Value', 'Label'])
            Y_train = np.array(df_train['Value']).reshape(1,-1)
            scaler = MinMaxScaler()
            scaler.fit(Y_train.T)
            Y = scaler.transform(Y.T).T

        name = f'{filename}-test'

        # Label the data
        labels = np.array(df['Label'])

        # Downsampling
        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling

            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            labels = np.pad(labels, (right_padding, 0))

            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
            labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

        labels = labels[None, :]
        entity = Entity(Y=Y, name=name, labels=labels, verbose=verbose)
        data = Dataset(entities=[entity], name=name, verbose=verbose)
        return data