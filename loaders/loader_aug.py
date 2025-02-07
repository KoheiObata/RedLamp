import numpy as np
import torch as t

from .dataset import Dataset, Entity

from typing import List, Tuple, Optional, Union


class Loader_aug(object):
    def __init__(self,
                 dataset: Union[Dataset, Entity],
                 batch_size: int,
                 window_size: int,
                 window_step: int,
                 anomaly_types: list,
                 anomaly_types_for_dict: list = None,
                 min_range: int = 1,
                 min_features: int = 1,
                 max_features: int = 5,
                 fast_sampling: bool = False,
                 shuffle: bool = True,
                 verbose: bool = False ) -> 'Loader_aug':
        """
        Parameters
        ----------
        dataset: Dataset object
            Dataset to sample windows.
        batch_size: int
            Batch size.
        windows_size: int
            Size of windows to sample.
        window_step: int
            Step size between windows.
        shuffle: bool
            Shuffle windows.
        verbose:
            Boolean for printing details.
        """
        if isinstance(dataset, Entity):
            dataset = Dataset(entities=[dataset], name=dataset.name, verbose=False)

        self.dataset = dataset
        self.batch_size = batch_size
        self.anomaly_types = anomaly_types
        self.min_range = min_range
        self.min_features = min_features
        self.max_features = max_features
        self.fast_sampling = fast_sampling
        self.shuffle = shuffle
        self.verbose = verbose

        if window_size > 0:
            self.window_size = window_size
            self.window_step = window_step
        else:
            self.window_size = dataset.total_time # TODO: will only work with 1 entity
            self.window_step = dataset.total_time

        if anomaly_types_for_dict:
	        self.anomaly_dict = self._get_anomaly_dict(anomaly_types_for_dict)
        else:
            self.anomaly_dict = self._get_anomaly_dict(self.anomaly_types)
        self._inject_anomalies()



    def _inject_anomalies(self):
        self.Y_windows = []
        self.Z_windows = []
        self.anomaly_mask = []
        self.label = []
        for anomaly_type in self.anomaly_types:
            for entity in self.dataset.entities:
                if entity.Y.shape[1] >= self.window_size:
                    y_windows, z_windows, anomaly_mask = self._array_to_windows(entity.Y, anomaly_type)
                    self.Y_windows.append(y_windows)
                    self.Z_windows.append(z_windows)
                    self.anomaly_mask.append(anomaly_mask)
                    self.label.append([anomaly_type for _ in range(self.Y_windows[-1].shape[0])])

        self.Y_windows = t.cat(self.Y_windows)
        self.Z_windows = t.cat(self.Z_windows)
        self.anomaly_mask = t.cat(self.anomaly_mask)
        self.label = [item for sublist in self.label for item in sublist]
        self.label = t.Tensor(self.generate_one_hot(self.label, self.anomaly_dict))
        self.n_idxs = len(self.Y_windows)
        self.n_batch_in_epochs = int(np.ceil(self.n_idxs / self.batch_size))

    def _array_to_windows(self, Y, anomaly_type):
        """
        input
        Y: (n_features, n_time)
        anomaly_type: 'normal', 'spike', ...
        output
        y_windows: (batch, n_features, window_size), contain anomaly
        z_windows: (batch, n_features, window_size), normal
        """
        n_features, n_time = Y.shape
        y_windows = []
        z_windows = []
        anomaly_mask = []
        window_start, window_end = 0, self.window_size
        while window_end<=n_time:
            Y_temp, Z_temp, mask_temp = self.select_anomalies(anomaly_type, Y, window_start, window_end)

            y_windows.append(t.Tensor(Y_temp))
            z_windows.append(t.Tensor(Z_temp))
            anomaly_mask.append(t.Tensor(mask_temp))

            window_start += self.window_step
            window_end += self.window_step
        y_windows = t.stack(y_windows, dim=0)
        z_windows = t.stack(z_windows, dim=0)
        anomaly_mask = t.stack(anomaly_mask, dim=0)
        return y_windows, z_windows, anomaly_mask

    def select_anomalies(self, anomaly_type, Y, window_start, window_end):
        if anomaly_type == 'normal':
            Y_temp = np.copy(Y[:, window_start:window_end])
            Z_temp = np.copy(Y[:, window_start:window_end])
            mask_temp = np.ones_like(Y_temp)
        elif anomaly_type == 'spike':
            Y_temp, Z_temp, mask_temp = self._inject_spike(Y, window_start, window_end, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'flip':
            Y_temp, Z_temp, mask_temp = self._inject_flip(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features, fast_sampling=self.fast_sampling)
        elif anomaly_type == 'speedup':
            Y_temp, Z_temp, mask_temp = self._inject_speedup(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features, fast_sampling=self.fast_sampling)
        elif anomaly_type == 'noise':
            Y_temp, Z_temp, mask_temp = self._inject_noise(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'cutoff':
            Y_temp, Z_temp, mask_temp = self._inject_cutoff(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'average':
            Y_temp, Z_temp, mask_temp = self._inject_average(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features, fast_sampling=self.fast_sampling)
        elif anomaly_type == 'scale':
            Y_temp, Z_temp, mask_temp = self._inject_scale(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'wander':
            Y_temp, Z_temp, mask_temp = self._inject_wander(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'contextual':
            Y_temp, Z_temp, mask_temp = self._inject_contextual(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'upsidedown':
            Y_temp, Z_temp, mask_temp = self._inject_upsidedown(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'mixture':
            Y_temp, Z_temp, mask_temp = self._inject_mixture(Y, window_start, window_end, min_range=self.min_range, min_features=self.min_features, max_features=self.max_features)
        elif anomaly_type == 'random':
            anomaly_types=['spike','flip','speedup','noise','cutoff','average','scale','wander','contextual','upsidedown','mixture']
            selected_anomaly = np.random.choice(anomaly_types, 1)[0]
            Y_temp, Z_temp, mask_temp = self.select_anomalies(selected_anomaly, Y, window_start, window_end)
        return Y_temp, Z_temp, mask_temp

    def _inject_spike(self, Y, window_start, window_end, min_features=1, max_features=5, scale=1):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        #window
        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1
        loc_time = np.random.randint(low=0, high=self.window_size, size=n_anom_features)
        loc_features = np.random.randint(low=0, high=n_features, size=n_anom_features)
        #add spike
        Y_temp[loc_features, loc_time] += np.random.normal(loc=0, scale=scale, size=1)

        #mask
        mask_temp[loc_features, loc_time] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_flip(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5, anomaly_range=200, fast_sampling=False):
        if fast_sampling:
            Y = Y[:, window_start:window_end]
            window_start, window_end = 0, window_end - window_start
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y)
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            #where to flip
            if min_range==self.window_size:
                anomaly_start = window_start
                anomaly_end = window_end
            else:
                if np.random.rand(1)>0.5:
                    anomaly_start = np.random.randint(low=window_start, high=window_end-min_range, size=1)[0]
                    anomaly_end = np.random.randint(low=anomaly_start+min_range, high=anomaly_start+anomaly_range, size=1)[0]
                    if anomaly_end>n_time: anomaly_end=n_time
                else:
                    anomaly_end = np.random.randint(low=window_start+min_range, high=window_end, size=1)[0]
                    anomaly_start = np.random.randint(low=anomaly_end-anomaly_range, high=anomaly_end-min_range, size=1)[0]
                    if anomaly_start<0: anomaly_start=0

            #flip sequence
            Y_temp[loc_feature, anomaly_start:anomaly_end] = Y_temp[loc_feature, anomaly_start:anomaly_end][::-1]
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

        Y_temp = Y_temp[:, window_start:window_end]
        mask_temp = mask_temp[:, window_start:window_end]
        return Y_temp, Z_temp, mask_temp

    def _inject_speedup(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5, frequency=[0.5, 2], fast_sampling=False):
        if fast_sampling:
            Y = Y[:, window_start:window_end]
            window_start, window_end = 0, window_end - window_start
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y)
        Z_temp = np.copy(Y_temp)
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            if min_range==self.window_size:
                anomaly_start = window_start
                anomaly_end = window_end
            else:
                anomaly_start = np.random.randint(low=window_start, high=window_end-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=window_end, size=1)[0]
            if anomaly_end>n_time: anomaly_end=n_time
            anomaly_length = anomaly_end-anomaly_start

            def time_stretch(x, f):
                t = len(x)
                original_time = np.arange(t)
                new_t = int(t / f)
                new_time = np.linspace(0, t-1, new_t)

                y = np.interp(new_time, original_time, x)
                return y
            freq = np.random.choice(frequency, size=1)[0]
            #when the sequence is not long enough
            if anomaly_start+int(freq*anomaly_length)+(window_end-anomaly_end)>n_time:
                freq = 0.5
            if freq<=1:
                x = time_stretch(Y[loc_feature],freq)
            else:
                x = Y[loc_feature, ::int(freq)]

            #speedup
            Y_temp[loc_feature, anomaly_start:anomaly_end] = x[int(anomaly_start/freq):int(anomaly_start/freq)+anomaly_length]
            Y_temp[loc_feature, anomaly_end:window_end] = Y[loc_feature, anomaly_start+int(freq*anomaly_length):anomaly_start+int(freq*anomaly_length)+(window_end-anomaly_end)]

            #after speedup is anomaly (strictly speaking, it's lagged)
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

            #interpolate anomaly with the average
            Z_temp[loc_feature, anomaly_start:anomaly_end] = np.mean(Z_temp[loc_feature, anomaly_start:anomaly_end])

        Y_temp = Y_temp[:, window_start:window_end]
        Z_temp = Z_temp[:, window_start:window_end]
        mask_temp = mask_temp[:, window_start:window_end]
        return Y_temp, Z_temp, mask_temp

    def _inject_noise(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5, scale=0.1):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            if min_range==self.window_size:
                anomaly_start = 0
                anomaly_end = min_range
            else:
                anomaly_start = np.random.randint(low=0, high=self.window_size-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=self.window_size, size=1)[0]

            #noise
            Y_temp[loc_feature, anomaly_start:anomaly_end] += np.random.normal(loc=0, scale=scale, size=anomaly_end-anomaly_start)
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

        return Y_temp, Z_temp, mask_temp

    def _inject_cutoff(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            if min_range==self.window_size:
                anomaly_start = 0
                anomaly_end = min_range
            else:
                anomaly_start = np.random.randint(low=0, high=self.window_size-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=self.window_size, size=1)[0]

            max_value = max(Y_temp[loc_feature])
            min_value = min(Y_temp[loc_feature])
            Y_temp[loc_feature, anomaly_start:anomaly_end] = np.random.uniform(low=min_value, high=max_value, size=1)
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

        return Y_temp, Z_temp, mask_temp


    def _inject_average(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5, ma_window=20, anomaly_range=200, fast_sampling=False):
        if fast_sampling:
            Y = Y[:, window_start:window_end]
            window_start, window_end = 0, window_end - window_start
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y)
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            #where to do moving average
            if min_range==self.window_size:
                anomaly_start = window_start
                anomaly_end = window_end
            else:
                if np.random.rand(1)>0.5:
                    anomaly_start = np.random.randint(low=window_start, high=window_end-min_range, size=1)[0]
                    anomaly_end = np.random.randint(low=anomaly_start+min_range, high=anomaly_start+anomaly_range, size=1)[0]
                    if anomaly_end>n_time: anomaly_end=n_time
                else:
                    anomaly_end = np.random.randint(low=window_start+min_range, high=window_end, size=1)[0]
                    anomaly_start = np.random.randint(low=anomaly_end-anomaly_range, high=anomaly_end-min_range, size=1)[0]
                    if anomaly_start<0: anomaly_start=0

            #MA sequence
            def moving_average_with_padding(x, w):
                pad_width = w // 2
                padded_x = np.pad(x, pad_width, mode='edge')
                return np.convolve(padded_x, np.ones(w), 'valid') / w
            Y_temp[loc_feature, anomaly_start:anomaly_end] = moving_average_with_padding(Y_temp[loc_feature, anomaly_start:anomaly_end], ma_window)[1:]
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

        Y_temp = Y_temp[:, window_start:window_end]
        mask_temp = mask_temp[:, window_start:window_end]
        return Y_temp, Z_temp, mask_temp

    def _inject_scale(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5, scale=1):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        #window
        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            if min_range==self.window_size:
                anomaly_start = 0
                anomaly_end = min_range
            else:
                anomaly_start = np.random.randint(low=0, high=self.window_size-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=self.window_size, size=1)[0]
            #scale
            Y_temp[loc_feature, anomaly_start:anomaly_end] *= abs(np.random.normal(loc=1, scale=scale, size=1))
            #mask
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_wander(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5, scale=1):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            #window
            if min_range==self.window_size:
                anomaly_start = 0
                anomaly_end = min_range
            else:
                anomaly_start = np.random.randint(low=0, high=self.window_size-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=self.window_size, size=1)[0]
            #wander
            baseline = np.random.normal(loc=0, scale=scale, size=1)[0]
            Y_temp[loc_feature, anomaly_start:anomaly_end] += np.linspace(0, baseline, anomaly_end-anomaly_start)
            Y_temp[loc_feature, anomaly_end:] += baseline

            # #interpolate anomaly with the average
            # Z_temp[loc_feature, anomaly_start:anomaly_end] = np.mean(Z_temp[loc_feature, anomaly_start:anomaly_end])

            #after wander is anomaly (strictly speaking, it's scaled)
            mask_temp[loc_feature, anomaly_start:] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_contextual(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5, scale=1):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            #window
            if min_range==self.window_size:
                anomaly_start = 0
                anomaly_end = min_range
            else:
                anomaly_start = np.random.randint(low=0, high=self.window_size-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=self.window_size, size=1)[0]
            #contextual
            a = np.random.normal(loc=1, scale=scale, size=1)[0]
            b = np.random.normal(loc=0, scale=scale, size=1)[0]
            Y_temp[loc_feature, anomaly_start:anomaly_end] = a*Y_temp[loc_feature, anomaly_start:anomaly_end] + b

            #mask
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_upsidedown(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            #window
            if min_range==self.window_size:
                anomaly_start = 0
                anomaly_end = min_range
            else:
                anomaly_start = np.random.randint(low=0, high=self.window_size-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=self.window_size, size=1)[0]
            #upside down
            mean = np.mean(Y_temp[loc_feature, anomaly_start:anomaly_end])
            Y_temp[loc_feature, anomaly_start:anomaly_end] = -(Y_temp[loc_feature, anomaly_start:anomaly_end]-mean) + mean

            #mask
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_mixture(self, Y, window_start, window_end, min_range=1, min_features=1, max_features=5):
        n_features, n_time = Y.shape
        if max_features > n_features: max_features = n_features

        if min_features==max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0]+1

        Y_temp = np.copy(Y[:, window_start:window_end])
        Z_temp = np.copy(Y[:, window_start:window_end])
        mask_temp = np.ones_like(Y_temp)

        loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
        for loc_feature in loc_features_list:
            if min_range==self.window_size:
                anomaly_start = 0
                anomaly_end = min_range
            else:
                anomaly_start = np.random.randint(low=0, high=self.window_size-min_range, size=1)[0]
                anomaly_end = np.random.randint(low=anomaly_start+min_range, high=self.window_size, size=1)[0]
            anomaly_length = anomaly_end - anomaly_start
            #mixture
            mixture_start = np.random.randint(low=0, high=n_time-anomaly_length, size=1)[0]
            mixture_end = mixture_start+anomaly_length
            Y_temp[loc_feature, anomaly_start:anomaly_end] = Y[loc_feature, mixture_start:mixture_end]

            #mask
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    @staticmethod
    def _get_anomaly_dict(anomaly_types):
        # anomaly_dict = {'normal':0, 'spike':1, 'flip':2, 'speedup':3, 'noise':4, 'cutoff':5, 'average':6, 'scale':7, 'wander':8, 'contextual':9, 'upsidedown':10, 'mixture':11}
        anomaly_types = list(dict.fromkeys(anomaly_types))
        anomaly_dict = {}
        for i, anomaly_type in enumerate(anomaly_types):
            anomaly_dict[anomaly_type] = i
        return anomaly_dict

    @staticmethod
    def generate_one_hot(anomaly_types, anomaly_dict):
        '''
        input
        anomaly_types = ['normal','spike','cutoff']
        output
        one_hot_vectors = [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                           [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
                           [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        '''
        labels = [anomaly_dict[atype] for atype in anomaly_types]
        one_hot_vectors = np.eye(len(anomaly_dict))[labels]
        return one_hot_vectors

    @staticmethod
    def generate_anomaly_types(one_hot_vectors, anomaly_dict):
        '''
        input
        one_hot_vectors = [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                           [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
                           [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        output
        anomaly_types = ['normal','spike','cutoff']
        '''
        inverse_dict = {v: k for k, v in anomaly_dict.items()}

        if one_hot_vectors.dim()==1: one_hot_vectors = one_hot_vectors[None, :]
        labels = np.argmax(np.array(one_hot_vectors), axis=1)
        return [inverse_dict[label] for label in labels]


    def __len__(self):
        return self.n_idxs

    def __iter__(self):
        if self.shuffle:
            sample_idxs = np.random.choice(a=self.n_idxs, size=self.n_idxs)
        else:
            sample_idxs = np.arange(self.n_idxs)

        for idx in range(self.n_batch_in_epochs):
            batch_idx = sample_idxs[(idx * self.batch_size) : (idx + 1) * self.batch_size]
            batch_idx = [int(idx) for idx in batch_idx]
            batch = self.__get_item__(idx=batch_idx)
            yield batch

    def __get_item__(self, idx):
        """
        """
        # Index windows from tensors
        Y_batch = self.Y_windows[idx]
        Z_batch = self.Z_windows[idx]
        anomaly_mask_batch = self.anomaly_mask[idx]
        label_batch = self.label[idx]

        # Batch
        batch = {'Y': t.as_tensor(Y_batch),
                 'Z': t.as_tensor(Z_batch),
                 'anomaly_mask': t.as_tensor(anomaly_mask_batch),
                 'label': t.as_tensor(label_batch),
                 'idx': idx}

        return batch

    def __str__(self):
        # TODO: Complete the loader.
        return 'I am a loader'


class Loader_aug_batch(Loader_aug):
    def __init__(self,
                 data: t.Tensor,
                 batch_size: int,
                 anomaly_types: list,
                 anomaly_types_for_dict: list = None,
                 min_range: int = 1,
                 min_features: int = 1,
                 max_features: int = 5,
                 fast_sampling: bool = False,
                 shuffle: bool = True,
                 verbose: bool = False):

        self.data = t.Tensor(data)
        self.batch_size = batch_size
        self.window_size = data.shape[2]

        self.anomaly_types = anomaly_types
        self.min_range = min_range
        self.min_features = min_features
        self.max_features = max_features
        self.fast_sampling = fast_sampling
        self.shuffle = shuffle
        self.verbose = verbose


        if anomaly_types_for_dict:
	        self.anomaly_dict = self._get_anomaly_dict(anomaly_types_for_dict)
        else:
            self.anomaly_dict = self._get_anomaly_dict(self.anomaly_types)

        self._inject_anomalies_batch()


    def _inject_anomalies_batch(self):
        self.Y_windows = []
        self.Z_windows = []
        self.anomaly_mask = []
        self.label = []
        for anomaly_type in self.anomaly_types:
                y_windows, z_windows, anomaly_mask = self._array_to_windows_batch(self.data, anomaly_type)
                self.Y_windows.append(y_windows)
                self.Z_windows.append(z_windows)
                self.anomaly_mask.append(anomaly_mask)
                self.label.append([anomaly_type for _ in range(self.Y_windows[-1].shape[0])])

        self.Y_windows = t.cat(self.Y_windows)
        self.Z_windows = t.cat(self.Z_windows)
        self.anomaly_mask = t.cat(self.anomaly_mask)
        self.label = [item for sublist in self.label for item in sublist]
        self.label = t.Tensor(self.generate_one_hot(self.label, self.anomaly_dict))
        self.n_idxs = len(self.Y_windows)
        self.n_batch_in_epochs = int(np.ceil(self.n_idxs / self.batch_size))

    def _array_to_windows_batch(self, data, anomaly_type):
        """
        input
        data: (batch, n_features, n_time)
        anomaly_type: 'normal', 'spike', ...
        output
        y_windows: (batch, n_features, window_size), contain anomaly
        z_windows: (batch, n_features, window_size), normal
        """
        batch, n_features, n_time = data.shape
        y_windows = []
        z_windows = []
        anomaly_mask = []
        window_start, window_end = 0, self.window_size
        for Y_temp in data:
            Y_temp, Z_temp, mask_temp = self.select_anomalies(anomaly_type, Y_temp, window_start, window_end)

            y_windows.append(t.Tensor(Y_temp))
            z_windows.append(t.Tensor(Z_temp))
            anomaly_mask.append(t.Tensor(mask_temp))

        y_windows = t.stack(y_windows, dim=0)
        z_windows = t.stack(z_windows, dim=0)
        anomaly_mask = t.stack(anomaly_mask, dim=0)
        return y_windows, z_windows, anomaly_mask
