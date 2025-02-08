import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn


from models.meta import ConvAEC
import utils
import datautils



def model_parameters(args):
    params_model = utils.AttrDict(
        name=args.model,
        # Model params
        n_features = args.n_features,
        n_time = args.window_size,
        num_filters = [128, 128, 256, 256],
        embedding_dim = args.embedding_dim,
        kernel_size = 4,
        dropout = 0.2,
        normalization = 'batch',
        stride = 2,
        padding = 2,

        anomaly_types = args.anomaly_types,
        classes = len(args.anomaly_types),
        classifier_dim = 32,
        c_loss_ratio = args.c_loss_ratio,

        apply_anomaly_mask = args.apply_anomaly_mask,
        label_smoothing = args.label_smoothing,
        alpha = 0.1,
        beta = 0.01,
    )
    return params_model


class REDLAMP:
    def __init__(
        self,
        model_dir = "./training",
        params = None,
        device = 'cpu',
    ):

        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.params = params
        self.epoch = params.epoch
        self.device = device

        self.autocast = torch.cuda.amp.autocast()
        self.scaler = torch.cuda.amp.GradScaler()
        self.model = ConvAEC(self.params).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.params.lr)



    def train(self, train_dataloader, val_dataloader=None):
        stop_counter = 0
        best_val_loss = np.inf
        time_list, train_loss_list, val_loss_list = [],[],[]
        train_loss_ae_list, val_loss_ae_list, train_loss_c_list, val_loss_c_list = [],[],[],[]

        for epoch in range(self.epoch):
            self.model.train()
            starttime = time.time()
            cum_loss, step_count = 0, 0
            cum_loss_ae, cum_loss_c = 0, 0
            for step, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                inputs = batch['Y'] #(batch, n_features, window)
                inputs = inputs.transpose(2,1).to(self.device) #(batch, window, n_features)
                inputs_normal = batch['Z']
                inputs_normal = inputs_normal.transpose(2,1).to(self.device) #(batch, window, n_features)
                anomaly_mask = batch['anomaly_mask']
                anomaly_mask = anomaly_mask.transpose(2,1).to(self.device) #(batch, window, n_features)
                label = batch['label'].to(self.device)
                if inputs.shape[0]==1: #BatchNorm of Classifier doesn't work if batchsize=1
                    continue

                with self.autocast:
                    predicted, pred_label, pred_enc = self.model(inputs)
                    loss, loss_ae, loss_c = self.model.calculate_loss(inputs, predicted, label, pred_label, anomaly_mask, epoch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if torch.isnan(loss).any():
                    print(f'Detected NaN loss at epoch {epoch}')
                    # raise RuntimeError(f'Detected NaN loss at epoch {epoch}')
                else:
                    cum_loss += loss.item()
                    cum_loss_ae += loss_ae.item()
                    cum_loss_c += loss_c.item()
                    step_count += 1
            epoch_loss = cum_loss/step_count
            epoch_loss_ae = cum_loss_ae/step_count
            epoch_loss_c = cum_loss_c/step_count
            epoch_t = time.time() - starttime
            print('Epoch:', epoch, '     loss: ', str(epoch_loss)[0:6], '     loss_ae: ', str(epoch_loss_ae)[0:6], '     loss_c: ', str(epoch_loss_c)[0:6], '     time: ', str(epoch_t)[0:4], 'sec')
            time_list.append(epoch_t)
            train_loss_list.append(epoch_loss)
            train_loss_ae_list.append(epoch_loss_ae)
            train_loss_c_list.append(epoch_loss_c)

            # early stop
            if val_dataloader:
                val_loss, val_loss_ae, val_loss_c = self.validation(val_dataloader, epoch)
                val_loss_list.append(val_loss.item())
                val_loss_ae_list.append(val_loss_ae.item())
                val_loss_c_list.append(val_loss_c.item())
                if torch.isnan(val_loss).any():
                    stop_counter += 10
                elif val_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = val_loss
                    print("best validation loss is updated", '     loss: ', str(best_val_loss.item())[:6], '     loss_ae: ', str(val_loss_ae.item())[0:6], '     loss_c: ', str(val_loss_c.item())[0:6])
                    torch.save(self.model.state_dict(),f'{self.model_dir}/bestmodel.pkl')
                else:
                    stop_counter += 1

            np.savetxt(f'{self.model_dir}/time.txt',np.array(time_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/train_loss.txt',np.array(train_loss_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/valid_loss.txt',np.array(val_loss_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/train_loss_ae.txt',np.array(train_loss_ae_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/valid_loss_ae.txt',np.array(val_loss_ae_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/train_loss_c.txt',np.array(train_loss_c_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/valid_loss_c.txt',np.array(val_loss_c_list),fmt='%.4e')

            if val_dataloader and stop_counter > 9:
                break
        #################################################################################

    def validation(self, val_dataloader, epoch):
        loss = torch.tensor([0.0], requires_grad=False, device=self.device)
        loss_AE = torch.tensor([0.0], requires_grad=False, device=self.device)
        loss_C = torch.tensor([0.0], requires_grad=False, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch['Y'] #(batch, n_features, window)
                inputs = inputs.transpose(2,1).to(self.device) #(batch, window, n_features)
                inputs_normal = batch['Z']
                inputs_normal = inputs_normal.transpose(2,1).to(self.device) #(batch, window, n_features)
                anomaly_mask = batch['anomaly_mask']
                anomaly_mask = anomaly_mask.transpose(2,1).to(self.device) #(batch, window, n_features)
                label = batch['label'].to(self.device)

                predicted, pred_label, pred_enc = self.model(inputs)
                loss_aec, loss_ae, loss_c = self.model.calculate_loss(inputs, predicted, label, pred_label, anomaly_mask, epoch)
                loss += loss_aec
                loss_AE += loss_ae
                loss_C += loss_c
            return loss, loss_AE, loss_C


def test(test_dataloader, model_dir, params, device):
    model = ConvAEC(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl'))

    inputs_list = []
    prediction_list = []
    anomaly_mask_list = []
    label_list = []
    pred_label_list = []
    pred_enc_list = []

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = batch['Y'] #(batch, n_features, window)
            inputs = inputs.transpose(2,1).to(device) #(batch, window, n_features)
            inputs_normal = batch['Z']
            inputs_normal = inputs_normal.transpose(2,1).to(device) #(batch, window, n_features)
            anomaly_mask = batch['anomaly_mask']
            anomaly_mask = anomaly_mask.transpose(2,1).to(device) #(batch, window, n_features)
            label = batch['label'].to(device)

            predicted, pred_label, pred_enc = model(inputs)
            label_list.append(label)
            pred_label_list.append(pred_label)
            pred_enc_list.append(pred_enc)
            inputs_list.append(inputs)
            prediction_list.append(predicted)
            anomaly_mask_list.append(anomaly_mask)

        inputs_list = torch.cat(inputs_list, dim=0)
        inputs_list = inputs_list.to('cpu').detach().numpy().copy()
        prediction_list = torch.cat(prediction_list, dim=0)
        prediction_list = prediction_list.to('cpu').detach().numpy().copy()
        anomaly_mask_list = torch.cat(anomaly_mask_list, dim=0)
        anomaly_mask_list = anomaly_mask_list.to('cpu').detach().numpy().copy()

        label_list = torch.cat(label_list, dim=0)
        label_list = label_list.to('cpu').detach().numpy().copy()
        pred_label_list = torch.cat(pred_label_list, dim=0)
        pred_label_list = pred_label_list.to('cpu').detach().numpy().copy()
        pred_enc_list = torch.cat(pred_enc_list, dim=0)
        pred_enc_list = pred_enc_list.to('cpu').detach().numpy().copy()

        return inputs_list, prediction_list, anomaly_mask_list, label_list, pred_label_list, pred_enc_list


def convolve_minmax_score(score, w=50, minmax=True):
    # Create the convolution kernel and reshape it for broadcasting
    b = np.ones((w, 1)) / w  # Shape it as (w, 1) to convolve along the time axis

    # Apply convolution across the time dimension for all features simultaneously
    score = np.apply_along_axis(lambda m: np.convolve(m, b[:, 0], mode='same'), axis=0, arr=score)

    # Min-max normalization (if specified)
    if minmax:
        min_vals = score.min(axis=0, keepdims=True)
        max_vals = score.max(axis=0, keepdims=True)
        score = (score - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero
    return score

def mse(input, pred, mean=True):
    fn = nn.MSELoss(reduction='none')
    mse_score = np.array(fn(torch.Tensor(input), torch.Tensor(pred)))
    if mse_score.ndim==1: mse_score = np.expand_dims(mse_score, axis=1)
    if mean:
        return np.mean(np.array(mse_score), axis=1)
    else:
        return np.array(mse_score)

def label_score_selected_feature(label, axis=[0]):
    label_copy = np.copy(label)
    label_copy[:,axis] = 0
    label_copy = np.sum(label_copy,axis=1)
    return label_copy

def anomaly_scoreing(input, pred, pred_label, threshold=0.05):
    B,W,D = input.shape
    input = input.reshape(B, -1)
    pred = pred.reshape(B, -1)
    mse_score = mse(input, pred)
    mse_score = convolve_minmax_score(mse_score, w=int(W/2))

    mean_label = np.mean(pred_label, axis=0)
    indices = np.where(mean_label > threshold)[0]
    ce_score = label_score_selected_feature(pred_label, axis=indices)
    ce_score = convolve_minmax_score(ce_score, w=int(W/2))

    anomaly_score = (mse_score + ce_score)/2
    return anomaly_score



def get_meta_data(entity):
    anomaly_data_dir = './dataset/AnomalyArchive'
    if not os.path.exists(f'{anomaly_data_dir}/'):
        import loaders
        loaders.load.download_anomaly_archive(root_dir='./dataset')
    for file in os.listdir(anomaly_data_dir):
        if '_'.join(file.split('_')[:4]) in entity or file.split('_')[0]==entity or file.split('_')[2]==entity:
            fields = file.split('_')
            meta_data = {
                    'name': '_'.join(fields[:4]),
                    'train_end': int(fields[4]),
                    'anomaly_start_in_test': int(fields[5])-int(fields[4]),
                    'anomaly_end_in_test': int(fields[6][:-4])-int(fields[4]),
                }
            print(meta_data)
            return meta_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default='anomaly_archive', help='The dataset name, [anomaly_archive, beatgan_ecg, smd]')
    parser.add_argument('--entities', type=str, default='0', help='[machine-1-1, ...]')

    parser.add_argument('--downsampling', type=int, default=1, help='(defaults to 1)')
    parser.add_argument('--batch-size', type=int, default=16, help='The batch size (defaults to 16)')
    parser.add_argument('--window-size', type=int, default=100, help='The window size (defaults to 64)')
    parser.add_argument('--window-step', type=int, default=1, help='The sliding window (defaults to 1)')

    # Learning
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs')

    # Model
    parser.add_argument('--model', type=str, default='ConvAEC', help='The architecture name')
    parser.add_argument('--anomaly-types', type=str, default='normal,spike,flip,speedup,noise,cutoff,average,scale,wander,contextual,upsidedown,mixture', help='List of anomaly types')

    # Architecture
    parser.add_argument('--embedding_dim', type=int, default=128, help='The size of embedding')
    parser.add_argument('--c_loss_ratio', type=float, default=0.1, help='The weightage for cross-entropy loss (defaults to 0.1)')
    parser.add_argument('--min_features', type=int, default=1, help='The minimum number of augmented features')
    parser.add_argument('--max_features', type=int, default=1, help='The maximum number of augmented features')
    parser.add_argument('--min_range', type=int, default=1, help='The range of inserted anomaly')

    parser.add_argument('--apply_anomaly_mask', action="store_false", default=True, help='if True: reconstruct anomaly-free regions')
    parser.add_argument('--label_smoothing', action="store_false", default=True, help='if True: use soft labels')

    # Computer
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--run_name', type=str, default='test', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')

    args = parser.parse_args()
    print("Arguments:", str(args))

    device = utils.init_dl_program(args.gpu, seed=args.seed)
    print('Device', device)

    args.anomaly_types = args.anomaly_types.split(',') if args.anomaly_types else ['normal','spike','flip','speedup','noise','cutoff','average','scale','wander','contextual','upsidedown','mixture']

    if args.dataset == 'anomaly_archive':
        args.n_features = 1
        min_features = 1
        max_features = 1
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 1
        #for all subdataset
        entity_list = [str(i).zfill(3) for i in range(1,251)]
        #for convenience
        entity_list = ['028']
    elif args.dataset == 'iops':
        args.n_features = 1
        min_features = 1
        max_features = 1
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 10
        entity_list = ['KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3', 'KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0', 'KPI-301c70d8-1630-35ac-8f96-bc1b6f4359ea', 'KPI-42d6616d-c9c5-370a-a8ba-17ead74f3114', 'KPI-43115f2a-baeb-3b01-96f7-4ea14188343c', 'KPI-431a8542-c468-3988-a508-3afd06a218da', 'KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095', 'KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd', 'KPI-55f8b8b8-b659-38df-b3df-e4a5a8a54bc9', 'KPI-57051487-3a40-3828-9084-a12f7f23ee38', 'KPI-6a757df4-95e5-3357-8406-165e2bd49360', 'KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd', 'KPI-6efa3a07-4544-34a0-b921-a155bd1a05e8', 'KPI-7103fa0f-cac4-314f-addc-866190247439', 'KPI-847e8ecc-f8d2-3a93-9107-f367a0aab37d', 'KPI-8723f0fb-eaef-32e6-b372-6034c9c04b80', 'KPI-9c639a46-34c8-39bc-aaf0-9144b37adfc8', 'KPI-a07ac296-de40-3a7c-8df3-91f642cc14d0', 'KPI-a8c06b47-cc41-3738-9110-12df0ee4c721', 'KPI-ab216663-dcc2-3a24-b1ee-2c3e550e06c9', 'KPI-adb2fde9-8589-3f5b-a410-5fe14386c7af', 'KPI-ba5f3328-9f3f-3ff5-a683-84437d16d554', 'KPI-c02607e8-7399-3dde-9d28-8a8da5e5d251', 'KPI-c69a50cf-ee03-3bd7-831e-407d36c7ee91', 'KPI-da10a69f-d836-3baa-ad40-3e548ecf1fbd', 'KPI-e0747cad-8dc8-38a9-a9ab-855b61f5551d', 'KPI-f0932edd-6400-3e63-9559-0a9860a1baa9', 'KPI-ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa']
    elif args.dataset == 'smd':
        args.n_features = 38
        min_features = 1
        max_features = args.n_features
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 10
        entity_list = ["1-1","1-2","1-3","1-4","1-5","1-6","1-7","1-8","2-1","2-2","2-3","2-4","2-5","2-6","2-7","2-8","2-9","3-1","3-2","3-3","3-4","3-5","3-6","3-7","3-8","3-9","3-10","3-11"]
        entity_list = [f'machine-{entity}' for entity in entity_list]
    elif args.dataset == 'smap':
        args.n_features = 25
        min_features = 1
        max_features = args.n_features
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 10
        entity_list = ['smap']
    elif args.dataset == 'msl':
        args.n_features = 55
        min_features = 1
        max_features = args.n_features
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 10
        entity_list = ['msl']


    for entity in entity_list:
        if args.dataset == 'anomaly_archive':
            meta_data = get_meta_data(entity)
            train_end = int(meta_data['train_end'])
            if train_end<10000:
                args.window_step = 1
            elif train_end>=10000 and train_end<100000:
                args.window_step = 10
            elif train_end>=100000:
                args.window_step = 100


        params = utils.AttrDict(
            # Training params
            batch_size=args.batch_size,
            lr=args.lr,
            epoch=args.epoch,
            max_grad_norm=1.0,
            seed=args.seed,
        )
        params.override(model_parameters(args))

        dataparams = utils.AttrDict(
            dataset=args.dataset,
            entities=entity,
            downsampling=args.downsampling,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_step=args.window_step,
            anomaly_types=args.anomaly_types,
            min_range=args.min_range,
            min_features=min_features,
            max_features=max_features,
        )


        base_dir = f'./result/{args.run_name}'
        data_dir = f'{args.dataset}/{entity}/d{dataparams.downsampling}_b{dataparams.batch_size}_w{dataparams.window_size}_s{dataparams.window_step}'
        model_dir = f'{base_dir}/{data_dir}/{args.seed}'

        if os.path.isfile(f'{model_dir}/test_all/input.npy'):
            print(f'{model_dir}/test_all/input.npy', 'exists')
            # continue


        train_dataloader, val_dataloader = datautils.load_dataloader_aug(dataparams, group='train')
        test_dataloader = datautils.load_dataloader_aug(dataparams, anomaly_types=['normal'], anomaly_types_for_dict=args.anomaly_types, group='test_all')
        print('# of train',len(train_dataloader))
        print('# of valid',len(val_dataloader) if val_dataloader else None)
        print('# of test', len(test_dataloader))

        args.Train=True
        if os.path.isfile(f'{model_dir}/test_all/input.npy'):
            print(f'{model_dir}/test/input.npy', 'exists')
            args.Train=False
        if args.Train:
            print('Train')
            model = REDLAMP(model_dir = model_dir, params = params, device = device)
            model.train(train_dataloader, val_dataloader)


        args.Test=True
        test_save_dir = f'{model_dir}/test_all'
        if os.path.isfile(f'{test_save_dir}/input.npy'):
            print(f'{test_save_dir}/input.npy', 'exists')
            args.Test=False
        if args.Test:
            print('Test:Test_all')
            test_inputs, test_prediction, test_anomaly_mask, test_label, test_pred_label, test_pred_enc = test(test_dataloader, model_dir, params, device)
            anomaly_score = anomaly_scoreing(test_inputs, test_prediction, test_pred_label, threshold=0.05)
            os.makedirs(test_save_dir, exist_ok=True)
            np.save(f'{test_save_dir}/input.npy',test_inputs)
            np.save(f'{test_save_dir}/pred.npy',test_prediction)
            np.save(f'{test_save_dir}/anomaly_mask.npy',test_anomaly_mask)
            np.save(f'{test_save_dir}/label.npy',test_label)
            np.save(f'{test_save_dir}/pred_label.npy',test_pred_label)
            np.save(f'{test_save_dir}/enc.npy',test_pred_enc)
            np.save(f'{test_save_dir}/anomaly_score.npy',anomaly_score)


        if entity in ['smap','msl']:
            if entity=='smap': each_entity_list = ['A-1', 'A-2', 'A-3', 'A-4', 'A-7', 'B-1', 'D-1', 'D-11', 'D-13', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3']
            if entity=='msl': each_entity_list = ['C-1', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15', 'T-12', 'T-13', 'T-4', 'T-5']
            for ent in each_entity_list:
                dataparams.entities=ent
                test_dataloader = datautils.load_dataloader_aug(dataparams, anomaly_types=['normal'], anomaly_types_for_dict=args.anomaly_types, group='test_all')
                args.Test=True
                test_save_dir = f'{model_dir}/test_each/{ent}/test_all'
                if os.path.isfile(f'{test_save_dir}/input.npy'):
                    print(f'{test_save_dir}/input.npy', 'exists')
                    args.Test=False
                if args.Test:
                    print('Test:Test_all')
                    test_inputs, test_prediction, test_anomaly_mask, test_label, test_pred_label, test_pred_enc = test(test_dataloader, model_dir, params, device)
                    anomaly_score = anomaly_scoreing(test_inputs, test_prediction, test_pred_label, threshold=0.05)
                    os.makedirs(test_save_dir, exist_ok=True)
                    np.save(f'{test_save_dir}/input.npy',test_inputs)
                    np.save(f'{test_save_dir}/pred.npy',test_prediction)
                    np.save(f'{test_save_dir}/anomaly_mask.npy',test_anomaly_mask)
                    np.save(f'{test_save_dir}/label.npy',test_label)
                    np.save(f'{test_save_dir}/pred_label.npy',test_pred_label)
                    np.save(f'{test_save_dir}/enc.npy',test_pred_enc)
                    np.save(f'{test_save_dir}/anomaly_score.npy',anomaly_score)

        if args.dataset == 'anomaly_archive':
            inputs = np.load(f'{test_save_dir}/input.npy')
            anomaly_score = np.load(f'{test_save_dir}/anomaly_score.npy')
            B,W,D = inputs.shape
            inputs = inputs[:, -1, 0]
            inputs = np.concatenate([np.zeros(W-1),inputs])
            anomaly_score = np.concatenate([np.zeros(W-1),anomaly_score])

            print('Plot')
            import matplotlib.pyplot as plt
            window = 1000
            window_e = 1000
            meta_data = get_meta_data(entity)
            anomaly_start = meta_data['anomaly_start_in_test']
            anomaly_end = meta_data['anomaly_end_in_test']
            if anomaly_start==anomaly_end:
                anomaly_end += 1
            anomaly_length = anomaly_end - anomaly_start

            plt.figure()
            if window>len(anomaly_score):
                window = anomaly_start
                window_e = len(anomaly_score)-anomaly_end
            plt.plot(anomaly_score[anomaly_start-window:anomaly_end+window_e], label='anomaly_score')
            plt.plot(inputs[anomaly_start-window:anomaly_end+window_e], label='input')
            plt.axvspan(window, window+anomaly_length, color='r', alpha=0.3)
            plt.legend()
            plt.savefig(f'{test_save_dir}/fig_{entity}.png')
            plt.close()

        print('Finish')
