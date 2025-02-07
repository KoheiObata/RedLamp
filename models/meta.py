import torch
import torch.nn as nn
import torch.nn.functional as F


from .cnn import ConvEncoder, ConvDecoder
from .classifier import NonLinClassifier

#########################################################################################################
# META CLASS
#########################################################################################################
class MetaAEC(nn.Module):
    def __init__(self, params):
        super(MetaAEC, self).__init__()

        self.encoder = None
        self.decoder = None
        self.classifier = None

        self.name = params.name

        self.classes = params.classes
        self.c_loss_ratio = params.c_loss_ratio #0.5

        self.apply_anomaly_mask = params.apply_anomaly_mask
        self.label_smoothing = params.label_smoothing
        self.alpha = params.alpha
        self.beta = params.beta

    def forward(self, x):
        # print('meta x [batch, window, in_feature]', x.shape)
        x_enc = self.encoder(x)
        # print('meta x_enc [batch, embedding, 1]', x_enc.shape)
        x_hat = self.decoder(x_enc)
        # print('meta x_hat [batch, window, in_feature]', x_hat.shape)
        x_out = self.classifier(x_enc.reshape(x_enc.size(0), -1))
        # print('meta x_out [batch, classes]', x_out.shape)
        return x_hat, x_out, x_enc


    def calculate_loss(self, inputs, predicted, label, pred_label, anomaly_mask, epoch):
        loss_AE = nn.MSELoss()
        loss_C = nn.CrossEntropyLoss(reduction='none')

        if self.apply_anomaly_mask:
            inputs = inputs*anomaly_mask
            predicted = predicted*anomaly_mask
        loss_ae = loss_AE(inputs, predicted)

        if self.label_smoothing:
            normal_loc = 0
            label = label * (1 - self.alpha - self.beta*self.classes + self.beta) + (1 - label) * self.beta
            label[:, normal_loc] += self.alpha

        loss_c = loss_C(pred_label, label)
        loss_c = torch.mean(loss_c)
        return (1-self.c_loss_ratio)*loss_ae + self.c_loss_ratio*loss_c, loss_ae, loss_c

    def calculate_loss_residual(self, residual, predicted, label, pred_label, anomaly_mask, epoch):
        loss_AE = nn.MSELoss()
        loss_C = nn.CrossEntropyLoss(reduction='none')

        loss_ae = loss_AE(residual, predicted)

        if self.label_smoothing:
            normal_loc = 0
            label = label * (1 - self.alpha - self.beta*self.classes + self.beta) + (1 - label) * self.beta
            label[:, normal_loc] += self.alpha


        loss_c = loss_C(pred_label, label)
        loss_c = torch.mean(loss_c)
        # print('loss_c',loss_c.shape, loss_c)
        # print('loss_ae', loss_ae, 'loss_c', loss_c)
        return (1-self.c_loss_ratio)*loss_ae + self.c_loss_ratio*loss_c, loss_ae, loss_c



class ConvAEC(MetaAEC):
    def __init__(self, params):
        super(ConvAEC, self).__init__(params=params)

        # x: (batch, n_time, n_features)

        num_inputs = params.n_features
        seq_len = params.n_time
        classes = params.classes

        num_filters = params.num_filters
        embedding_dim = params.embedding_dim
        kernel_size = params.kernel_size
        dropout = params.dropout
        normalization = params.normalization
        stride = params.stride
        padding = params.padding
        classifier_dim = params.classifier_dim #32

        self.encoder = ConvEncoder(num_inputs, num_filters, embedding_dim, kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout, normalization=normalization)
        self.decoder = ConvDecoder(embedding_dim, num_filters, seq_len, num_inputs, kernel_size, stride=stride, padding=padding, dropout=dropout, normalization=normalization)
        self.classifier = NonLinClassifier(embedding_dim, classes, d_hidd=classifier_dim, dropout=dropout, norm=normalization)