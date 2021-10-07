import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tcn import TemporalConvNet
import torch
import utils

class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x_out = self.rnn(x_packed)[0]
        x_padded = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]
        return x_padded


class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y


class Model(nn.Module):
    # Bidirectional = False
    # Rnn_n_layer = 4
    # D_rnn = 64
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        print(params.d_in)
        self.inp = nn.Linear(params.d_in, params.d_rnn, bias=False)

        if params.rnn_n_layers > 0:
            self.rnn = RNN(params.d_rnn, params.d_rnn, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=0.2)

        d_rnn_out = params.d_rnn * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.d_rnn
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=0.0)

    def forward(self, x, x_len):
        x = self.inp(x)
        if self.params.rnn_n_layers > 0:
            x = self.rnn(x, x_len)
        y = self.out(x)
        return y


class MuseModel(nn.Module):
    def __init__(self,params, num_outputs =1, tcn_in = 512 , tcn_channels= (512, 512), num_dilations =4, tcn_kernel_size=3,
        dropout=0.2, use_norm= False, features_dropout=0., num_last_regress = 64, features ='vggface'):
        super(MuseModel, self).__init__()
        self.params = params
        self.num_stacks_tcn = len(tcn_channels)
        print("Stack TCN: "+ str(self.num_stacks_tcn))
        print("Use Norm: "+ str(use_norm))
        self.features = features
        if features_dropout > 0:
            self._dropout = nn.Dropout(p=features_dropout)
        else:
            self._dropout = None

        self._temporal = self.get_temporal_layers(tcn_in, tcn_channels, num_dilations, tcn_kernel_size, dropout,
                                                  use_norm)
        if params.rnn_n_layers > 0:
            self.rnn = RNN(tcn_channels[-1], params.d_rnn, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=0.2)
        self._regression = nn.Sequential(nn.Linear(params.d_rnn, num_last_regress, bias=False), nn.ReLU(), 
                                         nn.Linear(num_last_regress, num_outputs, bias=False))
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_temporal_layers(self, tcn_in, tcn_channels, num_dilations, tcn_kernel_size, dropout, use_norm):
        # input of TCN should have dimension (N, C, L)
        if self.num_stacks_tcn == 1:
            temporal_layers = TemporalConvNet(tcn_in, (tcn_channels[0],) * num_dilations, tcn_kernel_size, dropout,
                                              use_norm=use_norm)
        else:
            list_layers = []
            for idx in range(self.num_stacks_tcn):
                tcn_in_index = tcn_in if idx == 0 else tcn_channels[idx - 1]
                list_layers.append(
                    TemporalConvNet(tcn_in_index, (tcn_channels[idx],) * num_dilations, tcn_kernel_size, dropout,
                                    use_norm=use_norm))
            temporal_layers = nn.Sequential(*list_layers)

        return temporal_layers

    

    def forwardx(self, x,x_len, temporal_module, regression_module, feat_dropout):
        # Input has size batch_size x sequence_length x num_channels (N x L x C)
        if feat_dropout is not None:
            x = feat_dropout(x)

        # Transform to (N, C, L) first
        x = x.permute(0, 2, 1)
        x = temporal_module(x)
        # Transform back to (N, L, C)

        x = x.permute(0, 2, 1)

        if self.params.rnn_n_layers > 0:
            x = self.rnn(x, x_len)

        x = regression_module(x)
        return x
     
    def forward(self, x, x_len):
        pred_scores= self.forwardx(x,x_len, self._temporal, self._regression, self._dropout)
        return pred_scores

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.multihead_attn(src, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

class MuseModelWithSelfAttention(nn.Module):
    def __init__(self,params, num_outputs =1, num_last_regress = 64):
        super(MuseModelWithSelfAttention, self).__init__()
        self.params = params
        self.fea_dr=params.fea_dr
        self.num_stacks_tcn = len(self.params.tcn_channels)
        if self.fea_dr > 0:
            self.fea_dr_layer = nn.Dropout(self.fea_dr)
        else:
            self.fea_dr_layer = None
        if params.attn_layer>0:
            self.attn = SelfAttentionLayer(params.d_in, params.n_heads, dropout= params.attn_dr)
        if params.tcn_layer >0:
            self.tcn = self.get_temporal_layers(params.d_in, params.tcn_channels, params.num_dilations, params.tcn_kernel_size, params.tcn_dr,
                                                  params.tcn_norm)
        if params.rnn_n_layers > 0:
            self.rnn = RNN(params.d_rnn_in, params.d_rnn, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dr)

        self._regression = nn.Sequential(nn.Linear(params.d_rnn, num_last_regress, bias=False), nn.ReLU(), 
                                         nn.Linear(num_last_regress, num_outputs, bias=False))
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_temporal_layers(self, tcn_in, tcn_channels, num_dilations, tcn_kernel_size, dropout, use_norm):
        # input of TCN should have dimension (N, C, L)
        if self.num_stacks_tcn == 1:
            temporal_layers = TemporalConvNet(tcn_in, (tcn_channels[0],) * num_dilations, tcn_kernel_size, dropout,
                                              use_norm=use_norm)
        else:
            list_layers = []
            for idx in range(self.num_stacks_tcn):
                tcn_in_index = tcn_in if idx == 0 else tcn_channels[idx - 1]
                list_layers.append(
                    TemporalConvNet(tcn_in_index, (tcn_channels[idx],) * num_dilations, tcn_kernel_size, dropout,
                                    use_norm=use_norm))
            temporal_layers = nn.Sequential(*list_layers)

        return temporal_layers
     
    def forward(self, x, x_len):
        # Input has size batch_size x sequence_length x num_channels (B x L x C)
        if self.fea_dr > 0:
            x = self.fea_dr_layer(x)

        if self.params.attn_layer>0:
            x= x.transpose(0,1)  # (LxBxC)
            mask = utils.get_padding_mask(x, x_len)
            x= self.attn(x, src_key_padding_mask=mask )
            x= x.transpose(0,1)
        

        if self.params.tcn_layer > 0:
            # Transform to (B, C, L) first
            x = x.permute(0, 2, 1)
            x = self.tcn(x)
            # Transform back to (B, L, C)
            x = x.permute(0, 2, 1)

        if self.params.rnn_n_layers > 0:
            x = self.rnn(x, x_len)

        

        x= self._regression(x)
        return x

class MuseModel2(nn.Module):
    def __init__(self,params, num_outputs =1, tcn_in1 = 512 ,tcn_in2=512, tcn_channels1= (512, 512),tcn_channels2= (512,512), num_dilations =4, tcn_kernel_size=3,
        dropout=0.2, use_norm= False, features_dropout=0., num_last_regress = 128, features ='vggface',d_rnn1=64, d_rnn2=64):
        super(MuseModel2, self).__init__()
        self.params = params
        self.num_stacks_tcn = len(tcn_channels1)
        self.features = features
        self.features_dropout = features_dropout
        if self.features_dropout > 0:
            self._dropout = nn.Dropout(p=features_dropout)
        else:
            self._dropout = None

        self._temporal1 = self.get_temporal_layers(tcn_in1, tcn_channels1, num_dilations, tcn_kernel_size, dropout,
                                                  use_norm)
        if params.rnn_n_layers > 0:
            self.rnn1 = RNN(tcn_channels1[-1], d_rnn1, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=0.2)
            self.rnn2 = RNN(tcn_channels2[-1], d_rnn2, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=0.2)

        self._temporal2 = self.get_temporal_layers(tcn_in2, tcn_channels2, num_dilations, tcn_kernel_size, dropout,
                                                  use_norm)                                        
        self._regression = nn.Sequential(nn.Linear(d_rnn1 +d_rnn2, num_last_regress, bias=False), nn.ReLU(),
                                         nn.Linear(num_last_regress, num_outputs, bias=False))
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_temporal_layers(self, tcn_in, tcn_channels, num_dilations, tcn_kernel_size, dropout, use_norm):
        # input of TCN should have dimension (N, C, L)
        if self.num_stacks_tcn == 1:
            temporal_layers = TemporalConvNet(tcn_in, (tcn_channels[0],) * num_dilations, tcn_kernel_size, dropout,
                                              use_norm=use_norm)
        else:
            list_layers = []
            for idx in range(self.num_stacks_tcn):
                tcn_in_index = tcn_in if idx == 0 else tcn_channels[idx - 1]
                list_layers.append(
                    TemporalConvNet(tcn_in_index, (tcn_channels[idx],) * num_dilations, tcn_kernel_size, dropout,
                                    use_norm=use_norm))
            temporal_layers = nn.Sequential(*list_layers)

        return temporal_layers

    def forward(self, x, y, feature_len):
        # Input has size batch_size x sequence_length x num_channels (N x L x C)
        if self.features_dropout is not None:
            x = self._dropout(x)
            y = self._dropout(y)

        # Transform to (N, C, L) first
        x = x.permute(0, 2, 1)
        x = self._temporal1(x)

        y = y.permute(0, 2, 1)
        y = self._temporal2(y)
        # Transform back to (N, L, C)

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        if self.params.rnn_n_layers > 0:
            x = self.rnn1(x, feature_len)
            y = self.rnn2(y, feature_len)
        cat = torch.cat((x, y), 2)
        pred = self._regression(cat)
        return pred

class BiCrossAttention(nn.Module):
    def __init__(self, d_model1, d_model2, d_model3,
                 d_out, attn_dropout=None):
        super(BiCrossAttention, self).__init__()
        self.d_k1 = d_out
        self.d_v1 = d_out
        self.d_k2 = d_out
        self.d_v2 = d_out
        self. attn_dropout = attn_dropout
        self.w_qs = nn.Linear(d_model1, d_out, bias=False)
        self.w_qs1 = nn.Linear(d_model1, d_out, bias=False)
        self.w_qs2 = nn.Linear(d_model1, d_out, bias=False)
        self.w_ks1 = nn.Linear(d_model2, d_out, bias=False)
        self.w_ks2 = nn.Linear(d_model3, d_out, bias=False)
        self.w_vs1 = nn.Linear(d_model2, d_out, bias=False)
        self.w_vs2 = nn.Linear(d_model3, d_out, bias=False)

        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter((torch.zeros(1)).cuda())
        if self.attn_dropout is not None:
            self.dropout1 = nn.Dropout(attn_dropout)
            self.dropout2 = nn.Dropout(attn_dropout)


    def forward(self, q, k1, v1, k2, v2):
        # Linear
        qs= self.w_qs(q)
        qs1 = self.w_qs1(q) #B x L x d_out
        qs2 = self.w_qs2(q) #B x L x d_out
        ks1 = self.w_ks1(k1) #B x L x d_out
        ks2 = self.w_ks2(k2) #B x L x d_out
        vs1 = self.w_vs1(v1) #B x L x d_out
        vs2 = self.w_vs2(v2) #B x L x d_out

        # Attention
        attn1 = torch.matmul(qs1, ks1.transpose(1, 2))
        attn1 = self.softmax1(attn1)

        attn2 = torch.matmul(qs2, ks2.transpose(1, 2))
        attn2 = self.softmax2(attn2)

        if self.attn_dropout is not None:
            attn1 = self.dropout1(attn1)
            attn2 = self.dropout2(attn2)


        # Attention output
        o1= torch.matmul(attn1,vs1)
        o2 = torch.matmul(attn2,vs2)

        o = o1 + o2 #B x L x dK1 + B x L x dv2

        # Output
        out = self.gamma * o + qs
        return out




class MuseModelBiCrossAttention(nn.Module):
    def __init__(self,params, num_outputs =1, num_last_regress = 64, features ='vggface', dropout=0., d_attention_out=128, features_dropout=None, attn_dropout=None):
        super(MuseModelBiCrossAttention, self).__init__()
        self.features_dropout = features_dropout
        if self.features_dropout is not None:
            self.feature_dropout1 = nn.Dropout(features_dropout)
            self.feature_dropout2 = nn.Dropout(features_dropout)
            self.feature_dropout3 = nn.Dropout(features_dropout)

        self.params = params
        self.features = features
        self.d_visual_in = params.rnn_in['visual']
        self.d_audio_in = params.rnn_in['audio']
        self.d_text_in = params.rnn_in['text']

        self.d_visual_out = params.rnn_out['visual']
        self.d_audio_out = params.rnn_out['audio']
        self.d_text_out = params.rnn_out['text']
        self.d_attention_out= d_attention_out

        if params.rnn_n_layers > 0:
            self.rnn1 = RNN(self.d_visual_in, self.d_visual_out, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=dropout)
            self.rnn2 = RNN(self.d_audio_in, self.d_audio_out, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=dropout)            
            self.rnn3 = RNN(self.d_text_in, self.d_text_out, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=dropout)
        self.biCrossAtt1 = BiCrossAttention(self.d_visual_out, self.d_audio_out, self.d_text_out, d_out=d_attention_out, attn_dropout=attn_dropout)
        self.biCrossAtt2 = BiCrossAttention(self.d_audio_out, self.d_visual_out, self.d_text_out, d_out=d_attention_out, attn_dropout= attn_dropout)
        self.biCrossAtt3 = BiCrossAttention(self.d_text_out, self.d_audio_out, self.d_visual_out, d_out=d_attention_out, attn_dropout= attn_dropout)

        self._regression = nn.Sequential(nn.Linear(d_attention_out*3, num_last_regress, bias=False), nn.ReLU(), 
                                         nn.Linear(num_last_regress, num_outputs, bias=False), nn.Tanh())
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, v, a, t, feature_len):
        if self.features_dropout is not None:
            v = self.feature_dropout1(v)
            a = self.feature_dropout2(a)
            t = self.feature_dropout3(t)
        # Input has size batch_size x sequence_length x num_channels (B x L x C)
        if self.params.rnn_n_layers > 0:
            visual = self.rnn1(v, feature_len)
            audio = self.rnn2(a, feature_len)
            text = self.rnn3(t, feature_len)
        att1 = self.biCrossAtt1(visual, audio, audio, text, text)
        att2 = self.biCrossAtt2(audio, visual, visual, text, text)
        att3= self.biCrossAtt3(text,audio, audio, visual, visual)

        cat = torch.cat((att1, att2, att3), 2)
        pred = self._regression(cat)
        return pred

