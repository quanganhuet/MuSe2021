import argparse
import os
import sys
import torch
import numpy
import random
from datetime import datetime
from dateutil import tz
from torch.nn import CrossEntropyLoss
from os.path import isdir
import json
import config
import matplotlib.pyplot as plt
from utils import Logger, seed_worker
from train import train_model
from eval import evaluate
from model import Model, MuseModel, MuseModel2, MuseModelBiCrossAttention, MuseModelWithSelfAttention
from loss import CCCLoss, MSELoss, MAELoss
from dataset import MuSeDataset
from data_parser import load_data


def parse_args():
    parser = argparse.ArgumentParser(description='Time-Continuous Emotion Recognition for MuSe 2021.')

    parser.add_argument('--task', type=str, required=True, choices=['wilder', 'sent', 'physio', 'stress'],
                        help='Specify the task (wilder, sent, physio or stress).')
    parser.add_argument('--feature_set', nargs='+', required=True,
                        help='Specify the features (one or several, required).')
    parser.add_argument('--emo_dim', default='arousal',
                        help='Specify the emotion dimension (default: arousal).')
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--norm_opts', type=str, nargs='+', default=['n'],
                        help='Specify which features to normalize ("y": yes, "n": no) in the corresponding order to '
                             'feature_set (default: n).')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--d_rnn', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true', default=False, 
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--save', action='store_true',
                        help='Specify whether to save the best model (default: False).')
    parser.add_argument('--save_path', type=str, default='preds',
                        help='Specify path where to save the predictions (default: preds).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved (default: False).')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')

    args = parser.parse_args()
    return args


def main(args):
    print("Save prediction:"+ str(args.save))
    # ensure reproducibility
    numpy.random.seed(10)
    random.seed(10)
    print('Loading data ...')
    encoding_position= True
    data = load_data(args.task, args.paths, args.feature_set, args.emo_dim, args.normalize, args.norm_opts,
                     args.win_len, args.hop_len, save=args.cache, encoding_position=encoding_position)
    data_loader = {}
    for partition in data.keys():  # one DataLoader for each partition
        set = MuSeDataset(data, partition)
        batch_size = args.batch_size if partition == 'train' else 1
        shuffle = True if partition == 'train' else False  # shuffle only for train partition
        data_loader[partition] = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                                             worker_init_fn=seed_worker)
        
    # args.d_in = data_loader['train'].dataset.get_feature_dim()
    args.n_targets = 1
    criterion = CCCLoss()
    score_str = 'CCC'
    d_in= {'vggface': 512, "egemaps": 88, "bert-4":  768, 'fau_intensity':17, 'deepspectrum': 4096, 'vggish': 128}
    d_rnn_in= {'vggface': 256, "egemaps": 128, "bert-4":  512,'fau_intensity':16, 'deepspectrum': 512, 'vggish': 128}
    # rnn_out= {'visual': 256, "audio": 128, "text": 256, "bio": 128}
    # args.rnn_in = rnn_in
    # args.rnn_out = rnn_out
    # tcn_ins = {'vggface': 512, 'egemaps':88, 'bert-4': 768, 'fau_intensity':17,'deepspectrum': 4096, 'vggish': 128}
    tcn_channels = {'vggface': (256,), 'egemaps':(128,), 'bert-4': (512,), 'fau_intensity':(16,), 'deepspectrum': (512,),'vggish': (128,)}
    d_rnn= {'vggface': 128, 'egemaps':64, 'bert-4': 128, 'fau_intensity':16, 'deepspectrum': 512, 'vggish': 128 }
    # model_file = 'egemaps_model_101.pth'
    # if args.save:  # Save predictions for all partitions (needed to subsequently do late fusion)
    #     print('Save all predictions...')
    #     # seed = int(model_file.split('_')[-3])
    #     seed= 101
    #     torch.manual_seed(seed)
       
    #     print(torch.cuda.is_available())
    #     best_model = torch.load(model_file)
    #     # Load data again without any segmentation
    #     data = load_data(args.task, args.paths, args.feature_set, args.emo_dim, args.normalize, args.norm_opts,
    #                      args.win_len, args.hop_len, save=args.cache, apply_segmentation=False)
    #     for partition in data.keys():
    #         dl = torch.utils.data.DataLoader(MuSeDataset(data, partition), batch_size=1, shuffle=False,
    #                                          worker_init_fn=seed_worker)
    #         evaluate(args.task, best_model, dl, criterion, args.use_gpu, predict=True,
    #                  prediction_path=args.paths['save'])
    #     return
    if args.eval_model is None:  # Train and validate for each seed
        seeds = range(args.seed, args.seed + args.n_seeds)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            #params setting
            if encoding_position:
                args.d_in = d_in[args.feature_set[0]]+1
            else:
                args.d_in = d_in[args.feature_set[0]]

            args.fea_dr=0.

            args.tcn_layer =1
            args.tcn_channels= tcn_channels[args.feature_set[0]]
            args.num_dilations=4
            args.tcn_kernel_size=3
            args.tcn_dr=0.1
            args.tcn_norm= True

            args.d_rnn_in = d_rnn_in[args.feature_set[0]]
            args.d_rnn = d_rnn[args.feature_set[0]]
            args.rnn_dr=0.2

            #Attention Parameter
            args.attn_layer=1
            args.n_heads=1
            args.attn_dr= 0.2



            # model = Model(args)
            # model= MuseModel(args, num_outputs=1, tcn_in= tcn_in, tcn_channels= bio_signal_tcn_channel, num_dilations=8,
            # dropout=0.2, use_norm=True, features_dropout=0.)
            # model=  MuseModelBiCrossAttention(args, dropout=0.2, features_dropout=None, attn_dropout=0.2, num_last_regress=32, d_attention_out=32)
            model= MuseModelWithSelfAttention(args, num_outputs =1, num_last_regress = 64)
            # model= MuseModel2(args, num_outputs=1, tcn_in1= tcn_in1,tcn_in2= tcn_in2,
            #  tcn_channels1= tcn_channels[args.feature_set[0]],tcn_channels2= tcn_channels[args.feature_set[0]], num_dilations=8,
            #  dropout=0.2, use_norm=True, features_dropout=0.2, d_rnn1= d_rnn1, d_rnn2= d_rnn2)
            print('=' * 50)
            print('Training model... [seed {}]'.format(seed))

            val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
                                                               args.lr, args.paths['model'], seed, args.use_gpu,
                                                               criterion, regularization=args.regularization)
            if not args.predict:  # run evaluation only if test labels are available
                test_loss, test_score = evaluate(args.task, model, data_loader['test'], criterion, args.use_gpu)
                test_scores.append(test_score)
                if args.task in ['physio', 'stress', 'wilder']:
                    print(f'[Test CCC]:  {test_score:7.4f}')
            # val_losses.append(val_loss)
            val_scores.append(val_score)
            best_model_files.append(best_model_file)
            # epochs= range(1,51)
            # plt.plot(epochs, tran_l, 'g', label='Training Loss')
            # plt.plot(epochs, val_l, 'b', label='Validation Loss')
            # plt.plot(epochs, val_s, 'r', label='Validation CCC')
            # plt.title(f'Training/Validation Loss and Score seed{seed}')
            # plt.xlabel('Epochs')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.show()

        best_idx = val_scores.index(max(val_scores))  # find best performing seed
        print('=' * 50)
        print(f'Best {score_str} on [Val] for seed {seeds[best_idx]}: '
              f'[Val {score_str}]: {val_scores[best_idx]:7.4f}'
              f"{f' | [Test {score_str}]: {test_scores[best_idx]:7.4f}' if not args.predict else ''}")
        print('=' * 50)

        model_file = best_model_files[best_idx]  # best model of all of the seeds

    else:  # Evaluate existing model (No training)
        model_file = args.eval_model
        model = torch.load(model_file)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], criterion, args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {score_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], criterion, args.use_gpu)
            print(f'[Test {score_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting test samples...')
        best_model = torch.load(model_file)
        evaluate(args.task, best_model, data_loader['test'], criterion, args.use_gpu, predict=True,
                 prediction_path=args.paths['predict'])
    if args.save:  # Save predictions for all partitions (needed to subsequently do late fusion)
        print('Save all predictions...')
        seed = int(model_file.split('_')[-3])
        torch.manual_seed(seed)
        best_model = torch.load(model_file)
        # Load data again without any segmentation
        data = load_data(args.task, args.paths, args.feature_set, args.emo_dim, args.normalize, args.norm_opts,
                         args.win_len, args.hop_len, save=args.cache, apply_segmentation=False)
        for partition in data.keys():
            dl = torch.utils.data.DataLoader(MuSeDataset(data, partition), batch_size=1, shuffle=False,
                                             worker_init_fn=seed_worker)
            evaluate(args.task, best_model, dl, criterion, args.use_gpu, predict=True,
                     prediction_path=args.paths['save'])
    

    # Delete model if save option is not set.
    if not args.save and not args.eval_model:
        if os.path.exists(model_file):
            os.remove(model_file)

    print('Done.')


if __name__ == '__main__':
    args = parse_args()
    print(f"BI: {args.rnn_bi}")
    print(f"Normalize: {args.normalize}")
    args.log_file_name = '{}_[{}]_[{}]_[{}_{}_{}]_[{}_{}]'.format(
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), '_'.join(args.feature_set), args.emo_dim,
        args.d_rnn, args.rnn_n_layers, args.rnn_bi, args.lr, args.batch_size)

    # adjust your paths in config.py
    args.paths = {'log': os.path.join(config.LOG_FOLDER, args.task).replace('\\','/'),
                  'data': os.path.join(config.DATA_FOLDER, args.task).replace('\\','/'),
                  'model': os.path.join(config.MODEL_FOLDER, args.task, args.log_file_name).replace('\\','/')}
    if args.predict:
        args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.log_file_name).replace('\\','/')
    if args.save:
        args.paths['save'] = os.path.join(args.save_path, args.task, args.log_file_name).replace('\\','/')
    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    args.paths.update({'features': config.PATH_TO_ALIGNED_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})
    print(isdir(args.paths['partition']))
    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt').replace('\\','/'))
    print(' '.join(sys.argv))
    main(args)
