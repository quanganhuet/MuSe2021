import os
import numpy as np
import pandas as pd
import pickle


def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)
    for row in df.values:
        vid, partition = str(row[0]), row[-1]
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[partition]:
            partition2vid[partition].append(vid)

    return vid2partition, partition2vid


def segment_sample(sample, win_len, hop_len, segment_type='normal'):
    #win_len =300, hop_len =50
    segmented_sample = []
    assert hop_len <= win_len and win_len >= 10
    if segment_type == 'normal':
        for s_idx in range(0, len(sample), hop_len):
            e_idx = min(s_idx + win_len, len(sample))
            segment = sample.iloc[s_idx:e_idx]
            segmented_sample.append(segment)
            if e_idx == len(sample):
                break
    else:
        print('No such segmentation available.')
    return segmented_sample


def normalize_data(data, idx_list, column_name='feature'):
    train_data = np.row_stack(data['train'][column_name])
    train_mean = np.nanmean(train_data, axis=0)
    train_std = np.nanstd(train_data, axis=0)

    for partition in data.keys():
        for i in range(len(data[partition][column_name])):
            for s_idx, e_idx in idx_list:
                data[partition][column_name][i][:, s_idx:e_idx] = \
                    (data[partition][column_name][i][:, s_idx:e_idx] - train_mean[s_idx:e_idx]) / (
                            train_std[s_idx:e_idx] + 1e-6)  # standardize
                data[partition][column_name][i][:, s_idx:e_idx] = np.where(  # replace any nans with zeros
                    np.isnan(data[partition][column_name][i][:, s_idx:e_idx]), 0.0,
                    data[partition][column_name][i][:, s_idx:e_idx])

    return data


def load_data(task, paths, feature_set, emo_dim, normalize=True, norm_opts=None, win_len=200, hop_len=100, save=False,
              apply_segmentation=True, encoding_position= True):
    print("Apply_Segmentation="+str(apply_segmentation))
    # task==stress
    #paths={
    # 'log': '/Muse2021/results/log_muse/stress', 
    # 'data': '/Muse2021/results/data_muse/stress', 
    # 'model': '/Muse2021/results/model_muse/stress/2021-05-11-13-17_[vggface]_[arousal]_[64_4_True]_[0.002_1024]', 
    # 'save': 'preds/stress/2021-05-11-13-17_[vggface]_[arousal]_[64_4_True]_[0.002_1024]',
    #  'features': '/Muse2021/data/c3_muse_stress/feature_segments',
    #  'labels': '/Muse2021/data/c3_muse_stress/label_segments',
    #  'partition': '/Muse2021/data/c3_muse_stress/metadata/partition.csv'
    # }
    #predict=False, regularization=0.0, rnn_bi=True, rnn_n_layers=4, save=True,
    #  save_path='preds', seed=101, task='stress', use_gpu=True, win_len=300

    feature_path = paths['features']
    label_path = paths['labels']

    data_file_name = f'data_{task}_{"_".join(feature_set)}_{emo_dim}_{"norm_" if normalize else ""}{win_len}_' \
        f'{hop_len}{"_seg" if apply_segmentation else ""}.pkl'
    data_file = os.path.join(paths['data'], data_file_name).replace('\\', '/')
    # if os.path.exists(data_file):  # check if file of preprocessed data exists
    #     print(f'Find cached data "{os.path.basename(data_file)}".')
    #     data = pickle.load(open(data_file, 'rb'))
    #     return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [],"vggface":[] , "egemaps": [], "bert":[], "timestamp":[],'label': [], 'meta': []},
            'devel': {'feature': [],"vggface":[] , "egemaps": [], "bert":[], "timestamp":[], 'label': [], 'meta': []},
            'test': {'feature': [],"vggface":[] , "egemaps": [],"bert":[], "timestamp":[], 'label': [], 'meta': []}}
    vid2partition, partition2vid = get_data_partition(paths['partition'])

    for partition, vids in partition2vid.items():
        for vid in vids:
            sample_data = []
            for i, feature in enumerate(feature_set):
                # parse feature
                feature_file = os.path.join(feature_path, feature, vid + '.csv')
                df = pd.read_csv(feature_file)
                if i == 0:
                    feature_data = df  # keep timestamp and segment id in 1st feature val
                else: 
                    feature_data= df.iloc[:, 2:]
                sample_data.append(feature_data)

            # parse labels
            label_file = os.path.join(label_path, emo_dim, vid + '.csv')
            df = pd.read_csv(label_file)
            timestampt = df.iloc[:,0]
            encoding_timestampt =timestampt/10e5

            label_data = pd.DataFrame(data=df['value'].values, columns=[emo_dim])
            if encoding_position:
                sample_data.append(encoding_timestampt)
            sample_data.append(label_data)

            # list with 3 Dataframe Item
            # concat
            sample_data = pd.concat(sample_data, axis=1)
            # vgg_face = pd.read_csv("{}/{}/{}.csv".format(feature_path, 'vggface',vid))
            # egemaps = pd.read_csv("{}/{}/{}.csv".format(feature_path, 'egemaps',vid))
            # bert=  pd.read_csv("{}/{}/{}.csv".format(feature_path, 'bert-4',vid))
            # if encoding_position:
            #     vgg_face_features = pd.concat([vgg_face.iloc[:, 2:], encoding_timestampt], axis=1)
            #     egemaps_features = pd.concat([egemaps.iloc[:, 2:], encoding_timestampt], axis=1)
            #     bert = pd.concat([bert.iloc[:, 2:], encoding_timestampt], axis=1)
            # else:
            #     vgg_face_features = vgg_face.iloc[:, 2:]
            #     egemaps_features = egemaps.iloc[:, 2:]
            #     bert = egemaps.iloc[:, 2:]
            # remove missing data
            if partition != 'test':
                sample_data = sample_data.dropna()

            # segment
            if apply_segmentation and partition == "train":
                samples = segment_sample(sample_data, win_len, hop_len, 'normal')
                # vgg_face_features = segment_sample(vgg_face_features, win_len, hop_len, 'normal')
                # egemaps_features = segment_sample(egemaps_features, win_len, hop_len, 'normal')
                # bert= segment_sample(bert, win_len, hop_len, 'normal')
            else:
                samples = [sample_data]
                # vgg_face_features= [vgg_face_features]
                # egemaps_features= [egemaps_features]
                # bert= [bert]
            # store
            for i, segment in enumerate(samples):  # each segment has columns: timestamp, segment_id, features, labels
                if len(segment.iloc[:, 2:-1].to_numpy()) > 0:  # check if there are features
                    meta = np.column_stack((np.array([int(vid)] * len(segment)),
                                            segment.iloc[:, :2].to_numpy()))  # video_id, timestamp, segment_id
                    data[partition]['meta'].append(meta)
                    data[partition]['label'].append(segment.iloc[:, -1:].to_numpy())
                    data[partition]['feature'].append(segment.iloc[:, 2:-1].to_numpy())
            # for i, segment in enumerate(vgg_face_features):
            #     data[partition]['vggface'].append(segment.to_numpy())
            # for i, segment in enumerate(egemaps_features):
            #     data[partition]['egemaps'].append(segment.to_numpy())
            # for i, segment in enumerate(bert):
            #     data[partition]['bert'].append(segment.to_numpy())
            
    print("Normalize value: {}".format(str(normalize)))
    # if normalize:
    #     idx_list = []
    #     assert norm_opts is not None and len(norm_opts) == len(feature_set)
    #     norm_opts = [True if norm_opt == 'y' else False for norm_opt in norm_opts]
    #     print(f'Feature dims: {feature_dims} ({feature_set})')
    #     feature_dims = np.cumsum(feature_dims).t
    #     feature_dims = [0] + feature_dims
    #     print(feature_dims)
    #     norm_feature_set = []  # normalize data per feature and only if norm_opts is True
    #     for i, (s_idx, e_idx) in enumerate(zip(feature_dims[0:-1], feature_dims[1:])):
    #         norm_opt, feature = norm_opts[i], feature_set[i]
    #         if norm_opt:
    #             norm_feature_set.append(feature)
    #             idx_list.append([s_idx, e_idx])
    #     print(f'Normalized features: {norm_feature_set}')
    #     data = normalize_data(data, idx_list)
        
    print("Save preprocessed Data: {}".format(str(save)))
    if save:  # save loaded and preprocessed data
        print('Saving data...')
        pickle.dump(data, open(data_file, 'wb'))
    
    return data

if __name__ == '__main__':
    load_data()