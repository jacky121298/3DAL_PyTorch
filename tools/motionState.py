import gc
import os
import torch
import random
import pickle
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

def fixSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def trackFeature(track: dict, trackGT: dict, training: bool=False):
    new_track = {}
    for trackID, obj in track.items():
        match = obj['match'][-1]
        bbox = np.array(obj['bbox'])
        types = np.array(obj['type'])
        point = np.vstack(obj['point'])
        if match == None or bbox.shape[0] < 7 or types[0] == 2 or point.shape[0] == 0:
            continue
        new_track[trackID] = obj
    
    trainX, trainY = [], []
    if training:
        static, dynamic = {}, {}

    for trackID, obj in tqdm(new_track.items()):
        match = obj['match'][-1]
        bbox = np.array(obj['bbox'])

        distance = np.linalg.norm(bbox[0, :3] - bbox[-1, :3])
        var = np.linalg.norm(np.var(bbox[:, :3], axis=0))
        trainX.append([distance, var])
        if int(trackGT[match]['static']) == 0:
            trainY.append(0)
        else:
            trainY.append(1)
        
        if training:
            if int(trackGT[match]['static']) == 0:
                dynamic[trackID] = obj
            else:
                static[trackID] = obj

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    if training:
        return trainX, trainY, static, dynamic
    return trainX, trainY, new_track

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_train', help='Path to train track data.')
    parser.add_argument('--track_val', help='Path to val track data.')
    parser.add_argument('--split', default=16, help='Number of train split.')
    args = parser.parse_args()

    # Fix the random seed
    fixSeed(seed=10922081)

    ###### Pre-processing data ######
    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Reading train data')
    track_train = {}
    for i in range(args.split):
        with open(os.path.join(args.track_train, f'track_{i}.pkl'), 'rb') as f:
            track_train_split = pickle.load(f)
        track_train = dict(list(track_train.items()) + list(track_train_split.items()))

    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Reading train GT data')
    with open(os.path.join(args.track_train, 'trackGT.pkl'), 'rb') as f:
        trackGT_train = pickle.load(f)
    
    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Processing train data')
    trainX, trainY, static, dynamic = trackFeature(track_train, trackGT_train, training=True)
    del track_train, trackGT_train
    gc.collect()

    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Saving train/trackStatic.pkl')
    static_list = list(static.items())
    for i in range(args.split):
        static_split = dict(static_list[len(static_list) * i // args.split: len(static_list) * (i + 1) // args.split])
        with open(os.path.join(args.track_train, f'trackStatic_{i}.pkl'), 'wb') as f:
            pickle.dump(static_split, f)
    
    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Saving train/trackDynamic.pkl')
    dynamic_list = list(dynamic.items())
    for i in range(args.split):
        dynamic_split = dict(dynamic_list[len(dynamic_list) * i // args.split: len(dynamic_list) * (i + 1) // args.split])
        with open(os.path.join(args.track_train, f'trackDynamic_{i}.pkl'), 'wb') as f:
            pickle.dump(dynamic_split, f)

    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Reading val data')
    with open(os.path.join(args.track_val, 'track.pkl'), 'rb') as f:
        track_val = pickle.load(f)
    
    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Reading val GT data')
    with open(os.path.join(args.track_val, 'trackGT.pkl'), 'rb') as f:
        trackGT_val = pickle.load(f)
    
    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Processing val data')
    valX, valY, new_track_val = trackFeature(track_val, trackGT_val, training=False)
    del track_val, trackGT_val
    gc.collect()

    ###### Start motion state classification ######
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Number of train: {trainX.shape[0]}')
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Number of val: {valX.shape[0]}')
    
    # clf = make_pipeline(StandardScaler(), SGDClassifier())
    clf = SVC(kernel='linear')
    clf = clf.fit(trainX, trainY)
    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Score on test set: {clf.score(valX, valY)}')
    y_pred = clf.predict(valX)

    trackStatic = {}
    trackDynamic = {}
    for idx, pred in enumerate(y_pred):
        trackID, obj = list(new_track_val.items())[idx]
        if pred == 1:
            trackStatic[trackID] = obj
        else: trackDynamic[trackID] = obj

    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Saving val/trackStatic.pkl')
    with open(os.path.join(args.track_val, 'trackStatic.pkl'), 'wb') as f:
        pickle.dump(trackStatic, f)
    del trackStatic
    gc.collect()

    print(f'{bcolors.OKCYAN}>{bcolors.ENDC} Saving val/trackDynamic.pkl')
    with open(os.path.join(args.track_val, 'trackDynamic.pkl'), 'wb') as f:
        pickle.dump(trackDynamic, f)
    del trackDynamic
    gc.collect()

if __name__ == '__main__':
    main()