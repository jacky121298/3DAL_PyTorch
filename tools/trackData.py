import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', help='Path to working dir.')
    parser.add_argument('--split', default=16, help='Number of train split.')
    args = parser.parse_args()

    if args.work_dir.split('/')[-1] == 'train':
        track = {}
        for i in range(args.split):
            with open(os.path.join(args.work_dir, f'trackData_{i}.pkl'), 'rb') as f:
                track_split = pickle.load(f)
            track = dict(list(track.items()) + list(track_split.items()))
    elif args.work_dir.split('/')[-1] == 'val':
        with open(os.path.join(args.work_dir, 'trackData.pkl'), 'rb') as f:
            track = pickle.load(f)
    else:
        raise NotImplementedError(f'Not supported.')
    
    tracking = {}
    for token, frame in tqdm(track.items()):
        ids, types, bboxs = frame['id'], frame['type'], frame['bbox']
        scores, points, matchs = frame['score'], frame['point'], frame['match']
        
        for idx in range(len(ids)):
            if ids[idx] not in tracking:
                tracking[ids[idx]] = {}
                tracking[ids[idx]]['type'] = [types[idx]]
                tracking[ids[idx]]['bbox'] = [bboxs[idx]]
                tracking[ids[idx]]['score'] = [scores[idx]]
                tracking[ids[idx]]['point'] = [points[idx]]
                tracking[ids[idx]]['match'] = [matchs[idx]]
                tracking[ids[idx]]['token'] = [token]
            else:
                tracking[ids[idx]]['type'].append(types[idx])
                tracking[ids[idx]]['bbox'].append(bboxs[idx])
                tracking[ids[idx]]['score'].append(scores[idx])
                tracking[ids[idx]]['point'].append(points[idx])
                tracking[ids[idx]]['match'].append(matchs[idx])
                tracking[ids[idx]]['token'].append(token)
    
    if args.work_dir.split('/')[-1] == 'train':
        tracking_list = list(tracking.items())
        for i in range(args.split):
            tracking_split = dict(tracking_list[len(tracking_list) * i // args.split: len(tracking_list) * (i + 1) // args.split])
            with open(os.path.join(args.work_dir, f'track_{i}.pkl'), 'wb') as f:
                pickle.dump(tracking_split, f)
    elif args.work_dir.split('/')[-1] == 'val':
        with open(os.path.join(args.work_dir, 'track.pkl'), 'wb') as f:
            pickle.dump(tracking, f)
    else:
        raise NotImplementedError(f'Not supported.')

if __name__ == '__main__':
    main()