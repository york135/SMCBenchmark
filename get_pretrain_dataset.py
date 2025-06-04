import os
from pathlib import Path
import glob
import pickle
import pathlib
import argparse
import numpy as np
from read_data import *


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path ###
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8"
        , "emopia", "tagatraum", "lmd"])

    ### output ###    
    parser.add_argument('--output_dir', default="Data/")

    args = parser.parse_args()
    return args


def extract(files, args, model):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)
    print(f'Number of files: {len(files)}') 

    segments = model.prepare_pretrain_data(files)
    output_file = os.path.join(args.output_dir, f'{args.dataset}.npy')

    np.save(output_file, segments)
    print(f'Data len: {len(segments)}, saved at {output_file}')

def main(): 
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = CP()

    if args.dataset == 'pop909':
        dataset = 'pop909_processed'
    elif args.dataset == 'emopia':
        dataset = 'EMOPIA_1.0'
    elif args.dataset == 'pianist8':
        dataset = 'joann8512-Pianist8-ab9f541'

    if args.dataset == 'pop909' or args.dataset == 'emopia':
        train_files = glob.glob(f'../MIDI-BERT-CP_classical/Data/Dataset/{dataset}/train/*.mid')
        valid_files = glob.glob(f'../MIDI-BERT-CP_classical/Data/Dataset/{dataset}/valid/*.mid')
        test_files = glob.glob(f'../MIDI-BERT-CP_classical/Data/Dataset/{dataset}/test/*.mid')
        files = sorted(train_files + valid_files + test_files)

    elif args.dataset == 'pianist8':
        train_files = glob.glob(f'../MIDI-BERT-CP_classical/Data/Dataset/{dataset}/train/*/*.mid')
        valid_files = glob.glob(f'../MIDI-BERT-CP_classical/Data/Dataset/{dataset}/valid/*/*.mid')
        test_files = glob.glob(f'../MIDI-BERT-CP_classical/Data/Dataset/{dataset}/test/*/*.mid')
        files = sorted(train_files + valid_files + test_files)

    elif args.dataset == 'tagatraum':
        train_files = glob.glob(f'../tagatraum/train_clean/*/*.mid')
        valid_files = glob.glob(f'../tagatraum/valid_clean/*/*.mid')
        test_files = glob.glob(f'../tagatraum/test_clean/*/*.mid')
        files = sorted(train_files + valid_files + test_files)

    elif args.dataset == 'lmd':
        files = glob.glob(f'../lmd_matched/*/*/*/*/*.mid')

    elif args.dataset == 'pop1k7':
        files = glob.glob('../MIDI-BERT-CP_classical/Data/Dataset/Pop1K7/midi_transcribed/*/*.midi')

    elif args.dataset == 'ASAP':
        files = pickle.load(open('../MIDI-BERT-CP_classical/Data/Dataset/ASAP_song.pkl', 'rb'))
        files = [f'../MIDI-BERT-CP_classical/Data/Dataset/asap-dataset/{file}' for file in files]

    else:
        print('not supported')
        exit(1)

    extract(sorted(files), args, model)

if __name__ == '__main__':
    main()
