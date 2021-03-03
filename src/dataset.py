import os
from os import listdir
from os.path import isfile, join
import torch
from torch.utils.data import Dataset
import pickle
class MovieDataset(Dataset):
    def __init__(self, input_path,window_size=10):
        self.input_path = input_path
        # Make list of in_files and out_files
        self.file_names = [os.path.join(input_path, f) for f in listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        self.window_size = window_size
        #Make index to window dict
        #Given an index (from the dataloader), we need to find the movie and starting index of the window (in the movie)
        num_windows_dataset  = 0
        index_to_window_dict = {}
        current_index_for_dict = 0
        for data_path in self.file_names:
            with open(data_path, 'rb') as f:
                
                data = pickle.load(f)
            num_shots = data['shot_end_frame'].shape[0]
            num_windows = num_shots - window_size+1
            num_windows_dataset += num_windows
            for jj in range(num_windows):
                index_to_window_dict[current_index_for_dict] = (data_path,jj)
                current_index_for_dict += 1
        self.index_to_window_dict = index_to_window_dict
        
        

    def __len__(self):
        return len(self.index_to_window_dict)

    def __getitem__(self, idx):
        data_path,window_start_idx = self.index_to_window_dict[idx]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        ii = window_start_idx
        return data['place'][ii:ii+self.window_size,:], \
                data['cast'][ii:ii+self.window_size,:], \
                data['action'][ii:ii+self.window_size,:], \
                data['audio'][ii:ii+self.window_size,:], \
                data['scene_transition_boundary_ground_truth'][ii:ii+self.window_size-1].float()