import os
from os import listdir
from os.path import isfile, join
import torch
from torch.utils.data import Dataset
import pickle
class MovieDataset(Dataset):
    '''
    The key variables here are self.index_to_window_dict and self.data_path_to_data_dict.
    Given a *dataloader* index, we need to find the specific movie and window that the index refers to.
    We do this using two dictionaries.
    
        * index_to_window_dict is a dictionary of dataloader indices -> (data_path,window_start_index)
        - The data path is a string (which will be looked up in the next dict), and essentially refers to the movie.
        - The window_start_index is the first index of the window.
            We can get the entire window by slicing [window_start_index,window_start_index+window_size]
        * data_path_to_data_dict is a dictionary of data_paths -> loaded pickle data
        - loaded pickle data is just what you get when you call pickle.load on the file
    Notes:
        We choose to load ALL the data into the class because theres not much data, and it speeds up training to not have to constantly open files
        If we had more data, this probably wouldn't be feasible.
        
    
    '''
    def __init__(self, input_path,window_size=10):
        self.input_path = input_path
        # Make list of in_files and out_files
        self.file_names = [os.path.join(input_path, f) for f in listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        self.window_size = window_size
        #Make index to window dict
        #Given an index (from the dataloader), we need to find the movie and starting index of the window (in the movie)
        index_to_window_dict = {}
        current_index_for_dict = 0
        data_path_to_data_dict={}
        for data_path in self.file_names:
            with open(data_path, 'rb') as f:
                
                data = pickle.load(f)
                data_path_to_data_dict[data_path] = data
            
            num_shots = data['shot_end_frame'].shape[0]
            num_windows = num_shots - window_size+1
            for jj in range(num_windows):
                index_to_window_dict[current_index_for_dict] = (data_path,jj)
                current_index_for_dict += 1
        #print(data_path_to_data_dict.keys())
        self.index_to_window_dict = index_to_window_dict
        self.data_path_to_data_dict = data_path_to_data_dict
        

    def __len__(self):
        return len(self.index_to_window_dict)

    def __getitem__(self, idx):
        data_path,window_start_idx = self.index_to_window_dict[idx]
        data = self.data_path_to_data_dict[data_path]
        ii = window_start_idx
        return data['place'][ii:ii+self.window_size,:], \
                data['cast'][ii:ii+self.window_size,:], \
                data['action'][ii:ii+self.window_size,:], \
                data['audio'][ii:ii+self.window_size,:], \
                data['scene_transition_boundary_ground_truth'][ii:ii+self.window_size-1].float()