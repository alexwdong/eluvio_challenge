if __name__ == "__main__":
    import os
    import sys
    import pickle
    import numpy as np


    #First, Find the mean transition percentage
    train_dir = '/home/jolteon/eluvio_challenge/data/train/'
    transition_pct_list = []
    for file in os.listdir(train_dir):
        if file.endswith('.pkl'):
            with open(train_dir+file, 'rb') as f:
                data = pickle.load(f)

        transitions = data['scene_transition_boundary_ground_truth'].numpy()
        transition_pct = np.mean(transitions)
        transition_pct_list.append(transition_pct)
    mean_transition_pct = np.mean(transition_pct_list)
    print('Mean Transition Percent is:', mean_transition_pct)
    ### Here, we make a baseline for RANDOM predictions
    #Now, randomly generate predictions based on a bernoulli with probability = mean_transition_pct
    test_dir = '/home/jolteon/eluvio_challenge/data/test/'
    dir_to_save = '/home/jolteon/eluvio_challenge/baseline_random/'
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    for file in os.listdir(test_dir):
        if file.endswith('.pkl'):
            with open(test_dir+file, 'rb') as f:
                data = pickle.load(f)
            transition_size = data['scene_transition_boundary_ground_truth'].numpy().astype(float).size
            #print('transition_size',transition_size)
            prediction = np.squeeze(np.random.binomial(1,mean_transition_pct,size=(1,transition_size)))
            #print(prediction.shape)
            data_to_pkl={}
            data_to_pkl['scene_transition_boundary_ground_truth'] = data['scene_transition_boundary_ground_truth'].numpy().astype(float)
            data_to_pkl['scene_transition_boundary_prediction'] = prediction
            data_to_pkl['shot_end_frame'] = data['shot_end_frame']
            data_to_pkl['imdb_id'] = file[:-4]
            #print(data_to_pkl)
            with open(dir_to_save+file, 'wb') as f:
                pickle.dump(data_to_pkl,f)
                
    ### Here, we make a baseline using ELuvio's Preliminary predictions
    test_dir = '/home/jolteon/eluvio_challenge/data/test/'
    dir_to_save = '/home/jolteon/eluvio_challenge/baseline_preliminary/'
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    for file in os.listdir(test_dir):
        if file.endswith('.pkl'):
            with open(test_dir+file, 'rb') as f:
                data = pickle.load(f)
            data_to_pkl={}
            data_to_pkl['scene_transition_boundary_ground_truth'] = \
                data['scene_transition_boundary_ground_truth'].numpy().astype(float)
            data_to_pkl['scene_transition_boundary_prediction'] = \
                data['scene_transition_boundary_prediction'].numpy().astype(float)
            data_to_pkl['shot_end_frame'] = data['shot_end_frame']
            data_to_pkl['imdb_id'] = file[:-4]
            
            with open(dir_to_save+file, 'wb') as f:
                pickle.dump(data_to_pkl,f)
                
    
    
    



