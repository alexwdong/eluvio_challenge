import os
import pickle
import numpy as np
import pandas as pd

def make_embedding_dot_prod_feat(embeddings,num_boundaries):
    '''
    Take the a sequence of length n+1 of embeddings, and make the n
    embeddings which are a dot product of sequential embeddings.
    returns a list, which will be converted into a df later
    '''
    embd_diff_list = []
    for ii in range(num_boundaries):
        # Place
        current_embd = embeddings[ii]
        next_embd = embeddings[ii+1]
        embd_diff = np.dot(current_embd,next_embd)
        embd_diff_list.append(embd_diff)
    return embd_diff_list
        
def make_dot_product_features(data):
    # Define Embeddings
    scene_boundary_truth = data['scene_transition_boundary_ground_truth'].numpy().astype(int)
    place_embedding = data['place'].numpy()
    cast_embedding = data['cast'].numpy()
    action_embedding = data['action'].numpy()
    audio_embedding = data['audio'].numpy()

    #Create embedding dot products
    num_boundaries = scene_boundary_truth.shape[0]
    place_embd_dp_list = make_embedding_dot_prod_feat(place_embedding, num_boundaries)
    cast_embd_dp_list = make_embedding_dot_prod_feat(cast_embedding, num_boundaries)
    action_embd_dp_list = make_embedding_dot_prod_feat(action_embedding, num_boundaries)
    audio_embd_dp_list = make_embedding_dot_prod_feat(audio_embedding, num_boundaries)
    
    df = pd.DataFrame(
        {'place_dp' : place_embd_dp_list,
         'cast_dp' : cast_embd_dp_list,
         'action_dp' : action_embd_dp_list,
         'audio_dp' : audio_embd_dp_list,
         'boundary_truth' : scene_boundary_truth,
        })
    return df
def make_all_dot_product_features_df(data_dir):
    '''
    makes the dot product features df of data_dir
    '''
    
    df_list= []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            with open(data_dir+file, 'rb') as f:
                data = pickle.load(f)
            df = make_dot_product_features(data)
            df_list.append(df)
    df_all = pd.concat(df_list)
    return df_all

#------------------Dot Product Above ----------
#------------------Embedding Below--------------


def make_embedding_difference_feat(embeddings,num_boundaries):
    embd_diff_list = []
    for ii in range(num_boundaries):
        # Place
        current_embd = embeddings[ii]
        next_embd = embeddings[ii+1]
        embd_diff = current_embd-next_embd
        embd_diff_list.append(embd_diff)
    return embd_diff_list


def make_all_embedding_difference_features_df(data_dir):
    '''
    makes the embedding features df of data_dir
    '''
    
    df_list = []
    list_of_X_movie = []
    list_of_y_movie = []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            with open(data_dir+file, 'rb') as f:
                data = pickle.load(f)
            scene_boundary_truth = data['scene_transition_boundary_ground_truth'].numpy().astype(int)
            place_embedding = data['place'].numpy()
            cast_embedding = data['cast'].numpy()
            action_embedding = data['action'].numpy()
            audio_embedding = data['audio'].numpy()
            
            #Make differences of embeddings
            num_boundaries = scene_boundary_truth.shape[0]
            place_embd_dp_list = make_embedding_difference_feat(place_embedding, num_boundaries)
            cast_embd_dp_list = make_embedding_difference_feat(cast_embedding, num_boundaries)
            action_embd_dp_list = make_embedding_difference_feat(action_embedding, num_boundaries)
            audio_embd_dp_list = make_embedding_difference_feat(audio_embedding, num_boundaries)
            
            features_list = []
            for ii in range(num_boundaries):
                #Concat each embedding difference into one difference of all of the embeddings
                place_embedding = place_embd_dp_list[ii]
                cast_embedding = cast_embd_dp_list[ii]
                action_embedding = action_embd_dp_list[ii]
                audio_embedding = audio_embd_dp_list[ii]
                super_embedding = np.concatenate([place_embedding ,cast_embedding, action_embedding, audio_embedding])
                features_list.append(super_embedding)
            #Stack to get our X for this movie, and then append to list
            X_movie = np.vstack(features_list)
            list_of_X_movie.append(X_movie)
            # No need to stack, just append to list
            list_of_y_movie.append(scene_boundary_truth)
            
    X = np.vstack(list_of_X_movie)
    y = np.concatenate(list_of_y_movie)
            
    return X,y