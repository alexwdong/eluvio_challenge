import os
import pickle
import numpy as np
import pandas as pd
import collections


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

def make_difference_features(data):
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
    y_movie = scene_boundary_truth
    return X_movie,y_movie
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

#----------------Embedding Above-------------
#----------------Generate Predictions below-----
def generate_predictions_dir_LR(model,input_dir,output_dir):
    #best_model should be LogisticRegression
    for file in os.listdir(input_dir):
        if file.endswith('.pkl'):
            with open(input_dir+file, 'rb') as f:
                data = pickle.load(f)
            df = make_dot_product_features(data)
            X = df[['place_dp','cast_dp','action_dp','audio_dp']]
            predictions = model.predict_proba(X)[:,1]
            data_to_pkl={}
            data_to_pkl['scene_transition_boundary_ground_truth'] = \
                data['scene_transition_boundary_ground_truth'].numpy()
            data_to_pkl['scene_transition_boundary_prediction'] = \
                predictions
            data_to_pkl['shot_end_frame'] = data['shot_end_frame']
            data_to_pkl['imdb_id'] = file[:-4]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_dir+file, 'wb') as f:
                pickle.dump(data_to_pkl,f)
def generate_predictions_dir_RF(model,input_dir,output_dir):
    #best_model should be RandomForestClassifier
    for file in os.listdir(input_dir):
        if file.endswith('.pkl'):
            with open(input_dir+file, 'rb') as f:
                data = pickle.load(f)
            X,_= make_difference_features(data)
            #print(X.shape)
            predictions = model.predict_proba(X)[:,1]
            #print(predictions)
            
            ground_truth =  data['scene_transition_boundary_ground_truth'].numpy()
            #print(ground_truth)
            data_to_pkl={}
            data_to_pkl['scene_transition_boundary_ground_truth'] = ground_truth
               
            data_to_pkl['scene_transition_boundary_prediction'] = \
                predictions
            data_to_pkl['shot_end_frame'] = data['shot_end_frame']
            data_to_pkl['imdb_id'] = file[:-4]
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_dir+file, 'wb') as f:
                pickle.dump(data_to_pkl,f)

       
       


def generate_predictions(model,data_from_pkl,window_size,device):
    '''
    Model is assumed to be a pytorch model
    '''
    model.eval()
    model.to(device)
    data = data_from_pkl
    num_shots = data['shot_end_frame'].shape[0]
    num_windows = num_shots - window_size+1

    # This is the data structure we use to store our predictions
    # Not all shots will have the same amount of predictions, due to the sliding window
    # In particular, the first few and last few shot boundaries have less predictions, but everything 
    # "in the middle" should have window_size amount of predictions.
    #
    # We will store all the predictions for each shot boundary, and then average each
    # shot boundaries list of predictions to get our final prediction for that shot boundary.
    index_to_predictions_list_dict = collections.defaultdict(list)

    for ii in range(num_windows):
        # Grab the window, unsqueeze since we need to add a batch_size dimension
        place = data['place'][ii:ii+window_size,:].unsqueeze(0)
        cast = data['cast'][ii:ii+window_size,:].unsqueeze(0)
        action = data['action'][ii:ii+window_size,:].unsqueeze(0)
        audio = data['audio'][ii:ii+window_size,:].unsqueeze(0)
        target = data['scene_transition_boundary_ground_truth'][ii:ii+window_size-1].float().unsqueeze(0)
        # Send to device
        place = place.to(device)
        cast = cast.to(device)
        action = action.to(device)
        audio = audio.to(device)
        target = target.to(device)
        embedding = place,cast,action,audio
        #Forward Pass
        out = model(embedding) #Should be dimension 1 x window_size
        #Compute loss and accuracy
        #loss = criterion(out, target)
        
        #some index trickery here, kk is index in the window (from 0 to window_size-1), 
        # jj is the index in the entire movie (e.g from (500 to 500+window_size-1))
        for kk,jj in enumerate(range(ii,ii+window_size-1)):
            index_to_predictions_list_dict[jj].append(out[0,kk].cpu().item())
    predictions = np.ones(shape = num_shots-1)*-99999
    for index,predictions_list in index_to_predictions_list_dict.items():
        predictions[index] = np.mean(predictions_list)
    return predictions

def generate_predictions_dir_NN(model,window_size,input_dir,output_dir,device):
     for file in os.listdir(input_dir):
        if file.endswith('.pkl'):
            with open(input_dir+file, 'rb') as f:
                data = pickle.load(f)
            predictions = generate_predictions(model,data,window_size,device)
            data_to_pkl={}
            data_to_pkl['scene_transition_boundary_ground_truth'] = \
                data['scene_transition_boundary_ground_truth'].numpy().astype(float)
            data_to_pkl['scene_transition_boundary_prediction'] = predictions
            data_to_pkl['shot_end_frame'] = data['shot_end_frame']
            data_to_pkl['imdb_id'] = file[:-4]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_dir+file, 'wb') as f:
                pickle.dump(data_to_pkl,f)
            

    


    

        