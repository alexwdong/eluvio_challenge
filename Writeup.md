# Writeup
# Intro/Summary
In this challenge, I decided to focus on maximizing performance using only the raw features provided. I did not use the preliminary predictions in my models, though I did use them to generate a baseline. I approached the problem as follows:

Generate three baselines to compare against:  
    1) Randomly predict scene transitions based on a Bernoulli Distribution, with $P(transition) = mean(observed\_transitions)$. In our training set, we observed the mean to be about ~.076.  
    2) Logistic Regression on dot product between features.  
    3) Preliminary predictions provided by Eluvio.  
    
We then created three models to do the predictions.  
    1) Random Forest on difference between features.  
    2) Neural Net: Fully connected layers structure on difference between features.  
    3) Neural Net: Attention module followed by fully connected layers.  

For our evaluation metric, we look at both the the mAP and Miou. Note that some models perform well on one but not both of the metrics.

We found that our Neural Net with only fully connected layers on the difference between features performed the best, with a mAP of ~0.37 and mIOU of ~0.30. However, this was outperformed by the predictions that Eluvio provided, which have a mAP of ~0.49 mIOU of ~0.47.

For each model, we used the exact same train/test splits. 

Final neural network models and result directories can be found in model_and_results.zip
# Baselines

### Baseline 1: Random
This is bare minimum baseline - we took number of transitions, and divided it by the total amount of potential transitions to get a transition probability, then sample a Bernoulli with that probability at each potential transition in the test set. The scores are obviously pretty bad. Perhaps surprisingly, the Miou is not terrible, but this is still meant to be the lower-bound baseline.
Scores: {  
    "AP": 0.0903779250550157,  
    "mAP": 0.09798375588604646,  
    "Miou": 0.30419257642839564,  
    "Precision": 0.0974340223226487,  
    "Recall": 0.07676671243899885,  
    "F1": 0.08329692954096599  
}
### Baseline 2: Logistic Regression on Dot product of features
For this baseline, we want to use a really simple, easy model, and see how well it performs.
Now, in order to predict a scene transition between two shots, we use only the four features ('place','cast', 'action', and 'audio') of those two shots.  

So, for each scene transition, we took the dot product between the first and second shot for each of the four features ('place','cast', 'action', and 'audio'). This leaves us with only four features (each with 1 dimension, instead of 2048 or 512) for each shot boundary. We then fit a logistic regression model to this (searching a few hyperparameters for $C$), and generated predictions for our test set, which gives us the following baseline result:

(There was an error/warning when generating precision, since our logistic regression model only predicts probabilities below 0.5)

Scores: {  
    "AP": 0.18439897955928014,  
    "mAP": 0.22990494989628843,  
    "Miou": 0.03214543822808406,  
    "Precision": 0.0,  
    "Recall": 0.0,  
    "F1": NaN  
}
### Baseline 3: Preliminary Eluvio Predictions
These are simply the metrics for the preliminary predictions provided by Eluvio.  

Scores: {  
    "AP": 0.4799354967433886,  
    "mAP": 0.49328420987394367,  
    "Miou": 0.4797450602748557,  
    "Precision": 0.3380879775551314,  
    "Recall": 0.6925031762816138,  
    "F1": 0.44427621620186347  
}
# Models
### Model 1: Random forest on difference of features 
For this model, we wanted to increase the complexity relative to the logistic regression baseline. The setup is similar, except instead of taking the dot product between the features, we subtract one from the other. This leaves us with 2048+512+512+512 = 3584 total features. We then fit a random forest model to this (searching hyperparamters min_samples_leaf and max_depth), and generated predictions for our test set, which gives us the following baseline result:

(Similar to the logistic regression, our random forest model only predicts probabilities below 0.5)

Scores: {  
    "AP": 0.4481750804878683,  
    "mAP": 0.4728530463875269,  
    "Miou": 0.03214543822808406,  
    "Precision": 0.0,  
    "Recall": 0.0,  
    "F1": NaN  
}

The mAP is surprisingly high, but the Miou is really bad. This is due to the fact that the random forest is never confident in its predictions.
## Notes for Neural Nets
Note 1 : The neural network architectures can be found in src/model.py.  

    Model 2: FC Neural Net is BoundaryDetectorSimple  
    Model 3: Attention->FC Net is BoundaryDetectorAttention  
    Fully Connected Block is FcBlock
    
Note 2: When we use the term Fully Connected Block, we mean the following sequence of layers:
    
    1) Linear
    2) Dropout
    3) ReLU
    4) BatchNormalization
Note 3: Models were trained on a Nvidia RTX 3060 Ti GPU. FC Neural net took approx 1 min/epoch, Attention Neural net took approx 2.5 min/epoch.
### Model 2: FC Neural Net
This model is meant to be similar to the random forest and logistic regression models as before, where we only use the features from the two shots before and after a potential scene transition.

In this model we build a neural network with the following architecture.

    0) Take the place, cast, action, audio embeddings for both scenes before and after the potential scene transition. Both sets of embeddings go through steps 1a,1b, and 2, using the same network.
    1a) Place embeddings go through 3 Fully Connected Blocks going from 2048 to 32 dimensions
    1b) Cast, action, audio go through 2 Fully Connected Blocks going from 512 to 32 dimensions
    2) Place,cast,action,audio embeddings are then concatenated into a 32*4 = 128 dimension embedding
    3) The difference of embeddings is taken, and so we are left with one 128 dimension embedding.
    4) That embedding goes through 2 Fully Connected Blocks, going from 128 to 1 dimension which is the final prediction. The last fully connected block has no batch normalization, and has a sigmoid activation function instead of a ReLU.
    
Our model has ~1.3M parameters, and we train for twenty epochs, and then take the best model.
Our results are the following:


Scores: {
    "AP": 0.3563740077987584,  
    "mAP": 0.37175532395490957,  
    "Miou": 0.3098844562627911,  
    "Precision": 0.7226293151293152,  
    "Recall": 0.09877458091669455,  
    "F1": 0.16799157518164576  
}

Overall, we think this is the best model, as it has a relatively good mAP score, while also having a good Miou score. It should also be noted that this model only took the features from only the two shots before and after the potential scene transition.
# Model 3: Attention -> FC layers
Here we try to really increase the complexity of the network by incorporating an attention window.
We use a window size of 12 which means that we take as input the features of 12 shots. We attempt to predict the 11 shot boundaries of those 12 shots.

In this model we build a neural network with the following architecture.

    0) Take the place, cast, action, audio embeddings for both scenes all sceneds in the window. Each shot's embeddings go through steps 1a,1b, and 2, using the same network.
    1a) Each features has a seperate attention module, which incorporates three linear networks to generate the query, key, and value embeddings, which are each of dimension 512.
    1b) MultiheadAttention is applied, with 32 heads.
    2a) We now have 4 dim=512 embeddings for the place,cast,action,audio for each of the 12 shots in the window.
    2b) Each of place,cast,action,audio go through two more fully connected blocks, going from 512 to 32 dimensions.
    3) Concatenate the place,cast,action,audio embeddings, to get a 128 dimension embedding for each shot in the window
    4) Take the difference between sequential embeddings, so we go from 12 shots with dim=128 embeddings, to 11 shot boundaries with dim = 128 embeddings.
    5) Each of the 11 embeddings goes through2 Fully Connected Blocks, going from 128 to 1 dimension which is the final prediction. The last fully connected block has no batch normalization, and has a sigmoid activation function instead of a ReLU.
    
Our model has ~10M parameters, and we train for twenty epochs, and then take the best model.
Our results are the following:  
Scores: {  
    "AP": 0.12271408528698208,  
    "mAP": 0.13798523838151633,  
    "Miou": 0.1693378531162345,  
    "Precision": 0.2514590672485409,  
    "Recall": 0.009354341066358218,  
    "F1": NaN  
}

I was a little bit surprised by the incredibly poor results, as I think this model architecture is closer to whats needed to effectively model the scene transition boundaries (long windows looking into the past and future). However, I think that the model here suffers severely from overfitting - this is likely due in part to a really small training set. I would be interested to see what happens if we add more training data. Note that the model's training accuracy was pretty high (92.8%, even with dropout applied). 