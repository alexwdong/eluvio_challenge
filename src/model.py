import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

#---------------------------------------------------------------------------------------------
# Start: Define Generic Layers
#---------------------------------------------------------------------------------------------
class FcBlock(nn.Module):
    def __init__(self, inc , outc, bnorm_c = None, activation=nn.ReLU, batch_norm=True):
        super(FcBlock, self).__init__()
        if bnorm_c is None:
            bnorm_c = outc
        
        self.fc = nn.Linear(int(inc), int(outc),)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(bnorm_c) if batch_norm else None
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        if self.activation:
            x = self.activation(x)
        
        if self.bn:
            x = self.bn(x)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

    
    torch.nn.MultiheadAttention
    
class DifferenceSequence(nn.Module):
    def __init__(self):
        super(DifferenceSequence, self).__init__()
    def forward(self,x):
        # x is presumed to be of dimension batch_size, window_size (or sequence length),embedding_size
        x_1 = x[:,0:-1,:]
        x_2 = x[:,1:,:]
       
        
        out = x_1-x_2
        return out

    

class AttentionModule(nn.Module):
    '''
    MultiheadAttention
    '''
    def __init__(self,input_dimension):
        super(AttentionModule, self).__init__()
        
        self.q_linear = nn.Linear(input_dimension, 512)
        self.v_linear = nn.Linear(input_dimension, 512)
        self.k_linear = nn.Linear(input_dimension, 512)
        self.attention = nn.MultiheadAttention(embed_dim = 512,num_heads=32,dropout=.2,)
        
    def forward(self, embedding):
        q = self.q_linear(embedding)
        k = self.k_linear(embedding)
        v = self.v_linear(embedding)
        output,_= self.attention(q,k,v,need_weights=False)
       
        return output
#---------------------------------------------------------------------------------------------
# Start: Define BoundaryDetectorSimple
#---------------------------------------------------------------------------------------------

class BoundaryDetectorSimple(nn.Module):
    def __init__(self,window_size):
        super(BoundaryDetectorSimple, self).__init__()
        if window_size!=2:
            raise ValueError('window_size should be 2')
        self.window_size = window_size
        self.place_linear1 = FcBlock(2048,512,bnorm_c=window_size)
        self.place_linear2 = FcBlock(512,128,bnorm_c=window_size)
        self.place_linear3 = FcBlock(128,32,bnorm_c=window_size)
        
        self.cast_linear1 = FcBlock(512,128,bnorm_c=window_size)
        self.cast_linear2 = FcBlock(128,32,bnorm_c=window_size)
        
        self.action_linear1 = FcBlock(512,128,bnorm_c=window_size)
        self.action_linear2 = FcBlock(128,32,bnorm_c=window_size)
        
        self.audio_linear1 = FcBlock(512,128,bnorm_c=window_size)
        self.audio_linear2 = FcBlock(128,32,bnorm_c=window_size)
        
        
        self.linear_mix1 = FcBlock(128,64)
        self.linear_mix2 = FcBlock(64,1,activation=nn.Sigmoid,batch_norm=False)
        #self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, embedding):
        #print('in forward')
        place_embd, cast_embd, action_embd, audio_embd = embedding
        if place_embd.shape[1] != 2:
            raise ValueError ('Tensor needs to be of dimension B_size, 2, embedding size')    
        place_out = self.place_linear1(place_embd)
        place_out = self.place_linear2(place_out)
        place_out = self.place_linear3(place_out)

        cast_out = self.cast_linear1(cast_embd)
        cast_out = self.cast_linear2(cast_out)

        action_out = self.action_linear1(action_embd)
        action_out = self.action_linear2(action_out)

        audio_out = self.audio_linear1(audio_embd)
        audio_out = self.audio_linear2(audio_out)

        embds = torch.cat([place_out,cast_out,action_out,audio_out],dim=2)
        embds = embds[:,0,:] - embds[:,1,:]
   
        out = self.linear_mix1(embds)
        out = self.linear_mix2(out)
        return out
        
class BoundaryDetectorAttention(nn.Module):
    def __init__(self,window_size):
        super(BoundaryDetectorAttention, self).__init__()
        self.window_size = window_size

        self.attention_place = AttentionModule(input_dimension=2048)
        self.attention_cast = AttentionModule(input_dimension=512)
        self.attention_action = AttentionModule(input_dimension=512)
        self.attention_audio = AttentionModule(input_dimension=512)
        
        self.place_linear1 = FcBlock(512,128,bnorm_c=window_size)
        self.place_linear2 = FcBlock(128,32,bnorm_c=window_size)
        
        self.cast_linear1 = FcBlock(512,128,bnorm_c=window_size)
        self.cast_linear2 = FcBlock(128,32,bnorm_c=window_size)
        
        self.action_linear1 = FcBlock(512,128,bnorm_c=window_size)
        self.action_linear2 = FcBlock(128,32,bnorm_c=window_size)
        
        self.audio_linear1 = FcBlock(512,128,bnorm_c=window_size)
        self.audio_linear2 = FcBlock(128,32,bnorm_c=window_size)
        
        self.difference_sequence = DifferenceSequence()
        self.linear_mix1 = FcBlock(128,64,bnorm_c=window_size-1)
        self.linear_mix2 = FcBlock(64,1, bnorm_c=window_size-1, activation=nn.Sigmoid, batch_norm=False)
        
    def forward(self, embedding):
        #print('in forward')
        place, cast, action, audio = embedding
        
        place_embd = self.attention_place(place)
        cast_embd = self.attention_cast(cast)
        action_embd = self.attention_action(action)
        audio_embd = self.attention_audio(audio)
        
        place_out = self.place_linear1(place_embd)
        place_out = self.place_linear2(place_out)

        cast_out = self.cast_linear1(cast_embd)
        cast_out = self.cast_linear2(cast_out)

        action_out = self.action_linear1(action_embd)
        action_out = self.action_linear2(action_out)

        audio_out = self.audio_linear1(audio_embd)
        audio_out = self.audio_linear2(audio_out)
        
        embds = torch.cat([place_out,cast_out,action_out,audio_out],dim=2)
        #print('embds shape',embds.shape)
        out = self.difference_sequence(embds)
        #print('difference shape',out.shape)

        out = self.linear_mix1(out)
        out = self.linear_mix2(out)
        out = out.squeeze(dim=2)
        return out
    '''
    def training_step(self, batch, batch_idx):
        #print('in train_step')
        place, cast, action, audio, target = batch
        embedding = place, cast, action, audio
        out = self.forward(embedding)
        loss = F.binary_cross_entropy(out, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
  
    def validation_step(self, batch, batch_idx):
        #print('in val_step')
        place, cast, action, audio, target = batch
        embedding = place, cast, action, audio
        out = self.forward(embedding)
        loss = F.binary_cross_entropy(out, target)
        self.log('val_loss', loss)
        return loss
    '''