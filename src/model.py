import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

#---------------------------------------------------------------------------------------------
# Start: Define Generic Layers
#---------------------------------------------------------------------------------------------
class FcBlock(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FcBlock, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
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
    
class DotProductSequence(nn.Module):
    def __init__(self):
        super(DotProductSequence, self).__init__()
    def forward(self,x):
        # x is presumed to be of dimension batch_size, window_size,embedding_size
        x_1 = x[:,0:-1,:]
        x_2 = x[:,1:,:]
        #print('x1 shape',x_1.shape)
        #print('x2 shape',x_2.shape)
        #Einsum..... trying to do a batched dot product over the last axis
        
        out = torch.einsum('bwi,bwi->bw', x_1, x_2)
        return out

    

class AttentionModule(nn.Module):

    def __init__(self,input_dimension):
        super(AttentionModule, self).__init__()
        
        self.q_linear = nn.Linear(input_dimension, 256)
        self.v_linear = nn.Linear(input_dimension, 256)
        self.k_linear = nn.Linear(input_dimension, 256)
        self.attention = nn.MultiheadAttention(embed_dim = 256,num_heads=16,dropout=.1,)
        self.sequential_dot_prod = DotProductSequence()
        
    def forward(self, embedding):
        q = self.q_linear(embedding)
        k = self.k_linear(embedding)
        v = self.v_linear(embedding)
        #print('qkv shape',q.shape,k.shape,v.shape)
        attn_output,_= self.attention(q,k,v,need_weights=False)
        output = self.sequential_dot_prod(attn_output)
        return output
#---------------------------------------------------------------------------------------------
# Start: Define BoundaryDetector
#---------------------------------------------------------------------------------------------


        
class BoundaryDetector(pl.LightningModule):
    def __init__(self,window_size):
        super(BoundaryDetector, self).__init__()
        
        self.attention_place = AttentionModule(input_dimension=2048)
        self.attention_cast = AttentionModule(input_dimension=512)
        self.attention_action = AttentionModule(input_dimension=512)
        self.attention_audio = AttentionModule(input_dimension=512)
        self.linear = nn.Linear(36,window_size-1)
        self.activation = nn.Sigmoid()
        
    def forward(self, embedding):
        #print('in forward')
        place, cast, action, audio = embedding
        
        place_embd = self.attention_place(place)
        cast_embd = self.attention_cast(cast)
        action_embd = self.attention_action(action)
        audio_embd = self.attention_audio(audio)
        
        #print('place embd shape',place_embd.shape)
        #print('cast embd shape',cast_embd.shape)
        #print('aciton embd shape',action_embd.shape)
        #print('audio embd shape',audio_embd.shape)


        
        embds = torch.cat([place_embd,cast_embd,action_embd,audio_embd],dim=1)
        #print('embeddings shape',embds.shape)
        out = self.linear(embds)
        out = self.activation(out)
        return out

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
