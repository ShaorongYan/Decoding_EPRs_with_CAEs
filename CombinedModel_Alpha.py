import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Parameter
from torch.nn import MSELoss, L1Loss, SmoothL1Loss

import numpy as np
import math
import json

import pickle
with open('Index_dic.pickle', 'rb') as file:  # Inde dictionary that contains information about each trial
    Index_dic = pickle.load(file)
with open('Freq_dic.pickle', 'rb') as file:  # Load lexical frequency 
    Freq_dic = pickle.load(file)
with open('Surp_dic.pickle', 'rb') as file:   # Load lexical surprisal 
    Surp_dic = pickle.load(file)
with open('G_Word_Dis.pickle', 'rb') as file:  # Load semantic distance 
    SemDis_dic = pickle.load(file)


# ## Model Class
class Lang_Model(nn.Module):
    def __init__(self, 
                 depth = 3,    # For word embeddings that have multiple layers, e.g., ELMo. 
                 embedding_size = 1024,    # The size of word embeddings, if included in the model
                 shrink_hidden_sizes = [],
                 shrink_output_size = 128,
                 regression_hidden_sizes = [100],
                 output_size = 50,
                 output_channel = 10,
                 output_timestep = 20,
                 device = "cuda:2", 
                 dot_attention = False,
                 Frequency = False,
                 Surprisal = False,
                 SemanticDistance = False,
                 GloVe = False,
                 ELMo = False
                ):
        
        super().__init__()

        self.device = device

        ## Parameters of model structure 
        self.shrink_hidden_sizes = shrink_hidden_sizes
        self.shrink_output_size = shrink_output_size
        self.regression_hidden_sizes = regression_hidden_sizes
        self.output_size = output_size
        self.output_channel = output_channel
        self.output_timestep = output_timestep
        self.depth = depth
        self.embedding_size = embedding_size
        self.dot_attention = dot_attention
        self.Freq = Frequency
        self.Surp = Surprisal
        self.SemDis = SemanticDistance
        self.GloVe = GloVe
        self.ELMo = ELMo
        
        ## Initialize subject random intercept and fixed effects
        self.intercept = nn.Linear(1, self.output_channel * self.output_timestep).to(self.device)

        if self.Freq:
            self.freq_slope = nn.Linear(1, self.output_channel * self.output_timestep).to(self.device)

        if self.Surp:
            self.surp_slope = nn.Linear(1, self.output_channel * self.output_timestep).to(self.device)

        if self.SemDis:
            self.semdis_slope = nn.Linear(1, self.output_channel * self.output_timestep).to(self.device)

        if self.GloVe or self.ELMo:
            self.predictor_slope = nn.Linear(self.output_size, self.output_channel * self.output_timestep).to(self.device)

            ## Definding model structure
            self.wordN_shrink = self._build_MLP(self.embedding_size * self.depth,
                                      self.shrink_hidden_sizes,
                                      self.shrink_output_size)
            self.regression_MLP = self._build_MLP(self.shrink_output_size,
                                                  self.regression_hidden_sizes,
                                                  self.output_size)

    
    ### Build an MLP module
    def _build_MLP(self, input_size, hidden_sizes, output_size):
        
        linear_maps = nn.ModuleList()
        
        last_size = input_size

        if hidden_sizes:
            for h in hidden_sizes:
                linmap = nn.Linear(last_size, h)
                linmap = linmap.to(self.device)
                linear_maps.append(linmap)
                last_size = h

        linmap = nn.Linear(last_size, output_size)
        linmap = linmap.to(self.device)
        linear_maps.append(linmap)
        return linear_maps

    ### Run an MLP module
    def _run_MLP(self, output, linear_maps):
        ## Define activation function as Tanh
        act = nn.Tanh()
        for i, linear_map in enumerate(linear_maps):
            output = linear_map(output).to(self.device)
            output = act(output)
        return output

    def forward(self, index):

        # Create intercept matrix
        intercept_tensor = torch.from_numpy(np.array([1 for ind in index]).reshape(len(index),1)).float().to(self.device)
        intercept_tensor = self.intercept(intercept_tensor)
        Output = intercept_tensor

        if self.Freq:
            freq_tensor = torch.from_numpy(np.array([Freq_dic[(Index_dic[i][1], Index_dic[i][2])] for i in index]).reshape(len(index),1)).float().to(self.device)
            freq_tensor = self.freq_slope(freq_tensor)
            Output += freq_tensor

        if self.Surp:
            surp_tensor = torch.from_numpy(np.array([Surp_dic[(Index_dic[i][1], Index_dic[i][2])] for i in index]).reshape(len(index),1)).float().to(self.device)
            surp_tensor = self.surp_slope(surp_tensor)
            Output += surp_tensor

        if self.SemDis:
            semdis_tensor = torch.from_numpy(np.array([SemDis_dic[(Index_dic[i][1], Index_dic[i][2])] for i in index]).reshape(len(index),1)).float().to(self.device)
            semdis_tensor = self.semdis_slope(semdis_tensor)
            Output += semdis_tensor

        if self.GloVe:
            ## Creating tensor that store input
            word_tensors = torch.zeros([len(index), self.embedding_size * self.depth]).to(self.device)
            
            for i in range(len(index)):               
            
                # Getting the word tensor
                word_tensors[i,:] = GloVe_dic[(Index_dic[index[i]][1], Index_dic[index[i]][2])]
                
            # Passing through an initial network to shrink the dimensions
            wordN_shrinked = self._run_MLP(word_tensors, self.wordN_shrink)

            # Further reducing for regression
            Output += self.predictor_slope(self._run_MLP(wordN_shrinked, self.regression_MLP))

        if self.ELMo:
            ## Creating tensor that store input
            word_tensors = torch.zeros([len(index), self.embedding_size * self.depth]).to(self.device)
                                     
            for i in range(len(index)):               
            
                # Getting the word tensor
                word_tensors[i,:] = ELMo_dic[(Index_dic[index[i]][1], Index_dic[index[i]][2])]
                
            # Passing through an initial network to shrink the dimensions
            wordN_shrinked = self._run_MLP(word_tensors, self.wordN_shrink)

            # Further reducing for regression
            Output += self.predictor_slope(self._run_MLP(wordN_shrinked, self.regression_MLP))

        # Add acativation function 
        Output_act = nn.Tanh()
        Output = Output_act(Output)
                                     
        Output = Output.reshape(len(index),
                                self.output_channel,
                                self.output_timestep)
        return Output

## The decoder part of an autoencoder
class Decoder(torch.nn.Module):
    def __init__(self, n_observed_channels = 32, n_hidden_channels = 32, shrink_list = [32, 32, 16, 8, 4], device = 'cuda:2'):
        super().__init__()
        
        self.component_map_decode = torch.nn.Linear(n_hidden_channels, n_observed_channels).to(device)
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(shrink_list[3], shrink_list[2], 4, stride=2, padding = 1),
            torch.nn.ReLU(True),            
            torch.nn.ConvTranspose1d(shrink_list[2], shrink_list[1], 6, stride=2, padding = 3),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose1d(shrink_list[1], shrink_list[0], 8, stride=3, padding = 3),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose1d(shrink_list[0], 32, 8, stride=2, padding = 4),
            torch.nn.Tanh()
        ).to(device)

    def forward(self, x):
                                     
        decoded = self.decoder(x)
        decoded = self.component_map_decode(decoded.transpose(2,1))
        return decoded.transpose(2,1)


## Combine the language predictor and the decoder
class Combined(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Lang_M = Lang_M

        self.decoder = eeg_decoder
        
    def forward(self, index):
        
        encoded = self.Lang_M(index)
        
        decoded = self.decoder(encoded)
        
        return decoded

## Training the combined model 
class Combined_Trainer(object):

    def __init__(self, 
             device= "cuda:2", 
             **kwargs):
    
        self.device = device
        self._init_kwargs = kwargs 


    def _initialize_trainer(self):
        self._combined = Combined(**self._init_kwargs)
        
        self._combined = self._combined.to(self.device)
         

    def fit(self, Train_seq, Dev_seq,
            Train_tensor, Dev_tensor, 
            epochs = 100,
            batch_size = 128,
            learning_rate = 0.001,
            wd = 0.001,
            model_name = 'best_model',
            output_file_name = 'output.txt'):

        self._initialize_trainer()

        criterion = torch.nn.MSELoss()

        ## Calculate variability at each fold 
        dev_var =  criterion(Dev_tensor + Dev_tensor.mean() - Dev_tensor, Dev_tensor).item()

        f = open(output_file_name, 'w')
        print("########## .   Model Parameters   ##############\n")
        f.write("########## .   Model Parameters   ##############\n")
        for name, param in self._combined.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                f.write(' '.join([name, str(param.shape), '\n']))
        print("\n")
        print("##############################################\n")
        f.write("#############################################\n")
        f.close()


        parameters = [p for p in self._combined.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters,
                                     lr=learning_rate,
                                     weight_decay= wd)

        for epoch in range(epochs):  # loop over the dataset multiple times
            self._combined.train()

            print("Running Epoch: {}".format(epoch+1))

            bidx_i = 0
            bidx_j = batch_size
        
            ## Keeping track of loss
            train_loss = 0.0
            best_dev_loss = float('inf')
    
            batch_num = 0
        
            while bidx_j < len(Train_seq):
                if bidx_i % 12288 == 0:
                    print('now at ' + str(bidx_i))

                inputs = Train_seq[bidx_i:bidx_j]

                #Zero grad
                optimizer.zero_grad()

                # Forward pass 
                reconstructed = self._combined(inputs)

                #Calculate Loss
                loss = criterion(reconstructed, Train_tensor[bidx_i:bidx_j])

                # Back propagate
                loss.backward()
                optimizer.step()
                
                # Go to next batch
                bidx_i = bidx_j
                bidx_j = bidx_i + batch_size

                # print statistics
                train_loss += loss.item()

                batch_num += 1

            if bidx_j >= len(Train_seq):
                inputs = Train_seq[bidx_i: ]

                #Zero grad
                optimizer.zero_grad()

                # Forward pass 
                reconstructed = self._combined(inputs)

                #Calculate Loss
                loss = criterion(reconstructed, Train_tensor[bidx_i:bidx_j])

                # Back propagate
                loss.backward()
                optimizer.step()
                
                # Go to next batch
                bidx_i = bidx_j
                bidx_j = bidx_i + batch_size

                # print statistics
                train_loss += loss.item()

                batch_num += 1

            # print statistics
            print('Epoch %d: Loss on training set: %.8f' %(epoch+1, train_loss/batch_num))
            
                # Check loss on dev set
            with torch.no_grad():  
                predicted = self._combined(Dev_seq) 
                loss = criterion(predicted, Dev_tensor)
                dev_loss = loss.item()

            print('Loss on development set: %.8f' %(dev_loss))
        
                    
            # Save the model if the validation loss is the best we've seen so far.
            if dev_loss < best_dev_loss:
                with open(model_name, 'wb') as f:
                    torch.save(self._combined.state_dict(), f)
                    best_dev_loss = dev_loss

                    f = open(output_file_name, 'a')   
                    f.write('Epoch %d: Loss on training set: %.8f' %(epoch+1, train_loss/batch_num))
                    f.write("\n")
                    f.write('Loss on development set: %.8f' %(dev_loss))
                    f.write("\n")
                    f.close()

## Intercept 
# Hyper parameters
lr_= 1e-3
EP = 100
wd_lst = [1e-5 ,1e-3,1e-1]
bs = 128

# Training
model_path = 'best_models/'
output_file_path = 'outputs/'
weight_head = '../Autoencoder/best_models/EEG_Autoencoder_alpha_w_lr0.001_wd1e-05fd' ## Weight of the trained autoencoder

with open('../Autoencoder/Train_seq_all.pickle', 'rb') as file:
   Train_seq_all = pickle.load(file)

for wd_ in wd_lst:
    for fd in range(5):

        # Load the decoder of a trained autoencoder                                     
        eeg_decoder = Decoder(n_observed_channels = 32,
                              n_hidden_channels = 32,
                              shrink_list = [32, 32, 16, 8, 4])

        # Load weights 
        with open(weight_head + str(fd) + '.pth', 'rb') as f:
            pretrained_dict = torch.load(f)

        model_dict = eeg_decoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        eeg_decoder.load_state_dict(pretrained_dict)
        del model_dict, pretrained_dict
        # Freeze weights
        for p in eeg_decoder.parameters():
            p.requires_grad = False

        Train_seq = Train_seq_all[fd][0]
        Dev_seq = Train_seq_all[fd][1]

        with open('../Autoencoder/EEG_tensors_bc.pickle', 'rb') as file:
            EEG_tensors = pickle.load(file)
        Train_tensor = EEG_tensors[Train_seq]
        Dev_tensor = EEG_tensors[Dev_seq]
        del EEG_tensors


        # Intercept + Lexical Frequency + Lecical Surprisal + Semantic Distance + ELMo
        head = 'Combined_alpha_IFSDE'
        model_name = model_path + head + '_lr' + str(lr_) + '_wd' + str(wd_) + 'fd' + str(fd) + '.pth'
        output_file_name = output_file_path + head + '_lr' + str(lr_)  + '_wd' + str(wd_) + 'fd' + str(fd)  + '.txt'

        print(head)

        # Specify which predictors to include
        par_list = [1,1,1,0,1]
        # Model specification
        # torch.manual_seed(123)
        Lang_M = Lang_Model(depth = 3,
                         embedding_size = 1024, 
                         shrink_hidden_sizes = [], # Shrink the size of the context and word vectors before entering regression
                         shrink_output_size = 200,
                         regression_hidden_sizes = [], 
                         output_size = 200,
                         output_channel = 8,
                         output_timestep = 9,
                         device = "cuda:2", 
                         dot_attention = False,
                         Frequency = par_list[0],
                         Surprisal = par_list[1],
                         SemanticDistance = par_list[2],
                         GloVe = par_list[3],
                         ELMo = par_list[4])


        Trainer = Combined_Trainer()
                                         
        Trainer.fit(Train_seq, Dev_seq,
                    Train_tensor, Dev_tensor, 
                    epochs = EP,
                    batch_size = bs,
                    learning_rate = lr_,
                    wd = wd_,
                    model_name = model_name,
                    output_file_name = output_file_name)




