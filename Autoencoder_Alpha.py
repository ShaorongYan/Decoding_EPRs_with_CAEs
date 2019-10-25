import numpy as np
import torch
import pickle
import torch.optim as optim

### Building the autoencoder
class EEGAutoencoder(torch.nn.Module):
    def __init__(self, n_observed_channels = 32, n_hidden_channels = 32, shrink_list = [32, 32, 16, 8, 4], device = 'cuda:2'):
        super().__init__()
        ### Fully connected layer to recombine all channels 
        self.component_map_encode = torch.nn.Linear(n_observed_channels, n_hidden_channels).to(device)
        self.component_map_decode = torch.nn.Linear(n_hidden_channels, n_observed_channels).to(device)
        ### Encoder 
        self.encoder = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Conv1d(32, shrink_list[0], 8, stride=2, padding=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4, stride=1, padding=2),
            torch.nn.Conv1d(shrink_list[0], shrink_list[1], 8, stride=3, padding=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4, stride=1, padding=2),
            torch.nn.Conv1d(shrink_list[1], shrink_list[2], 6, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4, stride=1, padding=2),
            torch.nn.Conv1d(shrink_list[2], shrink_list[3], 4, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4, stride=1, padding=2) 
        ).to(device)
        ### Decoder 
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
        reduced = self.component_map_encode(x.transpose(2,1))
        encoded = self.encoder(reduced.transpose(2,1))       
        decoded = self.decoder(encoded)
        decoded = self.component_map_decode(decoded.transpose(2,1))       
        return reduced, encoded, decoded.transpose(2,1)

### Training the autoencoder
class Autoencoder_Trainer(object):

    def __init__(self, 
             device= "cuda:2", 
             **kwargs):
    
        self.device = device
        self._init_kwargs = kwargs 


    def _initialize_autoencoder(self):
        self._autoencoder = EEGAutoencoder(device=self.device,
                                             **self._init_kwargs)
        
        self._autoencoder = self._autoencoder.to(self.device)
         

    def fit(self, X_train, X_dev,
            epochs = 100,
            batch_size = 128,
            learning_rate = 0.001,
            wd = 0.001,
            model_name = 'best_model',
            output_file_name = 'output.txt'):

        self._initialize_autoencoder()

        criterion = torch.nn.MSELoss()  

        ## Calculate variability at each fold 
        dev_var =  criterion(X_dev + X_dev.mean() - X_dev, X_dev).item()

        f = open(output_file_name, 'w')
        print("########## .   Model Parameters   ##############\n")
        f.write("########## .   Model Parameters   ##############\n")
        for name, param in self._autoencoder.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                f.write(' '.join([name, str(param.shape), '\n']))
        print("\n")
        print("##############################################\n")
        f.write("#############################################\n")
        f.close()


        parameters = [p for p in self._autoencoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters,
                                     lr=learning_rate,
                                     weight_decay= wd)

        for epoch in range(epochs):  # loop over the dataset for the number of epochs specified 
            self._autoencoder.train()

            print("Running Epoch: {}".format(epoch+1))

            bidx_i = 0
            bidx_j = batch_size
        
            ## Keeping track of loss
            train_loss = 0.0
            best_dev_loss = - float('inf')
    
            batch_num = 0
        
            while bidx_j < len(X_train):
                if bidx_i % 12288 == 0:
                    print('now at ' + str(bidx_i))

                inputs = X_train[bidx_i:bidx_j]

                #Zero grad
                optimizer.zero_grad()

                # Forward pass
                reduced, encoded, reconstructed = self._autoencoder(inputs)

                #Calculate Loss
                loss = criterion(reconstructed, inputs)

                # Back propagate
                loss.backward()
                optimizer.step()
                
                # Go to next batch
                bidx_i = bidx_j
                bidx_j = bidx_i + batch_size

                # Keep track of the loss for later print-out
                train_loss += loss.item()/np.prod(inputs.shape[0])

                batch_num += 1

            if bidx_j >= len(X_train): 
                inputs = X_train[bidx_i: ]

                #Zero grad
                optimizer.zero_grad()

                # Forward pass
                reduced, encoded, reconstructed = self._autoencoder(inputs)

                #Calculate Loss
                loss = criterion(reconstructed, inputs)

                # Back propagate
                loss.backward()
                optimizer.step()
                
                # Go to next batch
                bidx_i = bidx_j
                bidx_j = bidx_i + batch_size

                # Keep track of the loss for later print-out
                train_loss += loss.item()/np.prod(inputs.shape[0])

                batch_num += 1

            # Print metric
            print('Epoch %d: Loss on training set: %.8f' %(epoch+1, train_loss/batch_num))
            
            # Check loss on dev set
            with torch.no_grad():  
                a, b, predicted = self._autoencoder(X_dev) 
                loss = criterion(predicted, X_dev)
                dev_loss = 1 - loss.item()/dev_var

            print('Loss on development set: %.8f' %(dev_loss))
        
                    
            # Save the model if the validation loss is the best we've seen so far.
            if dev_loss > best_dev_loss:
                with open(model_name, 'wb') as f:
                    torch.save(self._autoencoder.state_dict(), f)
                    best_dev_loss = dev_loss

                    f = open(output_file_name, 'a')   
                    f.write('Epoch %d: Loss on training set: %.8f' %(epoch+1, train_loss/batch_num))
                    f.write("\n")
                    f.write('Loss on development set: %.8f' %(dev_loss))
                    f.write("\n")
                    f.close()
        
        
# Load training sequences
# This is generated with the kfold function in scikit-learn
with open('../Train_seq_new.pickle', 'rb') as file:
    Train_seq = pickle.load(file)

# Hyper parameters
lr_ = 1e-3 # learning rate 
EP = 100   # Number of epochs 
wd_list = [1e-5, 1e-3, 1e-1]
bs = 128

# Training
head = 'EEG_Autoencoder_alpha'
model_path = 'best_models/'
output_file_path = 'outputs/'

for wd_ in wd_list:   
    for i in range(len(Train_seq)):    
        with open('EEG_tensors_bc.pickle', 'rb') as file:
            EEG_tensors = pickle.load(file)
        Train_tensor = EEG_tensors[Train_seq[i][0]]
        Dev_tensor = EEG_tensors[Train_seq[i][1]]
        del EEG_tensors

        # torch.manual_seed(123)

        ## Model head
        model_name = model_path + head + '_lr' + str(lr_) + '_wd' + str(wd_) + 'fd' + str(i) + '.pth'
        output_file_name = output_file_path + head + '_lr' + str(lr_)  + '_wd' + str(wd_) + 'fd' + str(i)  + '.txt'

        Trainer = Autoencoder_Trainer(device = 'cuda:2',
                                      n_observed_channels = 32,
                                      n_hidden_channels = 32,
                                      shrink_list = [32, 32, 16, 8]) #Specify the number of latent channels at each level of compression

        Trainer.fit(Train_tensor, Dev_tensor,
                    epochs = EP,
                    batch_size = bs,
                    learning_rate = lr_,
                    wd = wd_,
                    model_name = model_name,
                    output_file_name = output_file_name)
