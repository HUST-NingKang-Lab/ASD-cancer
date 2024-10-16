import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True

def train_autoencoders(autoencoders: list, train_data, learning_rate: list, batch_size = 25, epochs = 100):
    train_set, valid_set = random_split(train_data, [int(0.8*len(train_data)), len(train_data) - int(0.8*len(train_data))])  # 80% train, 20% valid
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True)

    early_stop = EarlyStopping(patience = 10, min_delta = 0)
    trained_autoencoders = [] # list of trained autoencoders

    for i in range(len(autoencoders)):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoders[i].parameters(), lr = learning_rate[i], weight_decay = 1e-5)
        training_model = autoencoders[i].to(device)
        print(f'-------------------\nTraining autoencoder {i}..\n-------------------')

        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            
            for features in train_loader:   # Train
                feature = features[i].to(device) 
                optimizer.zero_grad()
                encoded, decoded = training_model(feature)
                loss = criterion(decoded,feature)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            with torch.no_grad():   # Valid   
                training_model.eval()         
                for features in val_loader:
                    feature = features[i].to(device)
                    encoded, decoded = training_model(feature)
                    loss = criterion(decoded,feature)
                    val_loss += loss.item()
                    
            early_stop(val_loss/len(valid_set))
                
            # print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_set)}, Valid Loss: {val_loss/len(valid_set)}')           
            
            if early_stop.early_stop:
                # print(f'Early stopping at epoch {epoch}.')
                early_stop.__init__(patience = 10, min_delta = 0)
                break
        
        trained_autoencoders.append(autoencoders[i].to('cpu')) # Save the trained autoencoder
        torch.cuda.empty_cache() # Empty the GPU cache

    return trained_autoencoders

def train_classifier(classifier, train_data, batch_size = 50, learning_rate = 0.001, epochs = 500):
    train_set, valid_set = random_split(train_data, [int(0.8*len(train_data)), len(train_data) - int(0.8*len(train_data))])  # 80% train, 20% valid
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

    early_stop = EarlyStopping(patience = 10, min_delta = 0)
    trained_classifier = classifier.to(device) # Convert to device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr = learning_rate)
    print(f'-------------------\nTraining classifier...\n-------------------')

    for epoch in range(epochs):
        for data in train_loader:   # Train 
            X = data[0].to(device)
            y = data[1].to(device) 
            optimizer.zero_grad()
            output = trained_classifier(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():   # Valid
            X = valid_set[:][0].to(device)
            y = valid_set[:][1].to(device)
            output = trained_classifier(X)
            loss = criterion(output, y)
            accuracy = (torch.argmax(output, dim = 1) == y).sum().item() / y.shape[0]
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()/X.shape[0]}， Accuracy: {accuracy}')
            
            early_stop(loss)
            
            if early_stop.early_stop:
                print(f'Early stopping at epoch {epoch}.')
                break
        
    return trained_classifier.to('cpu') # Save the trained classifier and convert to cpu