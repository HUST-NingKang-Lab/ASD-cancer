import torch
import numpy as np
import pandas as pd
import os
from src.Models import AutoEncoder
from src.Train import train_autoencoders
from src.utils import CoxPH
from src.utils import draw_lifelines
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class Ensemble(object):
    def __init__(self, 
                 num_of_omics: int, 
                 num_of_models: int,
                 num_of_features: list):
        
        self.num_of_omics = num_of_omics
        self.num_of_models = num_of_models
        self.num_of_features = num_of_features
        
        self.autoencoders = [AutoEncoder(num_of_features[i]) for i in range(num_of_omics)]
        self.models = [self.autoencoders for i in range(num_of_models)]
        
        self.scalers = [StandardScaler() for i in range(num_of_omics)]
        
    def fit_ae(self, 
               data, 
               learning_rate: list, 
               batch_size = 25, 
               epochs = 1000,
               model_dir = None,
               pretrained_model_dir = None):
        
        self.encoded_data = []
        # Train autoencoders
        if pretrained_model_dir:
            print('Loading pre-trained models...')
            
            for i, model in enumerate(self.models):
                for j, autoencoder in enumerate(model):
                    autoencoder.load_state_dict(torch.load(f'{pretrained_model_dir}/autoencoder_{i}_{j}.pt'))
            
            print('Training models...')
            
            for i, model in enumerate(self.models):
                print(f'Training model {i+1}/{self.num_of_models}...')
                train_autoencoders(model, data, learning_rate, batch_size, epochs)
            
            # Save models
            os.makedirs(model_dir, exist_ok = True)
            
            for i, model in enumerate(self.models):
                for j, autoencoder in enumerate(model):
                    torch.save(autoencoder.state_dict(), f'{model_dir}/autoencoder_{i}_{j}.pt')
            print('Finished')            
        
        else:
            print('Training models...')
            
            for i, model in enumerate(self.models):
                print(f'Training model {i+1}/{self.num_of_models}...')
                train_autoencoders(model, data, learning_rate, batch_size, epochs)
            
            # Save models
            os.makedirs(model_dir, exist_ok = True)
            
            for i, model in enumerate(self.models):
                for j, autoencoder in enumerate(model):
                    torch.save(autoencoder.state_dict(), f'{model_dir}/autoencoder_{i}_{j}.pt')
             
            print('Finished.')      
        # Encode data
        print('Encoding data...')
        
        for model in self.models:
            encoded_data = []
            
            for i, autoencoder in enumerate(model):
                unscaled = autoencoder(data[:][i])[0].detach()   # unscaled encoded data
                scaled = self.scalers[i].fit_transform(unscaled)
                encoded_data.append(scaled)
                
            self.encoded_data.append(encoded_data)
            
        self.encoded_data = np.array(self.encoded_data) # shape = [num_of_models, num_of_omics, num_of_samples, num_of_features]
                    
        print('Finished.') 
    
    def fit_survival(self, data, results_dir):    
        # Cox for each omic and combined data
        self.pval_frames = []   # list of dataframes containing p-values for each model
        print('Performing Cox PH analysis...')
        
        for i in range(self.num_of_models):
            pval_frame = [] # list of dataframes for each omic
            
            for j in range(self.num_of_omics): 
                pval_frame.append(CoxPH(self.encoded_data[i][j], data[:][-1])) # i: model, j: omic
            
            self.pval_frames.append(pval_frame)
        
        print('Finished.')
        
        # feature selection by p-value
        self.cox_summary = pd.DataFrame(columns = ['model', 'omic', 'feature', 'p'])
        
        self.selected_features = []
        print('Selecting features...')
        
        for i, pval_frame in enumerate(self.pval_frames):
            for j, frame in enumerate(pval_frame):
                for k, pval in enumerate(frame['p']):
                    if pval < 0.05:
                        self.cox_summary = pd.concat([self.cox_summary, pd.DataFrame([{'model': i,
                                                                                      'omic': j,
                                                                                      'feature': k,
                                                                                      'p': pval}])],
                                                        axis=0)
  

        self.cox_summary[['model', 'omic', 'feature']] = self.cox_summary[['model', 'omic', 'feature']].astype(int) # convert model, omic, feature to int
        
        for omic in self.cox_summary['omic'].unique(): # pick the feature for each omic
            omic_feature = pd.DataFrame(index = data.index)
            
            for model in self.cox_summary['model'].unique():
                omic_cox = self.cox_summary[(self.cox_summary['omic'] == omic) & (self.cox_summary['model'] == model)]
                omic_feature = pd.concat([omic_feature, pd.DataFrame(self.encoded_data[model][omic][:, omic_cox['feature'].values.tolist()],
                                                                     columns = [f'model_{model}_omic_{omic}_feature_{int(i)}' for i in omic_cox['feature']],
                                                                     index = data.index)], 
                                          axis = 1)
            
            self.selected_features.append(omic_feature)
        
        self.selected_features.append(pd.concat(self.selected_features, axis = 1)) # combined features
        print('Finished.')
        
        # Gaussion Mixture Model
        print('Performing GMM...')
        self.gmms = [GaussianMixture(random_state = 0) for i in range(self.num_of_omics + 1)] # + 1 for combined data
        self.gmm_result = pd.DataFrame(index = data.index)
        
        for i, gmm in enumerate(self.gmms):
            best_silhouette = 0
            
            for n_component in range(2, 5):
                gmm.set_params(n_components = n_component)
                gmm.fit(self.selected_features[i])
                running_silhouette = silhouette_score(self.selected_features[i], 
                                                      gmm.predict(self.selected_features[i]))                
                if running_silhouette > best_silhouette:
                    best_silhouette = running_silhouette
                    best_k = n_component                   
            
            gmm.set_params(n_components = 2)
        
        for i, gmm in enumerate(self.gmms):
            gmm.fit(self.selected_features[i])
            if i == self.num_of_omics:
                self.gmm_result = pd.concat([self.gmm_result, pd.DataFrame(gmm.predict(self.selected_features[i]), 
                                                                         columns = ['combined'], 
                                                                         index = data.index)], axis = 1)
            else:
                self.gmm_result = pd.concat([self.gmm_result, pd.DataFrame(gmm.predict(self.selected_features[i]), 
                                                                          columns = [f'omic_{i}'],
                                                                          index = data.index)], axis = 1)
        os.makedirs(results_dir, exist_ok = True)      
        self.gmm_result.to_csv(f'{results_dir}/gmm_result.csv')       
        print('Finished.')
        
        # Plot
        print('Plotting...')
        
        for col in self.gmm_result.columns:
            plot_df = pd.concat([self.gmm_result[col],
                                data.survival_meta['OS.time'], 
                                data.survival_meta['OS']], axis=1)
            
            os.makedirs(results_dir, exist_ok = True)
            draw_lifelines(plot_df, 
                           cluster_col=col,
                           title=f'{col} clustering result',
                           save_path=f'{results_dir}/{col}_clustering_result.pdf')
        
        print('Finished.')
        
        
        
                        
                        
                        