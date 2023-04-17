import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
    
class SurvivalDataSet(Dataset):
    def __init__(self, 
                 micro_dir: str, 
                 mRna_dir: str,
                 survival_dir: str, 
                 mRna_size: float = 0.8,
                 is_scale: bool = True,
                 survival_type: str = 'OS'):

        self.mRna_size = mRna_size
        self.is_scale = is_scale
        
        self.metadata = pd.read_csv(survival_dir, index_col = 0)
        self.index = self.metadata.index

        self.micro = pd.read_csv(micro_dir, index_col = 0)
        self.micro = self.micro.div(self.micro.sum(axis = 1), axis = 0) # Normalize the data
        self.micro = self.micro.loc[self.index] # Order the data by the metadata
        
        self.mRna = pd.read_csv(mRna_dir, index_col = 0)
        self.mRna = self.mRna.loc[self.index] # Order the data by the metadata

        # Select survival data
        self.survival_meta = self.metadata[[survival_type, f'{survival_type}.time']]
        
        self.feature_selection_()
            
    def feature_selection_(self):
        mRna_length = int(self.mRna.shape[1] * self.mRna_size)
        selected_gene = self.mRna.var().sort_values(ascending = False).index[:mRna_length]
        self.mRna = self.mRna[selected_gene]
        
        if self.is_scale:
            self.micro_scaler = StandardScaler()
            self.mRna_scaler = StandardScaler()
            self.micro = pd.DataFrame(self.micro_scaler.fit_transform(self.micro.values),
                                      index = self.micro.index,
                                      columns = self.micro.columns)
            self.mRna = pd.DataFrame(self.mRna_scaler.fit_transform(self.mRna.values),
                                     index = self.mRna.index,
                                     columns = self.mRna.columns)
        
    def __getitem__(self, index: int):
        return torch.tensor(self.micro.iloc[index].astype(float).values, dtype = torch.float32), \
                    torch.tensor(self.mRna.iloc[index].astype(float).values, dtype = torch.float32), \
                    torch.tensor(self.survival_meta.iloc[index].values, dtype = torch.float32)       

    def __len__(self):
        return self.survival_meta.shape[0]  