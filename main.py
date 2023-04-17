import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
import os
from src.Ensemble import Ensemble
from src.CancerDataSets import SurvivalDataSet  
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-a', '--micro_dir', type=str, help='directory of tumor microbiome abundance data ')
parser.add_argument('-r', '--mRna_dir', type=str, help='directory of host mRNA fpkm data')
parser.add_argument('-s', '--survival_dir', type=str, help='directory of host survival information')
parser.add_argument('-n', '--num_of_models', type=int, default=20, help='number of models')
parser.add_argument('-m', '--model_dir', type=str, help='directory to save trained models')
parser.add_argument('--load-model', action='store_true', help='load trained models')
parser.add_argument('-o', '--results_dir', type=str, help='directory to save results')

args = parser.parse_args()

if __name__ == '__main__':     
    dataset = SurvivalDataSet(micro_dir=args.micro_dir,
                              mRna_dir=args.mRna_dir,
                              survival_dir=args.survival_dir)
    
    ensemble = Ensemble(num_of_omics=2,
                        num_of_models=args.num_of_models,
                        num_of_features=[dataset.micro.shape[1], dataset.mRna.shape[1]])
    
    ensemble.fit_ae(dataset, 
                    learning_rate=[1e-5, 1e-5], 
                    epochs=1000, 
                    load_model=args.load_model,
                    model_dir=args.model_dir)
    
    ensemble.fit_survival(dataset, results_dir=args.results_dir)            