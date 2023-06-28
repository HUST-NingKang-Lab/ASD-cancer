# ASD-cancer
## Introduction
ASD-cancer (autoencoder-based subtypes detector for cancer) is a semi-supervised deep learning framework based on autoencoder. In our study, we used autoencoder models to extract relevant features from normalized microbiome abundance data and transcriptome data for identifying cancer survival subtypes. We then analyzed these extracted features using univariate Cox proportional hazards (Cox-PH) regression to identify a subset of survival-related features. To ensure an adequate number of features, we implemented an ensemble step using a total of 20 models. We determined the number of survival subtypes using Gaussian mixture models and the highest silhouette score.

![](ASD-cancer.png)

## Requirements
The code is written in Python 3.10. The required packages are listed in `requirements.txt`. To install the required packages, run the following command:
```
pip install -r requirements.txt
```
If you want to use GPU acceleration, you can install the GPU version of PyTorch according to the [official website](https://pytorch.org/get-started/locally/).

## Command line instructions
To run the script using command line with arguments, use the following format:

```
python main.py -a <micro_dir> \
-r <mRna_dir> \
-s <survival_dir> \
-m <model_dir> \
-p <pretrained_model_dir> \
-o <results_dir> \
```

The arguments are:

`-a` or `--micro_dir`: A CSV file containing the abundance of microbial communities in the sample. The rows represent the hosts and the columns represent the microbes. The format of the file should look like:
|| microbe1| microbe2|...|
|---|---|---|---|
|host1|0.01|0.05|...|
|host2|0|0.02|...|
|...|...|...|...|

`-r` or `--mRna_dir`: A CSV file containing the expression of fpkm normalized mRNA in the sample. The rows represent the hosts and the columns represent the genes. The format of the file should look like:
|| gene1| gene2|...|
|---|---|---|---|
|host1|20|70|...|
|host2|40|120|...|
|...|...|...|...|

`-s` or `--survival_dir`: A CSV file containing the survival information of the hosts. The file should contains three columns: `sample_id`, `OS` and `OS.time`. `sample_id` is the sample ID of the host, `OS` is the survival status of the host, and `OS.time` is the survival time of the host. The format of the file should look like:
|sample_id| OS| OS.time|
|---|---|---|
|host1|1|150|
|host2|0|200|
|...|...|...|

`-n` or `--num_of_models`: (optional) number of ensemble models. Default is 20.

`-m` or `--model_dir`: directory to save or load trained models.

`-p` or `--pretrained_model_dir`: (optional) directory to load pretrained models. If this argument is not provided, the script will train new models and save them in the directory specified by `-m` or `--model_dir`. If this argument is provided, the script will load the pretrained models from the directory specified by `-p` or `--pretrained_model_dir` and save the optimized models in the directory specified by `-m` or `--model_dir`.

`-o` or `--results_dir`: directory to save results. The results contain the following files: A CSV file containing the survival subtypes of the hosts and three PNG files containing the survival subtype results of the tumor microbiome, the host gene and their integration results with p-values tested by log-rank test.

## Example
We provide the data of LIHC (Liver hepatocellular carcinoma) as an example dataset in the `data` folder. To run the script using the sample dataset, use the following command:

To load the pretrained models and save the results in the `results` folder:

```
python main.py -a data/micro.csv \
-r data/mRNA-fpkm.csv \
-s data/survival_meta.csv \
-p models \
-m trained_models \
-o results \
```
