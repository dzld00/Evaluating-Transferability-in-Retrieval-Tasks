import os
import json
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import numpy as np
# from PIL import Image
import time
import random
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA, KernelPCA
# from statistics import mean, stdev
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.preprocessing import scale, normalize, StandardScaler
from sklearn.datasets import fetch_20newsgroups
from numpy.linalg import inv, norm, det
from scipy.linalg import logm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp, mannwhitneyu
from functools import reduce
import sys
import skbio
from skbio import DistanceMatrix
# import torchvision.transforms as transforms
# import timm
import scipy.io
import pickle

from text_dataset_preparation import prepare_dataset, get_aug_text
from utils_text_retrieval import create_text_model, create_embeddings, use_pre_computed_embeddings, check_numbers, log_dist, find_optimal_weight, weighted_difference, calculate_scores

warnings.filterwarnings("ignore")
from pdb import set_trace as keyboard

from scipy.stats import kendalltau
from scipy.stats import kendalltau, weightedtau
from sklearn.manifold import MDS


def find_correlation(train_vectors, val_vectors):
    corr, _ = kendalltau(train_vectors, val_vectors)
    return corr


def find_weighted_correlation(train_vectors, val_vectors):
    corr = weightedtau(train_vectors, val_vectors)
    return corr.correlation

def eval():
    # os.makedirs('figs_image', exist_ok=True)
    os.makedirs('raw_text_dataset', exist_ok=True)

    n_neg = 50 
    n_pos = 5
    n_samples = 5
    n_groups_pos = n_pos // n_samples
    n_groups_neg = n_neg // n_samples
    n_query = 90
    n_combo = 1
    n_components = 5
    ### datasets include: fiqa, nfcorpus, scifact
    #datasets = ['fever', 'fiqa', 'nfcorpus']
    datasets = ['nfcorpus']

    os.makedirs(f'processed_text_dataset_{n_pos}_pos_sample_{n_query}_query', exist_ok=True)
    os.makedirs(f'generated_embeddings_{n_pos}_pos_sample_{n_query}_query', exist_ok=True)
    os.makedirs(f'eval_metrics_{n_pos}_pos_sample_{n_query}_query_MMD', exist_ok=True)

    # random.seed(1234)
    # torch.manual_seed(1234)

    for dataset in datasets:
        if os.path.exists(os.path.join(os.getcwd(), f'processed_text_dataset_{n_pos}_pos_sample_{n_query}_query', f'{dataset}.pkl')):
            data_df = pd.read_pickle(os.path.join(os.getcwd(), f'processed_text_dataset_{n_pos}_pos_sample_{n_query}_query', f'{dataset}.pkl')).iloc[:n_query]
        else:
            data_df = prepare_dataset(dataset, n_query, n_pos, n_neg, random_state=1234)
            data_df.to_pickle(os.path.join(os.getcwd(), f'processed_text_dataset_{n_pos}_pos_sample_{n_query}_query', f'{dataset}.pkl'))
            aug_id_to_text = get_aug_text(data_df)
            with open(os.path.join(os.getcwd(), f'processed_text_dataset_{n_pos}_pos_sample_{n_query}_query', f'{dataset}_aug_text.json'), "w") as json_file:
                json.dump(aug_id_to_text, json_file)
        
        models = ['distilbert-base-uncased', 'all-MiniLM-L6-v1', 'all-distilroberta-v1', 'nq-distilbert-base-v1', 'all-MiniLM-L12-v1', 'msmarco-distilbert-dot-v5']
        pre_embed_models = ['openai']
        
        metrics = {}

        for model_name in models:

            if os.path.exists(os.path.join(os.getcwd(), f'generated_embeddings_{n_pos}_pos_sample_{n_query}_query', f'{dataset}_{model_name}.pkl')):
                embeddings_lists = pickle.load(open(os.path.join(os.getcwd(), f'generated_embeddings_{n_pos}_pos_sample_{n_query}_query', f'{dataset}_{model_name}.pkl'), 'rb'))[:n_query]
            else:
                if model_name in pre_embed_models:
                    embeddings_dir = os.path.join(os.path.dirname(os.getcwd()), f"{model_name}_embed")
                    embeddings_lists = use_pre_computed_embeddings(data_df, dataset, embeddings_dir)
                else:
                    model = create_text_model(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    embeddings_lists = create_embeddings(data_df, model, device='cuda' if torch.cuda.is_available() else 'cpu')
                pickle.dump(embeddings_lists, open(os.path.join(os.getcwd(), f'generated_embeddings_{n_pos}_pos_sample_{n_query}_query', f'{dataset}_{model_name}.pkl'), 'wb'))
            
            torch.cuda.empty_cache()

            metrics[model_name] = calculate_scores(
                        embeddings_lists, 
                        n_combo,
                        n_pos,
                        n_samples,
                        n_neg,
                        n_components)

        results = pd.DataFrame(metrics)
        results.to_csv(os.path.join(f'eval_metrics_{n_pos}_pos_sample_{n_query}_query_MMD', f'{dataset}.csv'))

        dic_Y = {}
        dic_Y['fiqa'] = [
            0.377709,  # distilbert-base-uncased
            0.439628,  # all-MiniLM-L6-v1
            0.396285,  # all-distilroberta-v1
            0.402477,  # nq-distilbert-base-v1
            0.458204,  # all-MiniLM-L12-v1
            0.433437   # msmarco-distilbert-dot-v5
        ]
        dic_Y['nfcorpus'] = [
            0.393189,  # distilbert-base-uncased
            0.470588,  # all-MiniLM-L6-v1
            0.430245,  # all-distilroberta-v1
            0.390093,  # nq-distilbert-base-v1
            0.486820,  # all-MiniLM-L12-v1
            0.424149   # msmarco-distilbert-dot-v5
        ]
        dic_Y['scifact'] = [
            0.65,  # distilbert-base-uncased
            0.60,  # all-MiniLM-L6-v1
            0.68,  # all-distilroberta-v1
            0.66,  # nq-distilbert-base-v1
            0.63,  # all-MiniLM-L12-v1
            0.69   # msmarco-distilbert-dot-v5
        ]
        #dic_Y['scifact'] = [0.41,0.44,0.38,0.42,0.47,0.44]

        X = [metrics[key]['MMD 1'] for key in metrics]
        corr_w = weightedtau(X, dic_Y[dataset]).correlation
        print(dataset, 'weighted Kendall rank correlation:', corr_w)
    

if __name__ == "__main__":
    eval()

