import os
import json
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
# requirements: transformers 3.5.1 dead kernel; pip install transformers==4.21.1
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess, load_model
from lavis.models.base_model import BaseModel
import time
import random
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
# requirements: transformers 3.5.1 dead kernel; pip install transformers==4.21.1
from sklearn.decomposition import PCA,KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split as svm_train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from statistics import mean,stdev
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.covariance import LedoitWolf, MinCovDet, EmpiricalCovariance
from sklearn.preprocessing import scale, normalize, StandardScaler
from numpy.linalg import inv, norm, det
from scipy.linalg import logm
from functools import reduce 
import sys
import skbio
from skbio import DistanceMatrix
import sys
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import math
from logME import LogME
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau
from scipy.stats import kendalltau, weightedtau
from sklearn.manifold import MDS
from scipy.spatial.distance import jensenshannon
from numpy import sqrt, histogram
from scipy.stats import entropy


warnings.filterwarnings("ignore")

# random.seed(1234)
# torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# Function to calculate KL divergence between two sets of embeddings
def kl_divergence(embeddings_p, embeddings_q, num_bins=50):
    # Flatten the embeddings to 1D
    embeddings_p_flat = embeddings_p.flatten()
    embeddings_q_flat = embeddings_q.flatten()
    
    # Calculate histograms for both distributions
    min_val = min(embeddings_p_flat.min(), embeddings_q_flat.min())
    max_val = max(embeddings_p_flat.max(), embeddings_q_flat.max())
    
    p_hist, _ = np.histogram(embeddings_p_flat, bins=num_bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(embeddings_q_flat, bins=num_bins, range=(min_val, max_val), density=True)
    
    # Add a small constant to avoid zeros in histograms (and hence log(0) in KL)
    p_hist += 1e-10
    q_hist += 1e-10
    
    # Normalize the histograms to sum to 1 (form a probability distribution)
    p_hist /= np.sum(p_hist)
    q_hist /= np.sum(q_hist)
    
    # Calculate the KL divergence
    kl_div = entropy(p_hist, q_hist)
    
    return kl_div


# Function to calculate Jensen-Shannon divergence between two sets of embeddings
def js_divergence(positive_embeddings, negative_embeddings, bins=50):
    # Creating histograms for both distributions
    min_val = min(positive_embeddings.min(), negative_embeddings.min())
    max_val = max(positive_embeddings.max(), negative_embeddings.max())

    pos_hist, _ = histogram(positive_embeddings, bins=bins, range=(min_val, max_val), density=True)
    neg_hist, _ = histogram(negative_embeddings, bins=bins, range=(min_val, max_val), density=True)

    # Ensure no zero values to avoid division by zero in Jensen-Shannon calculation
    pos_hist += 1e-10
    neg_hist += 1e-10

    # Normalize the histograms
    pos_hist /= pos_hist.sum()
    neg_hist /= neg_hist.sum()

    # Calculate Jensen-Shannon Divergence and convert to distance
    js_div = jensenshannon(pos_hist, neg_hist) ** 2
    js_dist = sqrt(js_div)
    
    return js_dist

# Dummy data for example
positive_embeddings = np.random.rand(1000)  # Replace with your actual data
negative_embeddings = np.random.rand(1000)  # Replace with your actual data

# Calculate JS distance
js_distance = js_divergence(positive_embeddings, negative_embeddings)
print("Jensen-Shannon Distance:", js_distance)


def permutation_test_mmd(x, y, kernel, degree=None, num_permutations=1000):
    
    # Assuming you have a MMD function that works with numpy arrays
    if kernel == 'poly':
        mmd_observed,_,_,_ = MMD(x, y, kernel='poly', degree=degree)
    else:
        mmd_observed,_,_,_ = MMD(x, y, kernel)
    
    combined = np.concatenate([x, y], axis=0)
    mmd_permuted_values = []
    
    for _ in range(num_permutations):
        perm_indices = np.random.permutation(len(combined))
        permuted = combined[perm_indices]
        x_perm = permuted[:len(x)]
        y_perm = permuted[len(x):]
        
        if kernel == 'poly':
            mmd_permuted,_,_,_ = MMD(x_perm, y_perm, kernel='poly', degree=degree)
        else:
            mmd_permuted,_,_,_ = MMD(x_perm, y_perm, kernel)
            
        mmd_permuted_values.append(mmd_permuted)
    
    # Calculate p-value
    mmd_permuted_values = np.array(mmd_permuted_values)
    p_value = np.mean(mmd_observed < mmd_permuted_values)
    
    return p_value


def find_correlation(train_vectors, val_vectors):
    corr, _ = kendalltau(train_vectors, val_vectors)
    return corr


def find_weighted_correlation(train_vectors, val_vectors):
    corr = weightedtau(train_vectors, val_vectors)
    return corr.correlation


def MMD(X, Y, kernel='rbf', **kwargs):
    n = X.shape[0]
    m = Y.shape[0]

    # Compute pairwise kernel values
    K_YY = pairwise_kernels(Y, Y, metric=kernel, **kwargs)
    np.fill_diagonal(K_YY, 0)
    
    # If X has only one element but it's not correct
    if n == 1:
        #raise ValueError("The size of X cannot be 1.")
        K_XY = pairwise_kernels(X, Y, metric=kernel, **kwargs)
        return K_YY.mean() - K_XY.mean()
    else:
        K_XX = pairwise_kernels(X, X, metric=kernel, **kwargs)
        np.fill_diagonal(K_XX, 0)
        K_XY = pairwise_kernels(X, Y, metric=kernel, **kwargs)
        #return (np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * K_XY.mean()) 
        #return (np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY)/(m*n)) 
        kxx = np.sum(K_XX) / (n * (n - 1))
        kyy = np.sum(K_YY) / (m * (m - 1))
        kxy = np.sum(K_XY)/(m*n)
        return kxx + kyy - 2*kxy, kxx, kyy, kxy
    

def MMD_sk(X, Y, kernel='rbf', **kwargs):
    n = X.shape[0]
    m = Y.shape[0]

    # Compute pairwise kernel values
    K_YY = pairwise_kernels(Y, Y, metric=kernel, **kwargs)
    np.fill_diagonal(K_YY, 0)
    
    K_XX = pairwise_kernels(X, X, metric=kernel, **kwargs)
    np.fill_diagonal(K_XX, 0)
    K_XY = pairwise_kernels(X, Y, metric=kernel, **kwargs)
    return (np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * K_XY.mean())


import ot 

def compute_wasserstein(X, Y):
    # Compute pairwise distance matrices for X and Y
    Mxx = ot.dist(X, X)
    Myy = ot.dist(Y, Y)
    Mxy = ot.dist(X, Y)
    
    # Compute marginals
    px = np.ones(X.shape[0]) / X.shape[0]
    py = np.ones(Y.shape[0]) / Y.shape[0]

    # Compute Wasserstein distance
    wasserstein_distance = ot.emd2(px, py, Mxy)
    
    return wasserstein_distance


def BC_dist(mu1, cov1, mu2, cov2):
    mean_cov = 0.5 * (cov1 + cov2)
    d = ((mu1 - mu2).T @ np.linalg.inv(mean_cov) @ (mu1 - mu2) / 8.).reshape(-1)[0]\
            + np.log(np.linalg.det(mean_cov)/(np.linalg.det(cov1) * np.linalg.det(cov2))**0.5) / 2.
    return -np.exp(-d)


def estimate_gamma(X, Y):
    # Compute pairwise distances between all pairs of samples
    distances = pairwise_distances(np.vstack((X, Y)), metric="euclidean")
    
    # Compute the median of the pairwise distances
    median_distance = np.median(distances)
    
    # Return the inverse of the median distance as gamma
    gamma = 1.0 / median_distance**2
    return gamma


def twonn_dimension(X):
    """
    Estimate intrinsic dimensionality using the Two Nearest Neighbors (TWONN) method.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features) - the data embeddings
    
    Returns:
    - Estimated intrinsic dimension
    """
    
    # Compute pairwise distances
    distances = euclidean_distances(X)
    
    # Set diagonal to high value to exclude self-distance
    np.fill_diagonal(distances, np.inf)
    
    # Sort each row and pick the first and second smallest values (smallest being the nearest neighbor)
    r1 = np.sort(distances)[:, 0]
    r2 = np.sort(distances)[:, 1]
    
    # Compute average ratio
    rho_avg = np.mean(r1 / (r2+1e-8))
    
    # Estimate intrinsic dimension
    D = 2 * np.log(2) / -np.log(rho_avg)
    
    return D


from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def mle_dimension(X, k=5):
    """
    Estimate intrinsic dimensionality using the MLE for Intrinsic Dimensionality (MIND) method.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features) - the data embeddings
    - k: int - the neighbor to use (typically a small integer like 5 or 6)
    
    Returns:
    - Estimated intrinsic dimension
    """
    
    # Compute pairwise distances
    distances = euclidean_distances(X)
    
    # Set diagonal to high value to exclude self-distance
    np.fill_diagonal(distances, np.inf)
    
    # Sort each row and pick the k-th and (k+1)-th smallest values
    r_k = np.sort(distances)[:, k-1]  # indices are 0-based, hence k-1
    r_kplus1 = np.sort(distances)[:, k]  # k-th neighbor
    
    # Compute average distances for R_k and R_kplus1
    R_k = np.mean(r_k)
    R_kplus1 = np.mean(r_kplus1)
    
    # Estimate intrinsic dimension using MLE
    D = -2 / np.log(R_k / R_kplus1)
    
    return D


def mds_dimension(embeddings):
    batch_size = embeddings.shape[0] 

    D = pairwise_distances(embeddings)   
    D = D / np.amax(D)
    
    #l_sorted = cmdscale(D) 
    l_sorted = eigen_mds(D)               
    
    # Calculate k,p based on number of large eigenvalues
    k = batch_size - next(x[0] for x in enumerate(l_sorted) if x[1] < 0.1 * l_sorted[0])    
    
    return k  


def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    
    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only                      
#     w, = np.where(evals > 0)
#     L  = np.diag(np.sqrt(evals[w]))
#     V  = evecs[:,w]
#     Y  = V.dot(L)
 
    return np.sort(evals)[::-1]


def eigen_mds(pd):   
    mds = MDS(n_components=len(pd), dissimilarity='precomputed')
    pts = mds.fit_transform(pd)

    _,l_sorted,_ = np.linalg.svd(pts)
    
    return l_sorted


from scipy.spatial import distance
def compute_curvature(embeddings, p, k=5):
    """
    Compute curvature of data based on PCA and circle fitting.
    
    Parameters:
    - embeddings: The set of embeddings.
    - k: Number of nearest neighbors to consider for circle fitting.
    
    Returns:
    - Average curvature of the data.
    """
    
    # Step 1: Project embeddings to first two principal components
    pca = PCA(n_components=p)
    projected_data = pca.fit_transform(embeddings)
    
    # Step 2: For every point, compute the curvature
    curvatures = []
    for point in projected_data:
        # Compute distances to other points
        distances = np.apply_along_axis(lambda x: distance.euclidean(x, point), 1, projected_data)
        # Get k nearest neighbors
        neighbors = projected_data[np.argsort(distances)[:k]]
        
        # Circle fitting
        # Using the formula: curvature = 1 / R, where R is the radius of the circle
        # We use a simple mean for R estimation from the distances
        R = np.mean(distances[np.argsort(distances)[:k]])
        curvature = 1 / R if R != 0 else 0  # Avoid division by zero
        
        curvatures.append(curvature)
        
    # Step 3: Return average curvature
    return np.mean(curvatures)



def lavis_extract_text_feature(model_name, item_dict, N, embedding_type):
    NAME = ''
    if model_name == 'albef':
        NAME = "albef_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="base", is_eval=True, device=device)
    elif model_name == 'blip':
        NAME = "blip_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="base", is_eval=True, device=device)
    elif model_name == 'blip2':
        NAME = "blip2_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="pretrain", is_eval=True, device=device)
    elif model_name == 'alpro':
        NAME = "alpro_retrieval"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="msrvtt", is_eval=True, device=device)
    elif model_name == 'clip':
        NAME = "clip_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="ViT-B-32", is_eval=True, device=device)
        #model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="ViT-L-14", is_eval=True, device=device)

    model.eval()

    embed_dim = 768
    if model_name == 'clip':
        embed_dim = 512 #768 #vit-L-14 #512 vit-B-32 
    elif embedding_type == 'text':
        embed_dim = 256

    features = torch.zeros(size=(len(item_dict), N, embed_dim))

    with torch.no_grad():
        idx1 = 0
        for image_name in item_dict:
            raw_image = Image.open(image_name).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            idx2 = 0
            cnt_text = 0
            for text in item_dict[image_name]:
                if model_name == 'clip':
                    sample = {"image": image, "text_input": text}
                    clip_features = model.extract_features(sample)
                    features[idx1,idx2,:] = clip_features.text_embeds_proj
                else:
                    if embedding_type == 'multimodal':
                        # # Multimodal embedding 
                        sample = {"image": image, "text_input": text}
                        feature = model.extract_features(sample)
                        features[idx1,idx2,:] = feature.multimodal_embeds[:,0,:] 
                    elif embedding_type == 'text':
                        # Text embedding
                        sample = {"image": None, "text_input": text}
                        features_text = model.extract_features(sample, mode="text")
                        features[idx1,idx2,:] = features_text.text_embeds_proj[:,0,:]
                cnt_text += 1
                if cnt_text == 5: # number of texts per image in COCO
                    break

                torch.cuda.empty_cache()
                idx2 += 1
            idx1 += 1
    return features


def lavis_extract_text_feature_per_image(model_name, item_dict, image_name, N, embedding_type):
    NAME = ''
    if model_name == 'albef':
        NAME = "albef_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="base", is_eval=True, device=device)
    elif model_name == 'blip':
        NAME = "blip_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="base", is_eval=True, device=device)
    elif model_name == 'blip2':
        NAME = "blip2_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="pretrain", is_eval=True, device=device)
    elif model_name == 'alpro':
        NAME = "alpro_retrieval"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="msrvtt", is_eval=True, device=device)
    elif model_name == 'clip':
        NAME = "clip_feature_extractor"
        model, vis_processors, txt_processors = load_model_and_preprocess(name=NAME, model_type="ViT-B-32", is_eval=True, device=device)

    model.eval()

    embed_dim = 768
    if model_name == 'clip':
        embed_dim = 512
    elif embedding_type == 'text':
        embed_dim = 256

    features = torch.zeros(size=(N, embed_dim))
    with torch.no_grad():
        raw_image = Image.open(image_name).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        idx = 0
        cnt_text = 0
        for text in item_dict[image_name]:
            if model_name == 'clip':
                sample = {"image": image, "text_input": text}
                clip_features = model.extract_features(sample)
                features[idx,:] = clip_features.text_embeds_proj
            else:
                if embedding_type == 'multimodal':
                    # # Multimodal embedding 
                    sample = {"image": image, "text_input": text}
                    feature = model.extract_features(sample)
                    features[idx,:] = feature.multimodal_embeds[:,0,:] 
                elif embedding_type == 'text':
                    # Text embedding
                    sample = {"image": None, "text_input": text}
                    features_text = model.extract_features(sample, mode="text")
                    features[idx,:] = features_text.text_embeds_proj[:,0,:]
            cnt_text += 1
            if cnt_text == 5: # number of texts per image in COCO
                break

            torch.cuda.empty_cache()
            idx += 1
    return features


def softplus(x):
    return np.log(1+np.exp(x))


def sigmoid(x):
    return 1 / (1+np.exp(-x))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="COCO", choices=['COCO'])
parser.add_argument("--model_name", type=str, default="blip")
parser.add_argument("--n_queries", default=100, type=int)
parser.add_argument("--n_pos", default=5, type=int)
parser.add_argument("--n_neg", default=50, type=int)
parser.add_argument("--n_components", default=50, type=int)
parser.add_argument("--embedding_type", type=str, default="multimodal", choices=['multimodal','text'])
parser.add_argument("--kernel_type", type=str, default='linear')
parser.add_argument("--degree", default=3, type=int, help='Degree for polynomial kernel')


args = parser.parse_args()

def main(args):
    # Set the paths to the Karpathy JSON file and images
    karpathy_json_file = "/data/mdai/lavis/coco_default/annotations/coco_karpathy_test.json"
    images_path = "/data/mdai/lavis/coco_default/images"
    n_neg = args.n_neg  # Number of negative captions to select for each image
    n_pos = args.n_pos

    # Load dataset
    with open(karpathy_json_file, "r") as f:
        dataset = json.load(f)

    num_images = len(dataset)
    random_indexes = random.sample(range(num_images), args.n_queries)
    #random_indexes = [1559, 1297, 2006, 1972, 3770]
    #print(random_indexes)

    # Randomly select 10 datapoints
    random_img_data_list = random.sample(dataset, args.n_queries)
    # Create a dictionary to store image file paths and captions
    image_caption_dict = {}
    # Select the images based on random indexes
    for index in random_indexes:
        img_data = dataset[index]

        # Get the image file path
        img_file_path = os.path.join(images_path, img_data['image'])

        # Get annotations
        annotations = img_data["caption"]

        # Add the image file path and captions to the dictionary
        image_caption_dict[img_file_path] = annotations

    # # Print the dictionary with image file paths and captions
    # for img_path, captions in image_caption_dict.items():
    #     print(f"Image: {img_path}")
    #     print(f"Captions: {captions}\n")


    # Select N negative captions for each image
    negative_caption_dict = {}
    for img_path, captions in image_caption_dict.items():
        negative_captions = []

        while len(negative_captions) < n_neg:
            # Randomly select an image
            random_img_data = random.choice(dataset)

            # Get the image file path
            random_img_path = os.path.join(images_path, random_img_data['image'])

            # Ensure the randomly selected image is not the same as the original image
            if img_path != random_img_path:
                # Randomly select a negative caption from the randomly selected image
                random_caption = random.choice(random_img_data["caption"])

                # Add the negative caption to the list if it's not already present
                if random_caption not in negative_captions:
                    negative_captions.append(random_caption)

        # Add the image file path and negative captions to the negative_caption_dict
        negative_caption_dict[img_path] = negative_captions

    # # Print the dictionary with image file paths and negative captions
    # for img_path, neg_captions in negative_caption_dict.items():
    #     print(f"Image: {img_path}")
    #     print(f"Negative Captions: {neg_captions}\n")

    #exit(0)


    model_name = args.model_name
    #print(model_name)
    features_pos = lavis_extract_text_feature(model_name, image_caption_dict, n_pos, args.embedding_type)
    features_neg = lavis_extract_text_feature(model_name, negative_caption_dict, n_neg, args.embedding_type)

    # print(features_pos.size())
    # print(features_neg.size())
    # exit(0)


    mmd_scores = []
    kxx_scores, kyy_scores, kxy_scores = [], [], []
    intrinsic_dimensions = []
    BC_list, logme_list = [], []
    cnt_p_value = 0
    f1_list = []
    from collections import defaultdict
    dic_kernel = defaultdict(int)

    for idx in range(len(image_caption_dict)): 
        positive_embeddings = features_pos[idx,:,:].squeeze(0).detach().cpu().numpy()
        negative_embeddings = features_neg[idx,:,:].squeeze(0).detach().cpu().numpy()
        # print(positive_embeddings.shape)
        # print(negative_embeddings.shape)

        # if PCA
        embeddings = np.vstack((positive_embeddings, negative_embeddings)) #- query_embedding
        sc = StandardScaler()
        standardized_embeddings = sc.fit_transform(embeddings)
        standardized_positive_embeddings = standardized_embeddings[:n_pos,:]
        standardized_negative_embeddings = standardized_embeddings[n_pos:,:]

        pca = PCA()
        pca.fit(standardized_embeddings)
        explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
        n_components = (explained_variance_ratio < 0.9).sum() + 1

        transform_ = PCA(n_components=n_components)
        standardized_embeddings = np.concatenate((standardized_positive_embeddings, standardized_negative_embeddings),axis=0)
        transform_pos_neg = transform_.fit(standardized_embeddings)
        transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
        positive_embeddings = transformed_embeddings[:n_pos,:]
        negative_embeddings = transformed_embeddings[n_pos:,:]

        # kernel_types = ['linear','cosine','poly','rbf','laplacian']
        # for kernel_type in kernel_types:
        #     p_val = permutation_test_mmd(positive_embeddings, negative_embeddings, kernel_type, degree=args.degree, num_permutations=1000)
        #     if p_val < 0.05:
        #         dic_kernel[kernel_type] += 1

        # # # # estimate intrinsic dimension of embeddings
        # # #estimated_dimension = twonn_dimension(embeddings)
        # # estimated_dimension = mle_dimension(embeddings)
        # # #estimated_dimension = mds_dimension(embeddings)
        # #estimated_dimension = compute_curvature(embeddings, p=n_components, k=int(np.sqrt(args.n_pos+args.n_neg)))
        # estimated_dimension = compute_curvature(embeddings, p=n_components, k=5)
        # intrinsic_dimensions.append(estimated_dimension)
        # #print(estimated_dimension)

        #mmd_value = js_divergence(positive_embeddings, negative_embeddings)
        mmd_value = kl_divergence(positive_embeddings, negative_embeddings)
        # if args.kernel_type == 'poly':
        #     mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel='poly',degree=args.degree)
        # else:
        #     mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel=args.kernel_type)
        # kxx_scores.append(kxx)
        # kyy_scores.append(kyy)
        # kxy_scores.append(kxy)
        mmd_scores.append(mmd_value)


        # ###LogME
        # logme = LogME(regression=False)
        # np_labels =np.asarray([0]*standardized_positive_embeddings.shape[0] + [1]*standardized_negative_embeddings.shape[0])
        # cur_logme = logme.fit(standardized_embeddings, np_labels)
        # logme_list.append(cur_logme)  

        # ### BC
        # transform_ = PCA(n_components=5)
        # transform_pos_neg = transform_.fit(standardized_embeddings)
        # transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
        # positive_embeddings = transformed_embeddings[:n_pos,:]
        # negative_embeddings = transformed_embeddings[n_pos:,:]
        # cur_BC = BC_dist(np.mean(positive_embeddings, axis=0), EmpiricalCovariance().fit(positive_embeddings).covariance_ , np.mean(negative_embeddings, axis=0), EmpiricalCovariance().fit(negative_embeddings).covariance_)
        # BC_list.append(cur_BC) 

        # p_val = permutation_test_mmd(positive_embeddings, negative_embeddings, args.kernel_type, degree=args.degree, num_permutations=1000)
        # if p_val < 0.05:
        #     cnt_p_value += 1  

        # SVM
        # X = np.concatenate([positive_embeddings, negative_embeddings], axis=0)
        # y = np.array([1] * len(positive_embeddings) + [0] * len(negative_embeddings))  # 1 for positive, 0 for negative
        # X_train, X_test, y_train, y_test = svm_train_test_split(X, y, test_size=0.2, random_state=42)
        # # Apply SMOTE to generate synthetic positive examples
        # smote = SMOTE(k_neighbors=min(len(y_train[y_train == 1]) - 1, 5))
        # X_train, y_train= smote.fit_resample(X_train, y_train)
        # svm_clf = SVC(kernel=args.kernel_type, degree=args.degree, random_state=42)
        # svm_clf.fit(X_train, y_train)
        # y_pred = svm_clf.predict(X_test)
        # f1 = f1_score(y_test, y_pred)
        # #recall = recall_score(y_test, y_pred)
        # f1_list.append(f1)

        torch.cuda.empty_cache()

    metric = np.mean(mmd_scores)#cnt_p_value #np.mean(mmd_scores) #np.log(np.mean(mmd_scores))

    # # print(np.log(np.sum(mmd_scores)))
    # # print('PCA components:', n_components)
    # print("Average intrinsic dimension:", round(np.mean(intrinsic_dimensions),2))
    # print("Std of intrinsic dimension:", round(np.std(intrinsic_dimensions),2))
    # print('kxx:', np.sum(kxx_scores), 'kyy:', np.sum(kyy_scores), 'kxy:', np.sum(kxy_scores))
    # print('kyy to kxx ratio: ', round(np.sum(kyy_scores)/np.sum(kxx_scores),2), '  kxy to kxx ratio: ', round(np.sum(kxy_scores)/np.sum(kxx_scores),2))
    # print('Number of significant p-value:', cnt_p_value)
    # print('CV:',np.std(mmd_scores)/np.mean(mmd_scores))

    return metric #dic_kernel #np.mean(f1_list) #metric, np.mean(BC_list), np.mean(logme_list)


if __name__ == "__main__":
    #main(args)

    model_names = ["clip", "albef", "blip","blip2"]  
    datasets = ['COCO']
    
    from collections import defaultdict
    dic_Y = defaultdict(list)
    dic_Y['COCO'] = [57.98,94.3,95.4,96.0]
    
    import csv
    with open('KRC_mm_mmd_ratio.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset', 'Degree', 'Weighted Kendall Rank Correlation mean', 'Weighted Kendall Rank Correlation std'])    
        
        for dataset in datasets:
            print('dataset:', dataset)
            print()
            for degree in range(10,11):
                print('degree:', degree)
                corr_list, corr_w_list = [], []
                corr_list_BC, corr_w_list_BC = [], []
                corr_list_logme, corr_w_list_logme = [], []
                for i in range(5):
                    print('Iteration {}'.format(i+1))
                    X = []
                    X_BC = []
                    X_logme = []
                    for model_name in model_names:
                        args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, "--degree", str(degree), 
                                                  "--kernel_type",'poly'])
                        metric = main(args)
                        X.append(metric)
                        # X_BC.append(BC)
                        # X_logme.append(logme)
                        #print(dataset, model_name, BC)
                    corr_w = weightedtau(X, dic_Y[dataset]).correlation
                    corr_w_list.append(corr_w)
                    # corr_w_BC = weightedtau(X_BC, dic_Y[dataset]).correlation
                    # corr_w_list_BC.append(corr_w_BC)
                    # corr_w_logme = weightedtau(X_logme, dic_Y[dataset]).correlation
                    # corr_w_list_logme.append(corr_w_logme)
                print('Weighted Kendall Rank correlation for proposed metric -- mean: %.5f'% np.mean(corr_w_list), 'std: %.5f' % np.std(corr_w_list))
                # print('Weighted Kendall Rank correlation for BC -- mean: %.5f'% np.mean(corr_w_list_BC), 'std: %.5f' % np.std(corr_w_list_BC))
                # print('Weighted Kendall Rank correlation for lomge --  mean: %.5f'% np.mean(corr_w_list_logme), 'std: %.5f' % np.std(corr_w_list_logme))
                print()
                
                # Write data to CSV
                writer.writerow([dataset, degree, "%.5f" % np.mean(corr_w_list), "%.5f" % np.std(corr_w_list)])


    # for dataset in datasets:
    #     print('dataset:', dataset)
    #     print()
    #     for degree in range(2,5):
    #         print('degree:', degree)
    #         X = []
    #         for model_name in model_names:
    #             args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, "--degree", str(degree)])
    #             print(model_name)
    #             metric = main(args)


    # model_names = ["clip", "albef", "blip","blip2"]
    # datasets = ['COCO']    
    # for dataset in datasets:
    #     # print()
    #     # print('dataset:', dataset)
    #     for model_name in model_names:
    #         # print(model_name)
    #         for degree in range(1,11):
    #             #print('degree:', degree)
    #             p_val_list = []
    #             for _ in range(1):
    #                 args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, "--degree", str(degree), 
    #                                             "--kernel_type",'poly'])
    #                 metric = main(args)
    #                 #print('Metric:',metric)
    #                 p_val_list.append(metric)
    #             print(model_name, degree, np.mean(p_val_list))


    # model_names = ["clip", "albef", "blip","blip2"]
    # datasets = ['COCO']   
    # for dataset in datasets:
    #     for model_name in model_names:
    #         best_p = 0
    #         best_kernel = ''
    #         args = parser.parse_args(["--model_name", model_name, "--dataset", dataset])
    #         dic_kernel = main(args)
    #         #print(dic_kernel)
    #         for kernel_type in dic_kernel:
    #             cnt_p = dic_kernel[kernel_type]
    #             if cnt_p > best_p:
    #                 best_p = cnt_p
    #                 best_kernel = kernel_type
    #         print('best kernel for', model_name, 'on', dataset, best_kernel, best_p)
