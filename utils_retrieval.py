import ot
import time
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torchvision.models as models
import torchvision.transforms as transforms
import timm
from sklearn.decomposition import PCA,KernelPCA
from sklearn.metrics import pairwise_distances
from statistics import mean,stdev
from sklearn.preprocessing import scale, normalize, StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import kendalltau
from scipy.stats import kendalltau, weightedtau
from sklearn.manifold import MDS


def find_correlation(train_vectors, val_vectors):
    corr, _ = kendalltau(train_vectors, val_vectors)
    return corr


def find_weighted_correlation(train_vectors, val_vectors):
    corr = weightedtau(train_vectors, val_vectors)
    return corr.correlation


def image_embeddings(image_paths, model, transform):
    embeddings = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            embedding = model(img_tensor)
        embeddings.append(embedding)

    # Stack the embeddings
    embeddings_tensor = torch.stack(embeddings).squeeze(1)

    return embeddings_tensor


def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)


def MMD(X, Y, kernel='rbf', **kwargs):
    n = X.shape[0]
    m = Y.shape[0]

    K_YY = pairwise_kernels(Y, Y, metric=kernel, **kwargs)
    np.fill_diagonal(K_YY, 0)
    K_XX = pairwise_kernels(X, X, metric=kernel, **kwargs)
    np.fill_diagonal(K_XX, 0)
    K_XY = pairwise_kernels(X, Y, metric=kernel, **kwargs)

    kxx = np.sum(K_XX) / (n * (n - 1))
    kyy = np.sum(K_YY) / (m * (m - 1))
    kxy = np.sum(K_XY)/(m*n)
    
    return kxx + kyy - 2*kxy, kxx, kyy, kxy
        

def compute_wasserstein(X, Y):
    Mxx = ot.dist(X, X)
    Myy = ot.dist(Y, Y)
    Mxy = ot.dist(X, Y)
    
    px = np.ones(X.shape[0]) / X.shape[0]
    py = np.ones(Y.shape[0]) / Y.shape[0]

    wasserstein_distance = ot.emd2(px, py, Mxy)
    
    return wasserstein_distance


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

