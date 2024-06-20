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
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Get the image embedding
        with torch.no_grad():
            embedding = model(img_tensor)

        embeddings.append(embedding)

    # Stack the embeddings
    embeddings_tensor = torch.stack(embeddings).squeeze(1)

    return embeddings_tensor


def get_resnet_model(model_name):
    if model_name == 'resnet18':
        return models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def BC_dist(mu1, cov1, mu2, cov2):
    mean_cov = 0.5 * (cov1 + cov2)
    d = ((mu1 - mu2).T @ np.linalg.inv(mean_cov) @ (mu1 - mu2) / 8.).reshape(-1)[0]\
            + np.log(np.linalg.det(mean_cov)/(np.linalg.det(cov1) * np.linalg.det(cov2))**0.5) / 2.
    return -np.exp(-d)


def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)


import numpy as np

def gaussian_kernel(x, y, bandwidth):
    return np.exp(-np.linalg.norm(x - y)**2 / (bandwidth**2))


def compute_median_bandwidth(X, Y):
    combined = np.vstack([X, Y])
    pairwise_dists = np.linalg.norm(combined[:, np.newaxis] - combined[np.newaxis, :], axis=-1)
    return np.median(pairwise_dists)


def MMD(X, Y, kernel='rbf', **kwargs):
    n = X.shape[0]
    m = Y.shape[0]

    # Compute pairwise kernel values
    K_YY = pairwise_kernels(Y, Y, metric=kernel, **kwargs)
    np.fill_diagonal(K_YY, 0)
    
    K_XX = pairwise_kernels(X, X, metric=kernel, **kwargs)
    np.fill_diagonal(K_XX, 0)
    K_XY = pairwise_kernels(X, Y, metric=kernel, **kwargs)
    #return (np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * K_XY.mean()) 
    #return (np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY)/(m*n)) 
    kxx = np.sum(K_XX) / (n * (n - 1))
    kyy = np.sum(K_YY) / (m * (m - 1))
    kxy = np.sum(K_XY)/(m*n)
    return kxx + kyy - 2*kxy, kxx, kyy, kxy
        

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


def estimate_gamma(X, Y):
    # Compute pairwise distances between all pairs of samples
    distances = pairwise_distances(np.vstack((X, Y)), metric="euclidean")
    
    # Compute the median of the pairwise distances
    median_distance = np.median(distances)
    
    # Return the inverse of the median distance as gamma
    gamma = 1.0 / median_distance**2
    return gamma


def compute_MMD(query_embedding, other_embeddings, kernel='rbf', **kwargs):
    n = 1  # since query_embedding is just a single point
    m = other_embeddings.shape[0]

    # Compute pairwise kernel values
    K_QQ = pairwise_kernels(query_embedding, query_embedding, metric=kernel, **kwargs)
    K_OO = pairwise_kernels(other_embeddings, other_embeddings, metric=kernel, **kwargs)
    np.fill_diagonal(K_OO, 0)
    K_QO = pairwise_kernels(query_embedding, other_embeddings, metric=kernel, **kwargs)

    return K_QQ.mean() + (np.sum(K_OO) / (m * (m - 1))) - 2 * K_QO.mean()


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
    rho_avg = np.mean(r1 / r2)
    
    # Estimate intrinsic dimension
    D = 2 * np.log(2) / -np.log(rho_avg)
    
    return D


def mds_dimension(embeddings):
    batch_size = embeddings.shape[0] 

    D = pairwise_distances(embeddings)   
    D = D / np.amax(D)
    
    l_sorted = cmdscale(D) 
    #l_sorted = eigen_mds(D)        
    
    # Calculate k based on number of large eigenvalues
    k = next(x[0] for x in enumerate(l_sorted) if x[1] < 0.01 * l_sorted[0])  
    
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


def BC_dist(mu1, cov1, mu2, cov2):
    epsilon = 1e-5
    cov1 += epsilon * np.eye(cov1.shape[0])
    cov2 += epsilon * np.eye(cov2.shape[0])
    mean_cov = 0.5 * (cov1 + cov2)
    d = ((mu1 - mu2).T @ np.linalg.inv(mean_cov) @ (mu1 - mu2) / 8.).reshape(-1)[0]\
            + np.log(np.linalg.det(mean_cov)/(np.linalg.det(cov1) * np.linalg.det(cov2))**0.5) / 2.
    return -np.exp(-d)


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


