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
from statistics import mean,stdev
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.covariance import LedoitWolf, MinCovDet
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


warnings.filterwarnings("ignore")

# random.seed(1234)
# torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="blip")
parser.add_argument("--n_queries", default=100, type=int)
parser.add_argument("--n_pos", default=5, type=int)
parser.add_argument("--n_neg", default=20, type=int)
parser.add_argument("--n_combo", default=5, type=int)
parser.add_argument("--n_samples_per_group", default=3, type=int)
parser.add_argument("--n_components", default=3, type=int)
parser.add_argument("--dim_reduce", type=str, default="pca", choices=['pca','kernel_pca'])
parser.add_argument("--embedding_type", type=str, default="multimodal", choices=['multimodal','text'])
parser.add_argument("--spd_metric", type=str, default="multimodal", choices=['multimodal','text'])

args = parser.parse_args()

#args.dim_reduce = 'kernel_pca'

# Set the paths to the Karpathy JSON file and images
karpathy_json_file = "/data/mdai/lavis/coco_default/annotations/coco_karpathy_test.json"
images_path = "/data/mdai/lavis/coco_default/images"
n_neg = args.n_neg  # Number of negative captions to select for each image
n_pos = args.n_pos
n_samples = args.n_samples_per_group
n_groups_pos = math.ceil(n_pos / n_samples) #n_pos // n_samples 
n_groups_neg = math.ceil(n_neg / n_samples) #n_neg // n_samples
n_combo = args.n_combo
#n_combo = max(args.n_combo, 5)

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



model_name = args.model_name
if args.n_combo > 1:
    features_pos = lavis_extract_text_feature(model_name, image_caption_dict, math.ceil(n_pos * n_combo / n_pos), args.embedding_type)
    features_neg = lavis_extract_text_feature(model_name, negative_caption_dict, math.ceil(n_neg  * n_combo / n_pos), args.embedding_type)
else:
    features_pos = lavis_extract_text_feature(model_name, image_caption_dict, 5, args.embedding_type)
    features_neg = lavis_extract_text_feature(model_name, negative_caption_dict, n_neg, args.embedding_type)

print(features_pos.size())
print(features_neg.size())


def log_dist(p1, p2): #suppress warning
    original_stdout = sys.stdout
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull

            A1 = logm(p1)
            A2 = logm(p2)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sys.stdout = original_stdout 
    return norm(A1-A2)


# distance between unit-determinant SPDMs
def cov_dist_unit(punit1, punit2):
    original_stdout = sys.stdout
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull

            p12 = np.sqrt(reduce(np.dot, [inv(punit1), np.square(punit2), inv(punit1)]))   
            p12 = np.nan_to_num(p12)
            A12 = logm(p12)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sys.stdout = original_stdout 
    return norm(A12)


# distance between any SPDMs
def cov_dist(p1, p2):
    n = p1.shape[0]
    p1 = p1 + np.eye(n) * 1e-6
    p2 = p2 + np.eye(n) * 1e-6
    
    pdet1 = det(p1)
    pdet2 = det(p2)
    
    #print(pdet1, pdet2)
    
    punit1 = p1 / (np.power(pdet1, (1/n)))
    punit2 = p2 / (np.power(pdet2, (1/n)))
    
    c = 1/n
    g1 = np.square(cov_dist_unit(punit1, punit2)) 
    g2 = np.multiply(c, np.square(np.log(pdet1) - np.log(pdet2)))
    square_d = g1 + g2
    dist = np.sqrt(square_d)
    
    #return g1
    return dist


def standardize_and_PCA(embeddings, n_components, mode='kernelPCA'):
    sc = StandardScaler()
    standardized_embeddings = sc.fit_transform(embeddings)

    if mode == 'kernelPCA':
        pca_ = KernelPCA(n_components=n_components, kernel='rbf')
    if mode == 'PCA':
        pca_ = PCA(n_components=n_components)

    transformed_embeddings = pca_.fit_transform(standardized_embeddings)
    return transformed_embeddings


def MCD_cov(embeddings):
    mcd = MinCovDet()
    # Fit the MCD estimator to the data
    mcd.fit(embeddings)

    cov = mcd.covariance_

    return cov


def MCD_mean(embeddings):
    mcd = MinCovDet()
    # Fit the MCD estimator to the data
    mcd.fit(embeddings)

    mean = mcd.location_

    return mean


def weighted_difference(weight, mean_diff, cov_diff):
    return weight * mean_diff + (1 - weight) * cov_diff


def find_optimal_weight(embeddings1, embeddings2):
    # Prepare the data for cross-validation
    data = np.concatenate([embeddings1, embeddings2])
    labels = np.concatenate([np.zeros(embeddings1.shape[0]), np.ones(embeddings2.shape[0])])

    # Set up the cross-validation using KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    #kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Initialize variables to store the optimal weight and the minimum error
    optimal_weight = 0
    min_error = float("inf")

    # Loop through the possible weights (you can use a finer grid if needed)
    for weight in np.arange(0, 1.1, 0.1):
        cv_errors = []

        # Perform cross-validation
        for train_idx, val_idx in kf.split(data, labels):
            train_data = data[train_idx]
            train_labels = labels[train_idx]
            val_data = data[val_idx]
            val_labels = labels[val_idx]

            # print(train_data[train_labels == 0].shape)
            # print(train_data[train_labels == 1].shape)

            try:
                # Calculate the mean and covariance differences
                mean_diff = np.linalg.norm(train_data[train_labels == 0].mean(axis=0) - train_data[train_labels == 1].mean(axis=0))
                #mean_diff = cosine(train_data[train_labels == 0].mean(axis=0), train_data[train_labels == 1].mean(axis=0))
                #cov_diff = log_dist(LedoitWolf().fit(train_data[train_labels == 0]).covariance_, LedoitWolf().fit(train_data[train_labels == 1]).covariance_)
                cov_diff = cov_dist(LedoitWolf().fit(train_data[train_labels == 0]).covariance_, LedoitWolf().fit(train_data[train_labels == 1]).covariance_)

                # Calculate the weighted differences for the validation set
                val_mean_diff = np.linalg.norm(val_data[val_labels == 0].mean(axis=0) - val_data[val_labels == 1].mean(axis=0))
                #val_mean_diff = cosine(val_data[val_labels == 0].mean(axis=0), val_data[val_labels == 1].mean(axis=0))
                #val_cov_diff = log_dist(LedoitWolf().fit(val_data[val_labels == 0]).covariance_, LedoitWolf().fit(val_data[val_labels == 1]).covariance_)
                val_cov_diff = cov_dist(LedoitWolf().fit(val_data[val_labels == 0]).covariance_, LedoitWolf().fit(val_data[val_labels == 1]).covariance_)
                val_weighted_diff = weighted_difference(weight, val_mean_diff, val_cov_diff)
            except:
                continue

            # Calculate the true weighted difference for the training set
            true_weighted_diff = weighted_difference(weight, mean_diff, cov_diff)

            # Calculate the mean squared error
            mse = mean_squared_error([true_weighted_diff], [val_weighted_diff])
            cv_errors.append(mse)

        # Calculate the average error across all folds
        avg_error = np.mean(cv_errors)

        # Update the optimal weight and minimum error if necessary
        if avg_error < min_error:
            min_error = avg_error
            optimal_weight = weight

    return optimal_weight


def BC_dist(mu1, cov1, mu2, cov2):
    mean_cov = 0.5 * (cov1 + cov2)
    d = ((mu1 - mu2).T @ np.linalg.inv(mean_cov) @ (mu1 - mu2) / 8.).reshape(-1)[0]\
            + np.log(np.linalg.det(mean_cov)/(np.linalg.det(cov1) * np.linalg.det(cov2))**0.5) / 2.
    return -np.exp(-d)


permanova_cov_list, permanova_mean_list = [], []
anosim_cov_list, anosim_mean_list = [], []
cov_dist_list, mean_dist_list = [], []
overall_cov_list, overall_mean_list = [], []
metric_list, weight_list = [], []
BC_list, logme_list = [], []
cnt_p05_mean, cnt_p05_cov = 0, 0 
cnt_0, cnt_1, cnt_middle = 0, 0, 0


for idx in range(len(image_caption_dict)): 
    # if idx % 50 == 0:
    #     print(idx)

# cnt_queries, cnt_pass = 0, 0
# for image_name in image_caption_dict: 
#     # cnt_pass += 1
#     # print(cnt_pass)
#     if cnt_queries == args.n_queries:
#         break

    # positive_embeddings = lavis_extract_text_feature_per_image(args.model_name, image_caption_dict, image_name, n_pos, args.embedding_type).detach().cpu().numpy()
    # negative_embeddings = lavis_extract_text_feature_per_image(args.model_name, negative_caption_dict, image_name, n_neg, args.embedding_type).detach().cpu().numpy()

    positive_embeddings = features_pos[idx,:,:].squeeze(0).detach().cpu().numpy()
    negative_embeddings = features_neg[idx,:,:].squeeze(0).detach().cpu().numpy()

    # Combine positive and negative embeddings
    embeddings = np.vstack((positive_embeddings, negative_embeddings))

    try:
        # Apply t-SNE
        tsne = TSNE(n_components=2)#random_state=42
        embeddings_2d = tsne.fit_transform(embeddings)
    except:
        continue

    # Create labels for the embeddings
    labels = ['Positive'] * len(positive_embeddings) + ['Negative'] * len(negative_embeddings)

    # Plot the t-SNE result
    plt.figure(figsize=(10, 6))
    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label)

    plt.legend()
    plt.title('t-SNE plot of positive and negative embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Save the plot as an image
    plt.savefig('figs/tsne_plot-{}.png'.format(model_name.split('/')[-1]))


    ### kernel pca
    # positive_sc = StandardScaler()
    # standardized_positive_embeddings = positive_sc.fit_transform(positive_embeddings)
    # negative_sc = StandardScaler()
    # standardized_negative_embeddings = negative_sc.fit_transform(negative_embeddings)

    sc = StandardScaler()
    standardized_embeddings = sc.fit_transform(embeddings)
    standardized_positive_embeddings = standardized_embeddings[:n_pos,:]
    standardized_negative_embeddings = standardized_embeddings[n_pos:,:]

    
    n_components = args.n_components
    if args.dim_reduce == 'pca':
        # positive_pca = PCA(n_components=n_components)
        # transformed_positive_embeddings = positive_pca.fit_transform(standardized_positive_embeddings)
        # negative_pca = PCA(n_components=n_components)
        # transformed_negative_embeddings = negative_pca.fit_transform(standardized_negative_embeddings)

        transform_ = PCA(n_components=n_components)
        standardized_embeddings = np.concatenate((standardized_positive_embeddings, standardized_negative_embeddings),axis=0)
        transform_pos_neg = transform_.fit(standardized_embeddings)
        transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
        transformed_positive_embeddings = transformed_embeddings[:n_pos,:]
        transformed_negative_embeddings = transformed_embeddings[n_pos:,:]

    elif args.dim_reduce == 'kernel_pca':
        # positive_pca = KernelPCA(n_components=n_components, kernel='rbf')#, gamma=gamma_pos)
        # transformed_positive_embeddings = positive_pca.fit_transform(standardized_positive_embeddings)
        # negative_pca = KernelPCA(n_components=n_components, kernel='rbf')#, gamma=gamma_neg)
        # transformed_negative_embeddings = negative_pca.fit_transform(standardized_negative_embeddings)

        transform_ = KernelPCA(n_components=n_components, kernel='rbf')     
        standardized_embeddings = np.concatenate((standardized_positive_embeddings, standardized_negative_embeddings),axis=0)
        transform_pos_neg = transform_.fit(standardized_embeddings)
        transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
        transformed_positive_embeddings = transformed_embeddings[:n_pos,:]
        transformed_negative_embeddings = transformed_embeddings[n_pos:,:]

    '''
    #try:
    if args.n_combo > 1:
        groups_neg= np.arange(n_groups_neg)
        groups_pos = np.arange(n_groups_pos)
        negative_embeddings_for_each_class = {i:transformed_negative_embeddings[i * n_samples: (i+1)* n_samples] for i in groups_neg}
        positive_embeddings_for_each_class = {i:transformed_positive_embeddings[i * n_samples: (i+1)* n_samples] for i in groups_pos}
        
        groups_neg = np.arange(math.ceil(n_combo * n_neg / n_pos))
        groups_pos = np.arange(n_combo)
        # Create a list of indices for the embeddings
        embedding_indices_pos = np.arange(n_pos)
        all_combinations_pos = list(combinations(embedding_indices_pos, n_samples))
        selected_combinations_pos = random.sample(all_combinations_pos, n_combo)
        embedding_indices_neg = np.arange(n_neg)
        all_combinations_neg = list(combinations(embedding_indices_neg, n_samples))
        selected_combinations_neg = random.sample(all_combinations_neg, math.ceil(n_combo * n_neg / n_pos))
        #print(selected_combinations)
        # Create a dictionary with each group index as the key and the corresponding embeddings as the value
        negative_embeddings_for_each_class = {i: transformed_negative_embeddings[list(group),:] for i, group in enumerate(selected_combinations_neg)}   
        positive_embeddings_for_each_class = {i: transformed_positive_embeddings[list(group),:] for i, group in enumerate(selected_combinations_pos)} 
    else:
        groups_neg = np.arange(math.ceil(n_neg / n_samples))
        groups_pos = np.arange(math.ceil(n_pos / n_samples))
        negative_embeddings_for_each_class = {i:transformed_negative_embeddings[i * n_samples: (i+1)* n_samples] for i in groups_neg}
        positive_embeddings_for_each_class = {i:transformed_positive_embeddings[i * n_samples: (i+1)* n_samples] for i in groups_pos} 


    positive_cov_for_each_class = {i:LedoitWolf().fit(positive_embeddings_for_each_class[i]).covariance_ for i in groups_pos}
    negative_cov_for_each_class = {i:LedoitWolf().fit(negative_embeddings_for_each_class[i]).covariance_ for i in groups_neg}
    #positive_cov_for_each_class = {i:MCD_cov(positive_embeddings_for_each_class[i]) for i in groups_pos}
    #negative_cov_for_each_class = {i:MCD_cov(negative_embeddings_for_each_class[i]) for i in groups_neg}

    # try:
    #     ### Overall mean and cov
    #     positive_cov = LedoitWolf().fit(standardized_positive_embeddings).covariance_
    #     negative_cov = LedoitWolf().fit(standardized_negative_embeddings).covariance_
    #     overall_cov_dist = log_dist(positive_cov , negative_cov)
    #     #overall_cov_dist = cov_dist(positive_cov , negative_cov)
    #     cov_dist_list.append(overall_cov_dist)
    #     overall_mean_diff = np.linalg.norm(np.mean(standardized_positive_embeddings, axis=0) - np.mean(standardized_negative_embeddings, axis=0))
    #     mean_dist_list.append(overall_mean_diff)
    # except:
    #     continue

    try:
        ### Overall mean and cov
        positive_cov = LedoitWolf().fit(transformed_positive_embeddings).covariance_
        negative_cov = LedoitWolf().fit(transformed_negative_embeddings).covariance_
        #overall_cov_dist = log_dist(positive_cov , negative_cov)
        overall_cov_dist = cov_dist(positive_cov , negative_cov)
        cov_dist_list.append(overall_cov_dist)
        overall_mean_diff = np.linalg.norm(np.mean(transformed_positive_embeddings, axis=0) - np.mean(transformed_negative_embeddings, axis=0))
        #overall_mean_diff = cosine(np.mean(transformed_positive_embeddings, axis=0), np.mean(transformed_negative_embeddings, axis=0))
        #print(overall_mean_diff)
        mean_dist_list.append(overall_mean_diff)
    except:
        continue

    cov_list = []#
    #cov_list.append(positive_cov)
    for key in positive_cov_for_each_class:
        cov_list.append(positive_cov_for_each_class[key])
    for key in negative_cov_for_each_class:
        cov_list.append(negative_cov_for_each_class[key])

    dist_cov = np.zeros((len(cov_list),len(cov_list)))

    try:
        for i in range(len(cov_list)):
            for j in range(i+1,len(cov_list)):
                    dist_cov[i,j] = log_dist(cov_list[i], cov_list[j])
                    #dist_cov[i,j] = cov_dist(cov_list[i], cov_list[j])
                    dist_cov[j,i] = dist_cov[i,j]
    except:
        continue

    plt.figure()
    plt.imshow(dist_cov, cmap='summer')
    plt.colorbar()
    plt.savefig('figs/cov_dist-{}-{}-ncomp_{}.png'.format(model_name.split('/')[-1],args.dim_reduce,args.n_components))


    positive_mean_for_each_class = {i:np.mean(positive_embeddings_for_each_class[i], axis=0).reshape(1, -1).T for i in groups_pos}
    negative_mean_for_each_class = {i:np.mean(negative_embeddings_for_each_class[i], axis=0).reshape(1, -1).T for i in groups_neg}
    #positive_mean_for_each_class = {i:MCD_mean(positive_embeddings_for_each_class[i]) for i in groups_pos}
    #negative_mean_for_each_class = {i:MCD_mean(negative_embeddings_for_each_class[i]) for i in groups_neg}

    mean_list = []#
    #cov_list.append(positive_cov)
    for key in positive_mean_for_each_class:
        mean_list.append(positive_mean_for_each_class[key])
    for key in negative_mean_for_each_class:
        mean_list.append(negative_mean_for_each_class[key])

    dist_mean = np.zeros((len(mean_list),len(mean_list)))
    for i in range(len(mean_list)):
        for j in range(i+1,len(mean_list)):
            dist_mean[i,j] = np.linalg.norm(mean_list[i] - mean_list[j])
            #print(mean_list[i].shape)
            #dist_mean[i,j] = cosine(np.squeeze(mean_list[i],axis=-1), np.squeeze(mean_list[j],axis=-1))
            dist_mean[j,i] = dist_mean[i,j]

    plt.figure()
    plt.imshow(dist_mean, cmap='hot')
    plt.colorbar()
    plt.savefig('figs/mean_dist-{}.png'.format(model_name.split('/')[-1]))

    # print(' ')
    # print("Trace of positive and negative cov:", np.trace(positive_cov), np.trace(negative_cov))
    # print('Determinant of positive and negative cov:', np.log(det(positive_cov + np.eye(positive_cov.shape[0]) * 1e-6)), np.log(det(negative_cov + np.eye(positive_cov.shape[0]) * 1e-6)))
    # print("Overall cov dist:", overall_cov_dist)
    # print("Overall mean dist:", overall_mean_diff)

    ### Evaluation
    if args.n_combo == 1:
        labels = [0] * math.ceil(n_pos / n_samples) + [1] * math.ceil(n_neg / n_samples)
    else:
        labels = [0] * n_combo + [1] * math.ceil(n_combo * n_neg / n_pos)

    
    cov_p05, mean_p05 = False, False
    #print(' ')
    #print(query_path)
    permanova_results = skbio.stats.distance.permanova(DistanceMatrix(dist_cov), labels) #dist_cov
    # if permanova_results['test statistic']>500:# or permanova_results['p-value'] > 0.05:
    #     continue
    #print("PERMANOVA cov: Test Statistic = {}, p-value = {}, sample size = {}".format(permanova_results['test statistic'], permanova_results['p-value'], permanova_results['sample size']))
    # Perform the ANOSIM test
    anosim_results = skbio.stats.distance.anosim(skbio.DistanceMatrix(np.array(dist_cov)), labels) #dist_cov
    # Print the ANOSIM results
    #print("ANOSIM cov: Test Statistic = {}, p-value = {}, sample size = {}".format(anosim_results['test statistic'], anosim_results['p-value'], anosim_results['sample size']))
    permanova_cov_list.append(permanova_results['test statistic'])
    anosim_cov_list.append(anosim_results['test statistic'])
    #print("Overall cov dist:", overall_cov_dist)
    
    cur_cov_dist = 0
    if permanova_results['p-value'] < 0.05 or anosim_results['p-value'] < 0.05:
        cur_cov_dist = overall_cov_dist
        cnt_p05_cov += 1
        cov_p05 = True
    overall_cov_list.append(cur_cov_dist)

    permanova_results = skbio.stats.distance.permanova(DistanceMatrix(dist_mean), labels) #dist_cov
    #print("PERMANOVA mean: Test Statistic = {}, p-value = {}, sample size = {}".format(permanova_results['test statistic'], permanova_results['p-value'], permanova_results['sample size']))
    # Perform the ANOSIM test
    anosim_results = skbio.stats.distance.anosim(skbio.DistanceMatrix(np.array(dist_mean)), labels) #dist_cov
    # Print the ANOSIM results
    #print("ANOSIM mean: Test Statistic = {}, p-value = {}, sample size = {}".format(anosim_results['test statistic'], anosim_results['p-value'], anosim_results['sample size']))
    permanova_mean_list.append(permanova_results['test statistic'])
    anosim_mean_list.append(anosim_results['test statistic'])
    #print("Overall mean dist:", overall_mean_diff)
    cur_mean_dist = 0
    if permanova_results['p-value'] < 0.05 or anosim_results['p-value'] < 0.05:
        cur_mean_dist = overall_mean_diff
        cnt_p05_mean += 1
        mean_p05 = True
    overall_mean_list.append(cur_mean_dist)

    optimal_weight = 0
    if cov_p05 == True and mean_p05 == True:
        optimal_weight = find_optimal_weight(transformed_positive_embeddings, transformed_negative_embeddings)
        cur_metric = optimal_weight * cur_mean_dist + (1-optimal_weight) * cur_cov_dist
    elif cov_p05 == False and mean_p05 == True:
        cur_metric = cur_mean_dist
        optimal_weight = 1
    elif cov_p05 == True and mean_p05 == False:
        cur_metric = cur_cov_dist
        optimal_weight = 0
    else:
        cur_metric = 0
        optimal_weight = -1

    if optimal_weight == 1:
        cnt_1 += 1
    elif optimal_weight == 0:
        cnt_0 += 1
    elif optimal_weight > 0 and optimal_weight < 1:
        cnt_middle += 1

    # Assign weights inversely proportional to distances
    weight_mean = 1 / (cur_mean_dist + 1e-10)
    weight_cov = 1 / (cur_cov_dist + 1e-10)
    # Normalize the weights so they sum to 1
    total_weight = weight_mean + weight_cov
    weight_mean /= total_weight
    weight_cov /= total_weight
    # Compute combined distance
    cur_metric = weight_mean * cur_mean_dist + weight_cov * cur_cov_dist
    '''


    positive_cov = LedoitWolf().fit(transformed_positive_embeddings).covariance_
    negative_cov = LedoitWolf().fit(transformed_negative_embeddings).covariance_
    #cur_cov_dist = log_dist(positive_cov, negative_cov)
    cur_cov_dist = cov_dist(positive_cov, negative_cov)
    cur_mean_dist = np.linalg.norm(np.mean(transformed_positive_embeddings, axis=0) - np.mean(transformed_negative_embeddings, axis=0))

    # Assign weights inversely proportional to distances
    # weight_mean = 1 / (cur_mean_dist + 1e-10)
    # weight_cov = 1 / (cur_cov_dist + 1e-10)
    # # Normalize the weights so they sum to 1
    # total_weight = weight_mean + weight_cov
    # weight_mean /= total_weight
    # weight_cov /= total_weight
    # Compute combined distance
    # cur_metric = weight_mean * cur_mean_dist + weight_cov * cur_cov_dist

    cur_metric = cur_mean_dist + cur_cov_dist
    
    metric_list.append(cur_metric)
    #weight_list.append(optimal_weight)
    overall_mean_list.append(cur_mean_dist)
    overall_cov_list.append(cur_cov_dist)
    
    

    # optimal_weight = find_optimal_weight(transformed_positive_embeddings, transformed_negative_embeddings)
    # positive_cov = LedoitWolf().fit(transformed_positive_embeddings).covariance_
    # negative_cov = LedoitWolf().fit(transformed_negative_embeddings).covariance_
    # #cur_cov_dist = log_dist(positive_cov, negative_cov)
    # cur_cov_dist = cov_dist(positive_cov, negative_cov)
    # cur_mean_dist = np.linalg.norm(np.mean(transformed_positive_embeddings, axis=0) - np.mean(transformed_negative_embeddings, axis=0))
    # cur_metric = optimal_weight * cur_mean_dist + (1-optimal_weight) * cur_cov_dist

    ### BC
    cur_BC = BC_dist(np.mean(transformed_positive_embeddings, axis=0), LedoitWolf().fit(transformed_positive_embeddings).covariance_ , np.mean(transformed_negative_embeddings, axis=0), LedoitWolf().fit(transformed_negative_embeddings).covariance_)
    #cur_BC = BC_dist(np.mean(standardized_positive_embeddings, axis=0), positive_cov , np.mean(standardized_negative_embeddings, axis=0), negative_cov)
    BC_list.append(cur_BC)

    # ###LogME
    logme = LogME(regression=False)
    np_labels =np.asarray([0]*positive_embeddings.shape[0] + [1]*negative_embeddings.shape[0])
    cur_logme = logme.fit(standardized_embeddings, np_labels)
    logme_list.append(cur_logme)


print(args.model_name)
print(args.dim_reduce)
# #print(permanova_cov_list)
# print('PERMANOVA cov:', mean(permanova_cov_list) if len(permanova_cov_list)>0 else [])
# #print(anosim_cov_list)
# print('ANOSIM cov:', mean(anosim_cov_list) if len(anosim_cov_list)>0 else [])
# #print(permanova_mean_list)
# print('PERMANOVA mean:', mean(permanova_mean_list) if len(permanova_mean_list)>0 else [])
# #print(anosim_cov_list)
# print('ANOSIM mean:', mean(anosim_mean_list) if len(anosim_mean_list)>0 else [])
# #print(cov_dist_list)
# print('Cov dist mean', mean(cov_dist_list))
# print('Mean dist mean', mean(mean_dist_list))
#print('Overall cov weighted mean', mean(overall_cov_list))
#print('Overall mean weighted mean', mean(overall_mean_list))


print('Averaged metric', np.mean(metric_list))
#print('Number of significant mean:', cnt_p05_mean, 'Number of significant cov:', cnt_p05_cov)
print('averaged overall mean:', np.mean(overall_mean_list))
print('averaged overall cov:', np.mean(overall_cov_list))
print('weight list', weight_list)
#print('Number of 1s:', cnt_1, 'Number of 0s:', cnt_0, 'Number of weighted:', cnt_middle)

# #print(BC_list)
print('Averaged BC', np.mean(BC_list))

print('Averaged LogME', np.mean(logme_list))

print(' ')






