import os
import json
import pandas as pd
import numpy as np
import argparse
import warnings
import sys
import math
from PIL import Image
import torch
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess, load_model
from lavis.models.base_model import BaseModel
import time
import random
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict
from utils_retrieval import *


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


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

    model.eval()

    embed_dim = 768
    if model_name == 'clip':
        embed_dim = 512
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
                        # Multimodal embedding 
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
    karpathy_json_file = "/data/mdai/lavis/coco_default/annotations/coco_karpathy_test.json"
    images_path = "/data/mdai/lavis/coco_default/images"
    n_neg = args.n_neg  # Number of negative captions to select for each image
    n_pos = args.n_pos

    # Load dataset
    with open(karpathy_json_file, "r") as f:
        dataset = json.load(f)

    num_images = len(dataset)
    random_indexes = random.sample(range(num_images), args.n_queries)

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

    model_name = args.model_name
    features_pos = lavis_extract_text_feature(model_name, image_caption_dict, n_pos, args.embedding_type)
    features_neg = lavis_extract_text_feature(model_name, negative_caption_dict, n_neg, args.embedding_type)

    mmd_scores = []

    # Main loop
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

        if args.kernel_type == 'poly':
            mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel='poly',degree=args.degree)
        else:
            mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel=args.kernel_type)
        mmd_scores.append(mmd_value)

        # p_val = permutation_test_mmd(positive_embeddings, negative_embeddings, args.kernel_type, degree=args.degree, num_permutations=1000)
        # if p_val < 0.05:
        #     cnt_p_value += 1  

        torch.cuda.empty_cache()

    metric = np.mean(mmd_scores)

    return metric 


if __name__ == "__main__":
    model_names = ["clip", "albef", "blip","blip2"]  
    dataset = 'COCO'
    
    for degree in range(1,11):
        for model_name in model_names:
            args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, "--degree", str(degree), 
                                      "--kernel_type",'poly'])
            metric = main(args)
            print('degree:', degree, 'model:', model_name, 'score:', metric)

