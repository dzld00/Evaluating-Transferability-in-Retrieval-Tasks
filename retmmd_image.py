import os
import sys
import numpy as np
import torch.nn as nn
import scipy.io
import argparse
import csv
import pandas as pd
import warnings
from collections import defaultdict
from utils_retrieval import *


# print(timm.list_models())

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SOP", choices=['SOP','CUB200', 'CARS196', 'Imagenet'])
parser.add_argument("--model_name", type=str, default="vit_small_patch16_224")
parser.add_argument("--n_queries", default=100, type=int)
parser.add_argument("--n_pos", default=5, type=int)
parser.add_argument("--n_neg", default=50, type=int)
parser.add_argument("--n_components", default=50, type=int)
parser.add_argument("--kernel_type", type=str, default='rbf')
parser.add_argument("--degree", default=1, type=int, help='Degree for polynomial kernel')
parser.add_argument("--pretrained", action="store_false")
parser.add_argument("--finetuned", action="store_true")
parser.add_argument("--finetune_method", type=str, default='contrastive', choices=['cls','contrastive'])

args = parser.parse_args()


def main(args):
    #print(args)

    n_neg = args.n_neg  
    n_pos = args.n_pos
    N = n_pos

    if args.dataset == 'SOP':
        image_folder = '/ebs/mdai/SOD/Stanford_Online_Products'
        data_file = os.path.join(image_folder, 'Ebay_test.txt')
        data = pd.read_csv(data_file, delimiter=' ', skiprows=1, names=['image_id', 'class_id', 'super_class_id', 'path'])
        
        # Count the number of images per class
        class_counts = data['class_id'].value_counts()
        classes_with_ = class_counts[class_counts > N].index
        
        # Filter the data to include only classes with more than N images
        filtered_data = data[data['class_id'].isin(classes_with_)]
        image_id_list = filtered_data['image_id'].tolist()

    if args.dataset == 'CUB200':
        cub200_folder = '/ebs/mdai/CUB200/CUB_200_2011'
        image_folder = '/ebs/mdai/CUB200/CUB_200_2011/images'
        data_file = os.path.join(cub200_folder, 'image_class_labels.txt')
        data = pd.read_csv(data_file, delimiter=' ', names=['image_id', 'class_id'])
        train_test_split = pd.read_csv(os.path.join(cub200_folder, 'train_test_split.txt'), delim_whitespace=True, header=None, names=['image_id', 'is_training_image'])
        image_class_labels = pd.read_csv(os.path.join(cub200_folder, 'image_class_labels.txt'), delim_whitespace=True, header=None, names=['image_id', 'class_id'])
        image_paths = pd.read_csv(os.path.join(cub200_folder, 'images.txt'), delim_whitespace=True, header=None, names=['image_id', 'path'])
        
        # Merge image_class_labels and image_paths
        data_with_class_labels = image_class_labels.merge(image_paths, on='image_id')
        
        # Merge the train-test split information with the data_with_class_labels dataframe
        data_with_split_info = data_with_class_labels.merge(train_test_split, on='image_id')
        
        # Filter the data to include only test images
        test_data = data_with_split_info[data_with_split_info['is_training_image'] == 0]
        
        # Count the number of test images per class
        test_class_counts = test_data['class_id'].value_counts()
        test_classes_with_N = test_class_counts[test_class_counts > N].index
        filtered_data = test_data[test_data['class_id'].isin(test_classes_with_N)]
        
        # Get a list of test image paths for the filtered test data
        image_id_list = filtered_data['path'].tolist()

    if args.dataset == 'CARS196':
        image_folder = '/ebs/mdai/CARS196'
        annotations_file = '/ebs/mdai/CARS196/cars_test_annos_withlabels.mat'
        annotations = scipy.io.loadmat(annotations_file)
        annotations_data = annotations['annotations']
        
        # Convert the annotations to a DataFrame
        annotations_df = pd.DataFrame([(anno[0][0][0], anno[1][0][0], anno[2][0][0], anno[3][0][0], anno[4][0][0], os.path.join('cars_test', anno[5][0])) for anno in annotations_data[0]], columns=['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class_id', 'path'])
        
        # Filter the data to include only test images
        test_data = annotations_df
        
        # Count the number of test images per class
        test_class_counts = test_data['class_id'].value_counts()
        
        # Filter classes with more than N test images
        test_classes_with_more_than_N = test_class_counts[test_class_counts > N].index
        
        # Filter the test data to include only classes with more than N test images
        filtered_data = test_data[test_data['class_id'].isin(test_classes_with_more_than_N)]
        
        # Get a list of test image paths for the filtered test data
        image_id_list = filtered_data['path'].tolist()

    if args.dataset == 'Imagenet':
        image_folder = '/ebs/mdai/imagenet/val'
        # List all class folders in the validation directory
        class_folders = [os.path.join(image_folder, folder) for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

        filtered_data_list = []

        # Iterate through each class folder
        for class_folder in class_folders:
            class_id = os.path.basename(class_folder)
            image_paths = [os.path.join(class_folder, image) for image in os.listdir(class_folder)]
            
            if len(image_paths) > N:
                filtered_data_list.extend([(class_id, image_path) for image_path in image_paths])

        # Convert the filtered data to a DataFrame
        filtered_data = pd.DataFrame(filtered_data_list, columns=['class_id', 'path'])

        # Get a list of image paths for the filtered data
        image_id_list = filtered_data['path'].tolist()


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    model = timm.create_model(args.model_name, pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    if args.finetuned:
        model_load_path = '/ebs/mdai/{}/{}_contrastive.pth'.format(args.dataset,args.model_name)
        loaded_state_dict = torch.load(model_load_path, map_location="cuda:0")
        model.load_state_dict(loaded_state_dict, strict=True)
        print('Loaded finetuned model', model_load_path)
    model = model.to(device)
    model.eval()

    mmd_scores = []
    w_scores = []

    # Main loop
    for idx in range(args.n_queries): 
        query_image = filtered_data.sample().iloc[0]
        query_image_id = query_image['path']
        query_class_id = query_image['class_id']
        num_images_in_class = filtered_data[filtered_data["class_id"] == query_class_id].shape[0]

        # Get positive samples: images from the same class as the query image
        positive_examples = filtered_data[(filtered_data["class_id"] == query_class_id) & (filtered_data["path"] != query_image_id)]
        positive_example_paths = positive_examples.sample(n_pos)['path'].tolist()

        # Get negative samples: images from different classes
        negative_examples = filtered_data[filtered_data["class_id"] != query_class_id]
        negative_example_paths = negative_examples.sample(n_neg)['path'].tolist()

        # Add the image_folder path to the example paths and query path
        positive_example_paths = [os.path.join(image_folder, path) for path in positive_example_paths]
        negative_example_paths = [os.path.join(image_folder, path) for path in negative_example_paths]
        query_path = os.path.join(image_folder, query_image['path'])

        # Get embeddings
        positive_image_embddings = image_embeddings(positive_example_paths, model, transform)
        negative_image_embddings = image_embeddings(negative_example_paths, model, transform)
        query_embedding = image_embeddings([query_path], model, transform)

        # Unconditioned vit embeddings
        if 'vit' or 'deit3' in args.model_name:
            positive_embeddings = positive_image_embddings[:, 0, :].detach().cpu().numpy()
            negative_embeddings = negative_image_embddings[:, 0, :].detach().cpu().numpy()
            query_embedding = query_embedding[:, 0, :].detach().cpu().numpy()
            if args.finetuned and args.finetune_method == 'cls':
                positive_embeddings = positive_image_embddings.detach().cpu().numpy()
                negative_embeddings = negative_image_embddings.detach().cpu().numpy()
                query_embedding = query_embedding.detach().cpu().numpy()                

        if positive_embeddings.shape[0] != args.n_pos:
            continue 

        # if PCA
        embeddings = np.vstack((query_embedding, positive_embeddings, negative_embeddings)) 
        sc = StandardScaler()
        standardized_embeddings = sc.fit_transform(embeddings)
        standardized_positive_embeddings = standardized_embeddings[1:n_pos+1,:]
        standardized_negative_embeddings = standardized_embeddings[n_pos+1:,:]

        pca = PCA()
        pca.fit(standardized_embeddings)
        explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
        n_components = (explained_variance_ratio < 0.99).sum() + 1

        transform_ = PCA(n_components=n_components)
        transform_pos_neg = transform_.fit(standardized_embeddings)
        transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
        query_embedding = transformed_embeddings[0,:].reshape(1, -1)
        positive_embeddings = transformed_embeddings[1:n_pos+1,:]
        negative_embeddings = transformed_embeddings[n_pos+1:,:]

        if args.kernel_type == 'poly':
            mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel='poly',degree=args.degree)
        else:
            mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel=args.kernel_type)

        weight = 1 / num_images_in_class
        mmd_scores.append(mmd_value*weight)
        #mmd_scores.append(mmd_value) 

        torch.cuda.empty_cache()

    metric = np.mean(mmd_scores) 

    return metric


if __name__ == "__main__":
    model_names = ["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_small_patch16_224_dino","vit_small_patch16_224_in21k","vit_base_patch16_224"]  
    kernel_types = ['linear','cosine','poly','rbf']
    dataset = 'CARS196' 
    
    for kernel_type in kernel_types:
        for model_name in model_names:
            args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, 
                                        "--kernel_type",kernel_type)
            metric = main(args)
            print('kernel:', kernel_type, 'model:', model_name, 'score:', metric)
