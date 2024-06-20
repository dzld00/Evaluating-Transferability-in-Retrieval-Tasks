import os
import pandas as pd
import numpy as np
import torch.nn as nn
import sys
import scipy.io
import argparse
import csv
import warnings
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from logME import LogME
from utils_retrieval import *
from tqdm import tqdm
import torchvision
from torchvision import transforms, datasets, models
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any


# print(timm.list_models())
# exit(0)

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Imagenet", choices=['SOP','CUB200', 'CARS196', 'Imagenet','CIFAR10','CIFAR100','SVHN','Flowers','Dogs'])
parser.add_argument("--source_dataset", type=str, default="Imagenet", choices=['SOP','CUB200', 'CARS196', 'Imagenet','CIFAR10','CIFAR100','SVHN','Flowers','Dogs'])
parser.add_argument("--model_name", type=str, default="resnet18")
parser.add_argument("--n_queries", default=100, type=int)
parser.add_argument("--n_components", default=50, type=int)
parser.add_argument("--kernel_type", type=str, default='linear')
parser.add_argument("--degree", default=1, type=int, help='Degree for polynomial kernel')
parser.add_argument("--method", type=int, default=1, choices=[1,2,3], help='1:mmd pos vs neg; 2: mmd query vs pos, query vs neg; 3: wasserstein pos vs neg')
parser.add_argument("--pretrained", action="store_false")
parser.add_argument("--finetuned", action="store_true")
parser.add_argument("--finetune_method", type=str, default='contrastive', choices=['cls','contrastive'])
parser.add_argument("--distance_type", type=str, default='mmd',choices=['mmd','wasserstein'])
parser.add_argument("--pca", action="store_false")

args = parser.parse_args()


def ratio(x1, x2, alpha=1):
    """
    Calculate transferability metric.
    
    :param x1: Performance after transfer learning.
    :param x2: Baseline performance (pre-trained model performance).
    :param alpha: Scaling factor to control the emphasis on higher baselines.
    :return: Transferability score.
    """
    # Improvement ratio
    improvement_ratio = (x1 - x2) / (1-x2)

    # Baseline scaling factor that grows faster as the baseline gets higher
    # baseline_scaling = np.exp(alpha * x2)

    # Transferability metric
    transfer_metric = improvement_ratio #* baseline_scaling
    return transfer_metric


def subsample_dataset(dataset, num_samples):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return torch.utils.data.Subset(dataset, indices)


def extract_features(model_name, model, data_loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for data, _ in tqdm(data_loader):
            data = data.to(device)
            if 'vit' in model_name or 'deit' in model_name:
                output = model(data)[:, 0, :]
                #output = model(data)[:, 0, :].mean(dim=1)
            else:
                output = model(data)
            features.append(output.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features


class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, extensions=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if extensions is not None:
            self.extensions = extensions
            self.samples = self._find_classes(self.root)
    
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        samples = []
        for target_class in sorted(class_to_idx.keys()):
            class_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self.extensions is None or self.is_valid_file(path):
                        item = path, class_to_idx[target_class]
                        samples.append(item)
        return samples


class CustomImageFolder_CARS196(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = custom_pil_loader(path)
        if sample is None:
            return self.__getitem__(index + 1)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    

def custom_pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        try:
            img = img.convert("RGB")
        except OSError as e:
            #logging.warning(f"Error loading image at {path}: {e}")
            return None
    return img


def main(args):
    #print(args)

    if args.dataset == 'SOP':
        # Load SOP, use 12 positive images per class
        image_folder = '/ebs/mdai/SOD/Stanford_Online_Products'
        data_file = os.path.join(image_folder, 'Ebay_test.txt')
        data = pd.read_csv(data_file, delimiter=' ', skiprows=1, names=['image_id', 'class_id', 'super_class_id', 'path'])
        # Count the number of images per class
        class_counts = data['class_id'].value_counts()
        # Filter classes with exactly 12 images
        #classes_with_ = class_counts[class_counts > 10].index
        classes_with_ = class_counts[class_counts > 1].index
        #print(f"Number of classes with more than {N} images: {len(classes_with_)}")
        # Filter the data to include only classes with exactly 12 images
        filtered_data = data[data['class_id'].isin(classes_with_)]
        image_id_list = filtered_data['image_id'].tolist()

    if args.dataset == 'CUB200':
        # Load CUB200, , use 30 positive images per class
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
        # Filter classes with exactly 60 test images
        #test_classes_with_30 = test_class_counts[test_class_counts > 25].index
        test_classes_with_30 = test_class_counts[test_class_counts > 1].index
        #print(f"Number of test classes with more than {N} images: {len(test_classes_with_30)}")
        # Filter the test data to include only classes with exactly 60 test images
        filtered_data = test_data[test_data['class_id'].isin(test_classes_with_30)]
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
        #N = 1 #30
        test_classes_with_more_than_N = test_class_counts[test_class_counts > 1].index
        #print(f"Number of test classes with more than {N} images: {len(test_classes_with_more_than_N)}")
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
            
            if len(image_paths) > 1:
                filtered_data_list.extend([(class_id, image_path) for image_path in image_paths])
                #print(f"Class ID: {class_id}, Number of Images: {len(image_paths)}")

        # Convert the filtered data to a DataFrame
        filtered_data = pd.DataFrame(filtered_data_list, columns=['class_id', 'path'])

        # Get a list of image paths for the filtered data
        image_id_list = filtered_data['path'].tolist()


    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    transform = transforms.Compose([
        transforms.Resize(256), # 256
        transforms.CenterCrop(224), #224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_small = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # Normalize each channel of the CIFAR-10 images
                            std=[0.2023, 0.1994, 0.2010])
    ])

    model = timm.create_model(args.model_name, pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    if args.finetuned:
        #model_load_path = '/ebs/mdai/{}/{}_unpretrained_contrastive.pth'.format(args.dataset,args.model_name)
        model_load_path = '/ebs/mdai/{}/{}_unpretrained_contrastive.pth'.format(args.source_dataset,args.model_name)
        loaded_state_dict = torch.load(model_load_path, map_location="cuda:0")
        model.load_state_dict(loaded_state_dict, strict=True)
        print('Loaded finetuned model', model_load_path)
    model.to(device)
    model.eval()

    #if args.dataset in ['Dummy']:
    if args.dataset in ['Imagenet','CUB200','CARS196','SOP']:
        dataset_embeddings = [] 
        for idx in tqdm(range(args.n_queries), desc='Processing queries'):
            query_image = filtered_data.sample().iloc[0]
            query_image_id = query_image['path']
            query_class_id = query_image['class_id']
            num_images_in_class = filtered_data[filtered_data["class_id"] == query_class_id].shape[0]
            n_pos = num_images_in_class - 1

            # Get positive samples: images from the same class as the query image
            positive_examples = filtered_data[(filtered_data["class_id"] == query_class_id) & (filtered_data["path"] != query_image_id)]
            positive_example_paths = positive_examples.sample(n_pos)['path'].tolist()
            # Add the image_folder path to the example paths and query path
            positive_example_paths = [os.path.join(image_folder, path) for path in positive_example_paths]
            # Get embeddings
            positive_image_embddings = image_embeddings(positive_example_paths, model, transform)

            ### Unconditioned vit embeddings
            if 'vit' in args.model_name or 'deit3' in args.model_name:
                positive_embeddings = positive_image_embddings[:, 0, :]
            else:
                positive_embeddings = positive_image_embddings        

            # if positive_embeddings.shape[0] != args.n_pos:
            #     continue 

            dataset_embedding = positive_embeddings.mean(dim=0) 
            dataset_embeddings.append(dataset_embedding)

            torch.cuda.empty_cache()

        all_dataset_embeddings = torch.stack(dataset_embeddings).detach().cpu().numpy()

    else:
        if args.dataset == 'CIFAR10':
            test_dataset = datasets.CIFAR10(root='/ebs/mdai', train=False, transform=transform, download=True)
        elif args.dataset == 'CIFAR100':
            test_dataset = datasets.CIFAR100(root='/ebs/mdai', train=False, transform=transform, download=True)  
        elif args.dataset == 'SVHN':
            test_dataset = datasets.SVHN(root='/ebs/mdai', split='test', transform=transform, download=True) 
        elif args.dataset == 'Dogs':
            data_dir = '/ebs/mdai/Dogs'
            test_dataset = CustomImageFolder(os.path.join(data_dir, 'test'), transform=transform) 
        elif args.dataset == 'Flowers':
            data_dir = '/ebs/mdai/Flowers' 
            test_dataset = CustomImageFolder(os.path.join(data_dir, 'test'), transform=transform) 

        # elif args.dataset == "Imagenet":
        #     data_dir = '/ebs/mdai/imagenet/val'
        #     test_dataset = CustomImageFolder(data_dir, transform=transform)  
        # elif args.dataset == 'CUB200':
        #     data_dir = '/ebs/mdai/CUB200/CUB_200_2011'
        #     test_dataset = CustomImageFolder(os.path.join(data_dir, 'test'), transform=transform) 
        # elif args.dataset == 'CARS196':
        #     data_dir = '/ebs/mdai/CARS196'
        #     test_dataset = CustomImageFolder_CARS196(os.path.join(data_dir, 'test'), transform=transform)
        # elif args.dataset == 'SOP':
        #     data_dir = '/ebs/mdai/SOD/Stanford_Online_Products'
        #     test_dataset = CustomImageFolder(os.path.join(data_dir, 'test'), transform=transform) 


        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, 
                            num_workers=4, pin_memory=True)
        subsampled_test_dataset = subsample_dataset(test_loader.dataset, min(args.n_queries*10,len(test_dataset)))
        test_loader = DataLoader(subsampled_test_dataset, batch_size=8, shuffle=False, 
                                    num_workers=4, pin_memory=True)
        
        all_dataset_embeddings = extract_features(args.model_name, model, test_loader, device)


    return all_dataset_embeddings


if __name__ == "__main__":
    #main(args)

    #Flowers_ = ratio(0.8900,0.8686)  #imagenet: 0.9804; cifar10:0.8900; cub200: 0.8228
    CUB200_ = ratio(0.4734, 0.4599)  #imagenet: 0.7228; cifar10: 0.4932; cars196: 0.4599; sop:0.4734; cifar100: 0.5096; svhn: 0.4856; dogs: 0.5463
    CARS196_ = ratio(0.2594, 0.3489)  #imagenet: 0.7696; #cifar10: 0.4370; cub200: 0.2594; sop:0.3299; cifar100: 0.3589; svhn:0.3198; dogs: 0.5440
    SOP_ = ratio(0.5690, 0.5646)  #imagenet: 0.78; cifar10: 0.5755; cub200:0.5690; svhn: 0.5807; dogs: 0.5768
    CIFAR10_ = ratio(0.9183, 0.9000)  #imagenet: 0.9465; cub200: 0.9171; cars196: 0.9183; sop:0.9133; cifar100: 0.9235; #svhn: 0.9253; dogs:0.9228
    CIFAR100_ = ratio(0.6551, 0.6058) #imagenet: 0.7516; cifar10:0.6862; cub200: 0.6556; cars196: 0.6551; sop:0.6469; svhn:0.6533; dogs: 0.676 
    SVHN_ = ratio(0.9653, 0.9594)  #imagenet: 0.9584; cifar10: 0.9650; cub200: 0.9640; cars196: 0.9653; sop:0.9621; cifar100: 0.9637; dogs:0.9635
    DOGS_ = ratio(0.4437, 0.4322)  #imagenet: 0.68; cifar10: 0.4797; cub200:0.4107; cars196: 0.4437; sop:0.4534; cifar100:0.4726; svhn:0.4557
    
    Y = [CUB200_, SOP_, CIFAR10_, CIFAR100_, SVHN_, DOGS_]  # Replace Y values

    model_name = "resnet18"          
    source_dataset = 'CARS196'      
    target_datasets = ['CUB200', 'SOP', 'CIFAR10','CIFAR100','SVHN', 'Dogs']      
    kernel_types = ['rbf']
    distance_type = 'mmd'
    distances = []

    for kernel_type in kernel_types:
        args = parser.parse_args(["--model_name", model_name, "--finetuned", "--dataset", source_dataset, "--source_dataset", source_dataset])
        print("Processing source dataset features")
        source_features = main(args)       
        for target_dataset in target_datasets:
            args = parser.parse_args(["--model_name", model_name, "--finetuned", "--dataset", target_dataset, "--source_dataset", source_dataset])
            print('processing features for', target_dataset)
            target_features = main(args) 

            embeddings = np.vstack((source_features, target_features)) 
            sc = StandardScaler()
            standardized_embeddings = sc.fit_transform(embeddings)

            pca = PCA()
            pca.fit(standardized_embeddings)
            explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
            n_components = (explained_variance_ratio < 0.9).sum() + 1
            
            #transform_ = PCA(n_components=n_components)
            transform_ = PCA(n_components=5)

            transform_pos_neg = transform_.fit(standardized_embeddings)
            transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
            source_features_transformed = transformed_embeddings[:len(source_features),:]
            target_features_transformed = transformed_embeddings[len(source_features):,:]

            if distance_type == 'mmd':
                distance, kxx, kyy, kxy = MMD(source_features_transformed, target_features_transformed, kernel=kernel_type)
            elif distance_type == 'wasserstein':
                distance = compute_wasserstein(source_features_transformed, target_features_transformed)
            distances.append(distance)

            print(target_dataset, 'distance:', distance)

    X = distances

    from scipy.stats import pearsonr
    # Calculating Pearson correlation coefficient
    correlation, p_value = pearsonr(X, Y)


    print(distances)
    print(Y)
    print(f"Correlation: {correlation}")
    print(f"P-value: {p_value}")

