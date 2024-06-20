import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import transforms, datasets, models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import os
import numpy as np
import random
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
import timm
import argparse
import datetime
from PIL import Image
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from utils_retrieval import *
import gc

# print(timm.list_models())
# exit(0)


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
    baseline_scaling = np.exp(alpha * x2)

    # Transferability metric
    transfer_metric = improvement_ratio #* baseline_scaling
    return transfer_metric


def subsample_dataset(dataset, num_samples):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return torch.utils.data.Subset(dataset, indices)


# Function to extract features from test dataset
def extract_features(model_name, model, data_loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for data, _ in tqdm(data_loader):
            data = data.to(device)
            if 'vit' in model_name or 'deit' in model_name:
                #output = model(data)[:, 0, :]
                output = model(data)[:, 1:, :].mean(dim=1)
            else:
                output = model(data)
            features.append(output.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features


def get_transform(train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224), #224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256), # 256
            transforms.CenterCrop(224), #224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CUB200", choices=['SOP','CUB200', 'CARS196', 'Imagenet'])
parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
parser.add_argument("--finetune_type", type=str, default="contrastive", choices=["classification","contrastive"])
parser.add_argument("--distance_type", type=str, default='mmd',choices=['mmd','wasserstein'])
parser.add_argument("--kernel_type", type=str, default='linear')
parser.add_argument("--n_samples", default=10000, type=int)
parser.add_argument("--pca", action="store_true")
parser.add_argument("--finetuned", action="store_true")


def main():
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_names = ['resnet18']#'efficientnet_b0']#"resnet18","deit3_small_patch16_224","vit_small_patch16_224"] #"vit_tiny_patch16_224", "vit_small_patch16_224", "vit_small_patch16_224_dino","vit_small_patch16_224_in21k","vit_base_patch16_224"]  
    dataset_names = ['Dogs','CUB200','CARS196','SOP','SVHN','CIFAR100','CIFAR10']#,'Flowers']
    distances = []

    batch_size = 8
    test_transform = get_transform(train=False)
    
    # ImageNet
    imagenet_folder = '/ebs/mdai/imagenet/val'
    imagenet_dataset = CustomImageFolder(imagenet_folder, transform=test_transform)  

    # imagenet_folder = '/ebs/mdai/Flowers'
    # imagenet_dataset = CustomImageFolder(os.path.join(imagenet_folder, 'test'), transform=test_transform) 

    imagenet_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    subsampled_dataset = subsample_dataset(imagenet_loader.dataset, min(args.n_samples,len(imagenet_dataset)))
    imagenet_loader = DataLoader(subsampled_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True)

    for args.model_name in model_names: 
        model = timm.create_model(args.model_name, pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        if args.finetuned:
            model_load_path = '/ebs/mdai/CUB200/vit_small_patch16_224_contrastive.pth'
            loaded_state_dict = torch.load(model_load_path, map_location="cuda:0")
            model.load_state_dict(loaded_state_dict, strict=False)
            print('Loaded finetuned model', model_load_path)
        model = model.to(device)
        model.eval()

        # Extract imagenet features
        print()
        print()
        print("Number of parameters in the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Extracting imagenet features')
        imagenet_features = extract_features(args.model_name, model, imagenet_loader, device)

        for args.dataset in dataset_names:
            if args.dataset == 'CUB200':
                data_dir = '/ebs/mdai/CUB200/CUB_200_2011'
            elif args.dataset == 'CARS196':
                data_dir = '/ebs/mdai/CARS196'
            elif args.dataset == 'SOP':
                data_dir = '/ebs/mdai/SOD/Stanford_Online_Products'
            elif args.dataset == 'Dogs':
                data_dir = '/ebs/mdai/Dogs'
            elif args.dataset == 'Flowers':
                data_dir = '/ebs/mdai/Flowers'

            if args.dataset == 'CARS196':
                test_dataset = CustomImageFolder_CARS196(os.path.join(data_dir, 'test'), transform=test_transform)
            elif args.dataset == 'CUB200' or args.dataset == 'SOP' or args.dataset == 'Dogs' or args.dataset == 'Flowers':
                test_dataset = CustomImageFolder(os.path.join(data_dir, 'test'), transform=test_transform) 

            if args.dataset == 'CIFAR10':
                #train_dataset = datasets.CIFAR10(root='/data/mdai/CIFAR10', train=True, transform=transform, download=True)
                test_dataset = datasets.CIFAR10(root='/data/mdai', train=False, transform=test_transform, download=True)
            elif args.dataset == 'CIFAR100':
                #train_dataset = datasets.CIFAR100(root='/data/mdai/CIFAR100', train=True, transform=transform, download=True)
                test_dataset = datasets.CIFAR100(root='/data/mdai', train=False, transform=test_transform, download=True)  
            elif args.dataset == 'SVHN':
                #train_dataset = datasets.CIFAR100(root='/data/mdai/CIFAR100', train=True, transform=transform, download=True)
                test_dataset = datasets.SVHN(root='/data/mdai', split='test', transform=test_transform, download=True)  

            print(len(test_dataset))

            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=4, pin_memory=True)
            subsampled_test_dataset = subsample_dataset(test_loader.dataset, min(args.n_samples,len(test_dataset)))
            test_loader = DataLoader(subsampled_test_dataset, batch_size=batch_size, shuffle=False, 
                                        num_workers=4, pin_memory=True)
            
            dataset_features = extract_features(args.model_name, model, test_loader, device)

            ### if PCA
            if args.pca:
                embeddings = np.vstack((imagenet_features, dataset_features)) 
                sc = StandardScaler()
                standardized_embeddings = sc.fit_transform(embeddings)

                pca = PCA()
                pca.fit(standardized_embeddings)
                explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
                n_components = (explained_variance_ratio < 0.99).sum() + 1
                transform_ = PCA(n_components=n_components)

                transform_pos_neg = transform_.fit(standardized_embeddings)
                transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
                imagenet_features_transformed = transformed_embeddings[:len(imagenet_features),:]
                dataset_features_transformed = transformed_embeddings[len(imagenet_features):,:]

            # Calculate distance
            if args.pca:
                if args.distance_type == 'mmd':
                    distance, kxx, kyy, kxy = MMD(imagenet_features_transformed, dataset_features_transformed, kernel=args.kernel_type)
                elif args.distance_type == 'wasserstein':
                    distance = compute_wasserstein(imagenet_features_transformed, dataset_features_transformed)
            else:
                if args.distance_type == 'mmd':
                    distance, kxx, kyy, kxy = MMD(imagenet_features, dataset_features, kernel=args.kernel_type)
                elif args.distance_type == 'wasserstein':
                    distance = compute_wasserstein(imagenet_features, dataset_features)

            print(args.dataset, args.model_name, 'distance:', distance)



            torch.cuda.empty_cache()
            gc.collect()

            distances.append(distance)

    # finetuned: pretrained / unpretrained
    Flowers_ = ratio(0.9804,0.8686)
    Dogs_ = ratio(0.68,0.4322)
    CUB200_ = ratio(0.7228,0.4566)
    Cars196_ = ratio(0.7696,0.3489)
    SOP_ = ratio(0.78,0.5646)
    SVHN_ = ratio(0.9584,0.9594)
    CIFAR10_ = ratio(0.9465,0.9000)
    CIFAR100_ = ratio(0.7516,0.6058)

    # Flowers_ = ratio(0.9804,0.0873)
    # Dogs_ = ratio(0.68,0.2156)
    # CUB200_ = ratio(0.7228,0.0134)
    # Cars196_ = ratio(0.7696,0.1784)
    # SOP_ = ratio(0.78,0.2246)
    # SVHN_ = ratio(0.9584,2453)
    # CIFAR10_ = ratio(0.9465,0.2127)
    # CIFAR100_ = ratio(0.7516,0.0759)


    X = distances
    Y = [Dogs_, CUB200_, Cars196_, SOP_, SVHN_, CIFAR100_, CIFAR10_]#, Flowers_]  # Replace with your Y values

    from scipy.stats import pearsonr
    # Calculating Pearson correlation coefficient
    correlation, p_value = pearsonr(X, Y)

    print(f"Correlation: {correlation}")
    print(f"P-value: {p_value}")

if __name__ == "__main__":
    main()