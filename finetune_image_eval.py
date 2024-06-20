import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
import os
import numpy as np
from typing import List, Dict, Tuple, Any
import random
import shutil
from sklearn.model_selection import train_test_split
import timm
import argparse
#Evaluation
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
from PIL import Image
import logging
import matplotlib.pyplot as plt


def get_transform(train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


def get_resnet_model(model_name, num_classes=None):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if num_classes is not None:
        # Get the number of input features of the last layer
        num_features = model.fc.in_features
        # Replace the last layer with a new one with the desired number of classes
        model.fc = nn.Linear(num_features, num_classes)

    return model


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
    

def custom_pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        try:
            img = img.convert("RGB")
        except OSError as e:
            #logging.warning(f"Error loading image at {path}: {e}")
            return None
    return img

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


def evaluate_recall_map(finetuned, model, data_loader, device, k, finetune_type):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data = data.to(device)
            target = target.to(device)
            #output = model(data)[:,0,:]
            output = model(data)[:, 1:, :].mean(dim=1)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    #print(features.shape)
    #distances = pairwise_distances(features)
    distances = cosine_distances(features)

    num_queries = len(labels)
    num_correct = 0

    for i, row in enumerate(distances):
        row_sorted_idx = np.argsort(row)
        top_k_indices = row_sorted_idx[1:k+1]  # Skip the first index (self)
        top_k_labels = labels[top_k_indices]

        if labels[i] in top_k_labels:
            num_correct += 1

    recall_at_k = num_correct / num_queries
    # print(f"Recall@{k}: {recall_at_k:.4f}")
    # mAP = mean_average_precision(distances, labels)
    # print(f"mAP: {mAP:.4f}")
    return recall_at_k#, mAP


def average_precision(sorted_labels, true_label):
    num_relevant = 0
    num_positive = 0
    ap = 0.0

    for i, label in enumerate(sorted_labels):
        if label == true_label:
            num_relevant += 1
            ap += num_relevant / (i + 1)

        if label == true_label:
            num_positive += 1

    if num_positive == 0:
        return 0.0

    return ap / num_positive

def mean_average_precision(distances, labels):
    num_queries = len(labels)
    ap_sum = 0

    for i, row in enumerate(distances):
        row_sorted_idx = np.argsort(row)
        sorted_labels = labels[row_sorted_idx[1:]]  # Skip the first index (self)
        true_label = labels[i]

        ap = average_precision(sorted_labels, true_label)
        ap_sum += ap

    mAP = ap_sum / num_queries
    return mAP


def evaluate_recall_per_class(model, data_loader, device, k):
    """
    Evaluate the recall@k for each class.
    
    :param model: The model to evaluate.
    :param data_loader: DataLoader for the test dataset.
    :param device: Device to run the computations on.
    :param k: Number of nearest samples to consider for recall.
    
    :return: Dictionary of recalls per class, average recall.
    """
    
    model.eval()  # Set model to evaluation mode
    
    # List to store embeddings and labels from the dataset
    embeddings = []
    labels = []
    
    # Extract embeddings and labels for the entire dataset
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)[:,1:,:].mean(axis=1)
            embeddings.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    # Convert lists to numpy arrays
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    # Compute pairwise distances
    distances = cosine_distances(embeddings, embeddings)
    
    recalls_per_class = {}

    # For each unique class in the dataset
    for unique_label in np.unique(labels):
        # Indices of samples belonging to the current class
        class_indices = np.where(labels == unique_label)[0]
        
        recall_count = 0
        
        # For each sample in the current class
        for idx in class_indices:
            sample_distances = distances[idx]
            
            # Exclude the sample itself by setting its distance to a large value
            sample_distances[idx] = float('inf')
            
            # Get indices of k-nearest samples
            nearest_indices = np.argsort(sample_distances)[:k]
            
            if labels[idx] in labels[nearest_indices]:
                recall_count += 1
        
        # Compute and store recall for the current class
        recalls_per_class[unique_label] = recall_count / len(class_indices)
    
    average_recall = np.mean(list(recalls_per_class.values()))
    
    class_counts = {cls: np.sum(labels == cls) for cls in recalls_per_class.keys()}
    
    return recalls_per_class, average_recall, class_counts


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CUB200", choices=['SOP','CUB200', 'CARS196', 'Imagenet'])
parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--pretrained", action="store_false")
parser.add_argument("--finetuned", action="store_true")
parser.add_argument("--finetune_type", type=str, default="contrastive", choices=["cls","contrastive"])

args = parser.parse_args()

train_transform = get_transform(train=True)
test_transform = get_transform(train=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_datasets = ['Flowers','Dogs','CUB200','CARS196']
model_names = ["vit_small_patch16_224"]#, "vit_small_patch16_224_dino","vit_small_patch16_224_in21k","vit_base_patch16_224"] 

for args.dataset in all_datasets:
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
        #train_dataset = datasets.CIFAR10(root='/ebs/mdai', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(root='/ebs/mdai', train=False, transform=test_transform, download=True)
    elif args.dataset == 'CIFAR100':
        #train_dataset = datasets.CIFAR100(root='/ebs/mdai', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(root='/ebs/mdai', train=False, transform=test_transform, download=True)  
    elif args.dataset == 'SVHN':
        #train_dataset = datasets.SVHN(root='/ebs/mdai', split='train', transform=train_transform, download=True)
        test_dataset = datasets.SVHN(root='/ebs/mdai', split='test', transform=test_transform, download=True)   

    # Use regular train_dataset and test_dataset for DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    for args.model_name in model_names:
        model = timm.create_model(args.model_name, pretrained=args.pretrained)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        if args.finetuned:
            model_load_path = '/ebs/mdai/{}/{}_contrastive.pth'.format(args.dataset,args.model_name)
            #model_load_path = '/ebs/mdai/CUB200/vit_small_patch16_224_contrastive.pth'
            loaded_state_dict = torch.load(model_load_path, map_location="cuda:0")
            model.load_state_dict(loaded_state_dict, strict=True)
            print('Loaded finetuned model', model_load_path)
        model = model.to(device)
        model.eval()

        recall = evaluate_recall_map(args.finetuned, model, test_loader, device, k=1, finetune_type=args.finetune_type)
        print(args.dataset, args.model_name, 'Recall@1:', recall)

        recalls_per_class, average_recall, class_counts = evaluate_recall_per_class(model, test_loader, device, k=1)
        print(args.dataset, args.model_name, 'Averaged recall@1:', average_recall)
        # Sort classes by their counts
        sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x])
        # Sort recalls in the same order
        sorted_recalls = [recalls_per_class[cls] for cls in sorted_classes]

        plt.figure(figsize=(10,5))
        plt.bar(range(len(sorted_classes)), sorted_recalls, color='blue')
        plt.xlabel('Class sorted by number of images in ascending order')
        plt.ylabel('Recall@1')
        plt.title('Recall@1 per class for {}'.format(args.dataset))
        plt.xticks([])  # Hide x-axis tick labels
        plt.grid(axis='y')
        plt.savefig("recall_vs_class_count_{}.png".format(args.dataset))



