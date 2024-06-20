import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
from collections import defaultdict
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

torch.autograd.set_detect_anomaly(True)


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
    

class ContrastiveDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # Create a mapping from label to indices
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index]

        # Get list of indices for the same class
        same_class_indices = self.label_to_indices[anchor_label]

        # Sample a distinct positive
        while True:
            pos_idx = random.choice(same_class_indices)
            if pos_idx != index:  # Ensure it's a different instance
                break
        pos_img, _ = self.dataset[pos_idx]

        return anchor_img, pos_img, anchor_label  # Note: We only return one label since anchor and positive labels are the same


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0) // 2
        anchor_embeddings = embeddings[:batch_size]
        positive_embeddings = embeddings[batch_size:]

        # Compute pairwise cosine similarity
        sim_matrix = F.cosine_similarity(anchor_embeddings.unsqueeze(1), positive_embeddings.unsqueeze(0), dim=2)

        # Extract anchor-positive similarities (diagonal of the matrix)
        pos_sim = torch.diag(sim_matrix)

        # Construct the numerator (exp of positive similarities)
        numerators = torch.exp(pos_sim / self.temperature)
        
        # For negatives, we won't use the label information. We'll just sum over all off-diagonal elements for the denominator.
        denominators = torch.exp(sim_matrix / self.temperature).sum(dim=1) - numerators

        # individual_losses = -torch.log(numerators / (denominators+1e-8))
        # loss = individual_losses.mean()

        loss = -torch.log(torch.clamp(numerators / (denominators + 1e-8), max=1)).mean()

        return loss


def train(model_name, model, train_loader, criterion, optimizer, device, num_epochs, epoch, local_rank=0):
    model.train()
    running_loss = 0.0

    if dist.is_available() and dist.is_initialized():
        local_rank = dist.get_rank()
    else:
        local_rank = 0

    if local_rank == 0:  # Only apply progress bar for rank 0
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    else:
        train_loader = train_loader  # Do not apply progress bar for other ranks

    for batch_idx, (anchor, positive, labels) in enumerate(train_loader):
        anchor, positive = anchor.to(device), positive.to(device)

        optimizer.zero_grad()

        if 'resnet' in model_name or 'efficientnet' in model_name:
            anchor_output = model(anchor)
            positive_output = model(positive)
        else:
            # anchor_output = model(anchor)[:, 0, :]
            # positive_output = model(positive)[:, 0, :]
            anchor_output = model(anchor)[:, 1:, :].mean(dim=1)
            positive_output = model(positive)[:, 1:, :].mean(dim=1)

        anchor_output = F.normalize(anchor_output, p=2, dim=-1)
        positive_output = F.normalize(positive_output, p=2, dim=-1)

        embeddings = torch.cat([anchor_output, positive_output], dim=0)
        # Duplicate and concatenate labels since anchor and positive have same labels

        labels = torch.cat([labels, labels], dim=0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss = criterion(embeddings, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)



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
    

def evaluate_recall_map(model_name, model, data_loader, device, k):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data = data.to(device)
            target = target.to(device)
            if 'resnet' in model_name or 'efficientnet' in model_name:
                output = model(data)
            else:
                #output = model(data)[:,0,:]
                output = model(data)[:, 1:, :].mean(dim=1)
            output = F.normalize(output, p=2, dim=-1)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
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
    mAP = mean_average_precision(distances, labels)

    if local_rank == 0:
        print(f"Recall@{k}: {recall_at_k:.4f}")
        print(f"mAP: {mAP:.4f}")

    return recall_at_k, mAP


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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CUB200", choices=['SOP','CUB200', 'CARS196', 'Imagenet'])
parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--pretrained", action="store_false")
parser.add_argument('--rank',default=-1,type=int)
parser.add_argument('--local_rank',default=-1,type=int)
parser.add_argument('--dist_backend',default='nccl',type=str)
parser.add_argument('--multiprocessing-distributed',action='store_true')


def main():
    args = parser.parse_args()

    os.environ['NCCL_DEBUG'] = 'INFO'
    global rank
    global local_rank
    global gpus_per_node
    global world_size

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    gpus_per_node = torch.cuda.device_count
    world_size = int(os.environ['WORLD_SIZE'])

    if rank == 0:
        print('Waiting for all processes to connect')

    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(0,30000))
    
    gpu_id = os.environ['LOCAL_RANK']
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model_names = ["resnet18"]#'efficientnet_b0']#"resnet50"]"vit_tiny_patch16_224", "vit_small_patch16_224", "vit_small_patch16_224_dino","vit_small_patch16_224_in21k","vit_base_patch16_224"]  
    dataset_names = ["SOP"]

    for args.dataset in dataset_names:
        num_epochs = args.epoch
        lr = 5e-5 * world_size
        batch_size = 64
        if args.dataset == 'SOP':
            num_epochs = 30
        elif args.dataset == 'CARS196':
            num_epochs = 100
            #lr = 1e-4

        # unpretrained
        lr = 1e-3 * world_size
        num_epochs = 100

        for args.model_name in model_names:
            if args.model_name == "vit_base_patch16_224":
                batch_size = 32
                lr = lr / 5
            if args.model_name == "vit_small_patch16_224_dino":
                lr = 5e-5

            if args.dataset == 'CUB200':
                data_dir = '/data/mdai/CUB200/CUB_200_2011'
            elif args.dataset == 'CARS196':
                data_dir = '/data/mdai/CARS196'
            elif args.dataset == 'SOP':
                data_dir = '/data/mdai/SOD/Stanford_Online_Products'

            train_transform = get_transform(train=True)
            test_transform = get_transform(train=False)

            if args.dataset == 'CARS196':
                train_dataset = CustomImageFolder_CARS196(os.path.join(data_dir, 'train'), transform=train_transform)
                test_dataset = CustomImageFolder_CARS196(os.path.join(data_dir, 'test'), transform=test_transform)
            elif args.dataset == 'CUB200' or args.dataset == 'SOP':
                train_dataset = CustomImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
                test_dataset = CustomImageFolder(os.path.join(data_dir, 'test'), transform=test_transform) 

            if args.dataset == 'CIFAR10':
                train_dataset = datasets.CIFAR10(root='/data/mdai', train=True, transform=train_transform, download=False)
                test_dataset = datasets.CIFAR10(root='/data/mdai', train=False, transform=test_transform, download=False)
            elif args.dataset == 'CIFAR100':
                train_dataset = datasets.CIFAR100(root='/data/mdai', train=True, transform=train_transform, download=False)
                test_dataset = datasets.CIFAR100(root='/data/mdai', train=False, transform=test_transform, download=False)  
            elif args.dataset == 'SVHN':
                train_dataset = datasets.SVHN(root='/data/mdai', split='train', transform=train_transform, download=False)
                test_dataset = datasets.SVHN(root='/data/mdai', split='test', transform=test_transform, download=False)     

            # Use regular train_dataset and test_dataset for DataLoader
            train_dataset = ContrastiveDataset(train_dataset)
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                    num_workers=16, pin_memory=True, sampler=train_sampler)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=16, pin_memory=True)

            
            print('Finished loading data, start training...')
            model = timm.create_model(args.model_name, pretrained=True).to(device)
            model = nn.Sequential(*list(model.children())[:-1])

            if 'resnet' in args.model_name or 'efficientnet' in args.model_name:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(gpu_id)], output_device=int(gpu_id))
            model_without_ddp = model.module

            # if local_rank == 0:
            #     print(model)

            # Set up the triplet loss and optimizer
            criterion = ContrastiveLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

           # Train the model
            model_save_path = '/data/mdai/{}/{}_contrastive.pth'.format(args.dataset,args.model_name)
            stats = []
            best_recall = 0
            if local_rank == 0:
                print(args.dataset, args.model_name)
            for epoch in range(1,num_epochs+1):
                loss = train(args.model_name, model, train_loader, criterion, optimizer, device, num_epochs, epoch)
                scheduler.step()
                if epoch % 10 == 0:
                    if local_rank == 0:
                        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")
                    recall, mAP  = evaluate_recall_map(args.model_name, model.module, test_loader, device, k=1)
                    stats.append([epoch, recall, mAP])
                    #torch.save(model_without_ddp.state_dict(), '/data/mdai/{}/{}_contrastive_epoch_{}.pth'.format(args.dataset,args.model_name,epoch))
                    if recall > best_recall:
                        best_recall = recall
                        torch.save(model_without_ddp.state_dict(), model_save_path)
            stats_save_path = '/data/mdai/{}/{}_contrastive_stats.txt'.format(args.dataset,args.model_name)
            with open(stats_save_path, "w") as file:
                for item in stats:
                    file.write(f"{item}\n")

            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()