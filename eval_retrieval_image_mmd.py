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

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# print(timm.list_models())
# exit(0)

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
parser.add_argument("--method", type=int, default=1, choices=[1,2,3], help='1:mmd pos vs neg; 2: mmd query vs pos, query vs neg; 3: wasserstein pos vs neg')
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
        # Load SOP, use 12 positive images per class
        image_folder = '/ebs/mdai/SOD/Stanford_Online_Products'
        data_file = os.path.join(image_folder, 'Ebay_test.txt')
        data = pd.read_csv(data_file, delimiter=' ', skiprows=1, names=['image_id', 'class_id', 'super_class_id', 'path'])
        # Count the number of images per class
        class_counts = data['class_id'].value_counts()
        # Filter classes with exactly 12 images
        #classes_with_ = class_counts[class_counts > 10].index
        classes_with_ = class_counts[class_counts > N].index
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
        test_classes_with_30 = test_class_counts[test_class_counts > N].index
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
        test_classes_with_more_than_N = test_class_counts[test_class_counts > N].index
        #print(f"Number of test classes with more than {N} images: {len(test_classes_with_more_than_N)}")
        # Filter the test data to include only classes with more than N test images
        filtered_data = test_data[test_data['class_id'].isin(test_classes_with_more_than_N)]
        # Get a list of test image paths for the filtered test data
        image_id_list = filtered_data['path'].tolist()

    if args.dataset == 'Imagenet':
        image_folder = '/ebs/mdai/imagenet/val'
        N = 10  # Update this value as needed

        # List all class folders in the validation directory
        class_folders = [os.path.join(image_folder, folder) for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

        filtered_data_list = []

        # Iterate through each class folder
        for class_folder in class_folders:
            class_id = os.path.basename(class_folder)
            image_paths = [os.path.join(class_folder, image) for image in os.listdir(class_folder)]
            
            if len(image_paths) > N:
                filtered_data_list.extend([(class_id, image_path) for image_path in image_paths])
                #print(f"Class ID: {class_id}, Number of Images: {len(image_paths)}")

        # Convert the filtered data to a DataFrame
        filtered_data = pd.DataFrame(filtered_data_list, columns=['class_id', 'path'])

        # Get a list of image paths for the filtered data
        image_id_list = filtered_data['path'].tolist()


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the pretrained ViT model
    # if args.finetuned == False:
    #     #print('Load model without finetuning:', args.model_name, 'Pretrained:',args.pretrained)
    #     if 'vit' or 'deit' in args.model_name:
    #         model = timm.create_model(args.model_name, pretrained=args.pretrained)
    #     elif 'resnet' in args.model_name:
    #         model = get_resnet_model(args.model_name)
    #     model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.eval()
    #     # Remove the last layer to get the embeddings
    #     model = torch.nn.Sequential(*list(model.children())[:-1])
    # else:
    #     if args.finetune_method == 'triplet':
    #         model_load_path = '/ebs/mdai/CUB200/{}_triplet.pth'.format(args.model_name)
    #         loaded_state_dict = torch.load(model_load_path)
    #         model = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    #         model = nn.Sequential(*list(model.children())[:-1]) # remove classification layer
    #     elif args.finetune_method == 'cls':
    #         model_load_path = '/ebs/mdai/CUB200/{}_classification.pth'.format(args.model_name)
    #         loaded_state_dict = torch.load(model_load_path)
    #         model = timm.create_model(args.model_name, pretrained=False, num_classes=loaded_state_dict['head.weight'].shape[0])
        
    #     model.load_state_dict(loaded_state_dict)
    #     model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.eval()
    #     #print('Loaded finetuned model', model_load_path)

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
    mmd_scores_positive, mmd_scores_negative = [], []
    kxx_scores, kyy_scores, kxy_scores = [], [], []
    w_scores = []
    intrinsic_dimensions = []
    from collections import defaultdict
    dic_score = defaultdict(list)
    BC_list, logme_list = [], []

    for idx in range(args.n_queries): 
        query_image = filtered_data.sample().iloc[0]
        #query_image = filtered_data[filtered_data['path'] == '/ebs/mdai/imagenet/val/n02091134/ILSVRC2012_val_00035403.JPEG'].iloc[0]
        #query_image = filtered_data[filtered_data['path'] == '070.Green_Violetear/Green_Violetear_0114_60809.jpg'].iloc[0]
        query_image_id = query_image['path']
        query_class_id = query_image['class_id']
        num_images_in_class = filtered_data[filtered_data["class_id"] == query_class_id].shape[0]
        #n_pos = min(n_pos, num_images_in_class - 1)

        # Get positive samples: images from the same class as the query image
        positive_examples = filtered_data[(filtered_data["class_id"] == query_class_id) & (filtered_data["path"] != query_image_id)]
        #n_pos = min(n_pos, num_images_in_class-1, len(positive_examples)-1)
        #print(n_pos)
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

        ### Unconditioned vit embeddings
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

        # # # estimate intrinsic dimension of embeddings
        # estimated_dimension = twonn_dimension(embeddings)
        # #estimated_dimension = mds_dimension(embeddings)
        # # intrinsic_dimensions.append(estimated_dimension)
        # print(estimated_dimension)

        if args.method == 1:
            if args.kernel_type == 'poly':
                mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel='poly',degree=args.degree)
            else:
                mmd_value, kxx, kyy, kxy = MMD(positive_embeddings, negative_embeddings, kernel=args.kernel_type)
            kxx_scores.append(kxx)
            kyy_scores.append(kyy)
            kxy_scores.append(kxy)

            weight = 1 / num_images_in_class
            #mmd_scores.append(mmd_value*weight)
            mmd_scores.append(mmd_value)
            #mmd_scores.append(np.exp(kxx-kxy))
            dic_score[query_class_id].append(mmd_value)
            
        if args.method == 2:
            mmd_query_positive = compute_MMD(query_embedding, positive_embeddings, kernel=args.kernel_type)
            mmd_query_negative = compute_MMD(query_embedding, negative_embeddings, kernel=args.kernel_type)
            mmd_scores_positive.append(mmd_query_positive)
            mmd_scores_negative.append(mmd_query_negative)
            mmd_scores.append(mmd_query_negative/mmd_query_positive)

        if args.method == 3:   
            w_value = compute_wasserstein(positive_embeddings, negative_embeddings) 
            weight = 1 / num_images_in_class
            w_scores.append(w_value*weight) 

        ###LogME
        logme = LogME(regression=False)
        np_labels =np.asarray([0]*standardized_positive_embeddings.shape[0] + [1]*standardized_negative_embeddings.shape[0])
        cur_logme = logme.fit(standardized_embeddings[1:,:], np_labels)
        logme_list.append(cur_logme)  

        ### BC
        transform_ = PCA(n_components=5)
        transform_pos_neg = transform_.fit(standardized_embeddings)
        transformed_embeddings = transform_pos_neg.transform(standardized_embeddings)
        query_embedding = transformed_embeddings[0,:].reshape(1, -1)
        positive_embeddings = transformed_embeddings[1:n_pos+1,:]
        negative_embeddings = transformed_embeddings[n_pos+1:,:]
        cur_BC = BC_dist(np.mean(positive_embeddings, axis=0), EmpiricalCovariance().fit(positive_embeddings).covariance_ , np.mean(negative_embeddings, axis=0), EmpiricalCovariance().fit(negative_embeddings).covariance_)
        BC_list.append(cur_BC) 

        torch.cuda.empty_cache()


    if args.method == 1:
        metric = np.mean(mmd_scores) 
        total_score = 0
        cnt_cls = 0
        for cur_class in dic_score:
            cnt_cls += 1
            total_score += np.mean(dic_score[cur_class])
        averaged_score = total_score / cnt_cls

    if args.method == 2:
        metric = round(np.mean(mmd_scores),4)

    if args.method == 3:
        metric = round(np.log(np.sum(w_scores)+1e-6),4)

    # print("Average intrinsic dimension:", round(np.mean(intrinsic_dimensions),2))
    # print("Std of intrinsic dimension:", round(np.std(intrinsic_dimensions),2))

    return metric #, np.mean(BC_list), np.mean(logme_list) #metric, np.log(averaged_score)


if __name__ == "__main__":
    #main(args)

    # model_names = ["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_small_patch16_224_dino","vit_small_patch16_224_in21k","vit_base_patch16_224"]  
    # datasets = ['CARS196','SOP','CUB200']

    # from collections import defaultdict
    # dic_Y = defaultdict(list)
    # dic_Y['CUB200'] = [0.67,0.76,0.74,0.75,0.78] #[0.7241, 0.8256, 0.7418, 0.8366, 0.8420] 
    # dic_Y['CARS196'] = [0.47,0.60,0.49,0.58,0.67] #[0.6184, 0.7594, 0.6360, 0.7335, 0.7695]
    # dic_Y['SOP'] = [0.74,0.7822,0.7795,0.779,0.81] #[0.7710,0.7824,0.7811,0.7931,0.7831]

    # with open('KRC_image_mmd.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Dataset', 'Degree', 'Weighted Kendall Rank Correlation mean', 'Weighted Kendall Rank Correlation std'])    
        
    #     for dataset in datasets:
    #         print()
    #         print('dataset:', dataset)
    #         for degree in range(1,2):
    #             print('degree:', degree)
    #             corr_list, corr_w_list = [], []
    #             corr_list_BC, corr_w_list_BC = [], []
    #             corr_list_logme, corr_w_list_logme = [], []
    #             for _ in range(10):
    #                 X = []
    #                 X_BC = []
    #                 X_logme = []
    #                 for model_name in model_names:
    #                     args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, "--degree", str(degree), 
    #                                               "--kernel_type",'poly',"--method",str(1)])
    #                     metric, BC, logme = main(args)
    #                     X.append(metric)
    #                     X_BC.append(BC)
    #                     X_logme.append(logme)
    #                     #print(dataset, model_name, BC)
    #                 corr_w = weightedtau(X, dic_Y[dataset]).correlation
    #                 corr_w_list.append(corr_w)
    #                 corr_w_BC = weightedtau(X_BC, dic_Y[dataset]).correlation
    #                 corr_w_list_BC.append(corr_w_BC)
    #                 corr_w_logme = weightedtau(X_logme, dic_Y[dataset]).correlation
    #                 corr_w_list_logme.append(corr_w_logme)
    #             print('Weighted Kendall Rank correlation for proposed metric -- mean: %.5f'% np.mean(corr_w_list), 'std: %.5f' % np.std(corr_w_list))
    #             print('Weighted Kendall Rank correlation for BC -- mean: %.5f'% np.mean(corr_w_list_BC), 'std: %.5f' % np.std(corr_w_list_BC))
    #             print('Weighted Kendall Rank correlation for lomge --  mean: %.5f'% np.mean(corr_w_list_logme), 'std: %.5f' % np.std(corr_w_list_logme))
    #             print()
                
    #             # Write data to CSV
    #             writer.writerow([dataset, degree, "%.5f" % np.mean(corr_w_list), "%.5f" % np.std(corr_w_list)])


    model_names = ["vit_small_patch16_224"]
    datasets = ['CUB200']    
    kernel_types = ['linear','cosine','poly','rbf']
    for kernel_type in kernel_types:
        for dataset in datasets:
            print()
            print('dataset:', dataset)
            for model_name in model_names:
                print(model_name)
                s_pretrained, s_finetuned = [], []
                for degree in range(1,2):
                    for _ in range(5):
                        # args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, "--degree", str(degree), 
                        #                             "--kernel_type",'poly',"--method",str(1)])
                        # metric = main(args)
                        # print('degree:', degree, metric)

                        args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, 
                                                    "--kernel_type",kernel_type,"--method",str(1)])
                        metric = main(args)
                        s_pretrained.append(metric)
                        args = parser.parse_args(["--model_name", model_name, "--dataset", dataset, 
                                                    "--kernel_type",kernel_type,"--method",str(1),"--finetuned"])
                        metric = main(args)
                        s_finetuned.append(metric)

        print('Kernel:', kernel_type)
        print('Pretrained:', np.mean(s_pretrained))
        print('Finetuned:', np.mean(s_finetuned))
        print()
