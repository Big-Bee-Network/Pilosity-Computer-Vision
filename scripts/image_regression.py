'''
Name: Luning Ding
Date: July 5, 2023
'''

from __future__ import print_function, division
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from PIL import Image
import cv2
from skimage.io import imread, imshow, imsave
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
import csv
from collections import defaultdict
from classes import HairinessRatingDataset, BasicBlock, Bottleneck, ResNet
from functions import show_landmarks_batch, train_hairiness_model


# Train and save the image regression model
def image_regression(csv_file, root_dir, model_save_path):
    '''
    csv_file: the path to csv file that store the ground truth (manual) ratings for the artifical bees
    root_dir: the directory to artificial bee crops
    model_save_path: the path to save the model
    '''

    # prepare the variables
    rating_frame = pd.read_csv(csv_file)
    rating_frame['rank'] = rating_frame.iloc[:, 10].rank(method='dense')
    # len(rating_frame.iloc[:,10].unique())

    # convert ranking dataframe to dictionary
    rank_dict = rating_frame.set_index(['7'])['1.8'].to_dict()
    # print(rank_dict)

    rating_dataset = HairinessRatingDataset(csv_file, root_dir)
    ratings = rating_dataset.rating_frame
    # show data distribution
    ratings.hist(bins=5, figsize=(8, 8))

    rating_dataset = HairinessRatingDataset(csv_file, root_dir, transform=transforms.Compose([
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1, 1, 1])
    ]))

    # Normalize the images by calculating the mean and std of the images in the dataset
    dataset_loader = DataLoader(rating_dataset,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0)  # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for inputs in dataset_loader:
        psum += inputs['image'].sum(axis=[0, 2, 3])
        psum_sq += (inputs['image'] ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(rating_dataset) * 256 * 256

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # Transfrom data
    data_transform = transforms.Compose([
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=total_mean,
                             std=total_std)
    ])

    # Set up the dataset
    rating_dataset = HairinessRatingDataset(csv_file, root_dir,
                                            transform=data_transform)

    # Set the ratio of the length of training dataset and validation dataset to be 8:2
    train_len = int(len(rating_dataset) * 0.8)
    val_len = len(rating_dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(rating_dataset, [train_len, val_len])
    # print('training set length:', len(train_set))
    # print('validation set length:', len(val_set))

    # Load the data
    train_dataloader = DataLoader(train_set,
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=0)  # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.
    val_dataloader = DataLoader(val_set,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0)  # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}

    # Visualize Datasets Only
    dataloader = DataLoader(train_set,
                            batch_size=4,
                            shuffle=True,
                            num_workers=0)  # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.

    # if __name__ == '__main__':
    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch, sample_batched['image'].size(),
        #       sample_batched['landmarks'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    # Configuration of the model
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])

    resnet18_config = ResNetConfig(block=BasicBlock,
                                   n_blocks=[2, 2, 2, 2],
                                   channels=[64, 128, 256, 512])

    resnet152_config = ResNetConfig(block=Bottleneck,
                                    n_blocks=[3, 8, 36, 3],
                                    channels=[64, 128, 256, 512])

    OUTPUT_DIM = 20

    resnet50 = ResNet(resnet50_config, OUTPUT_DIM)
    resnet18 = ResNet(resnet18_config, OUTPUT_DIM)
    resnet152 = ResNet(resnet152_config, OUTPUT_DIM)

    # define the model
    model = resnet50

    pretrained_model = models.resnet50(pretrained=True)

    IN_FEATURES = pretrained_model.fc.in_features

    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    pretrained_model.fc = fc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(pretrained_model.state_dict())

    idx_tensor = [idx for idx in range(20)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)
    rank_list = []
    for i in idx_tensor:
        rank_list.append(rank_dict[i.item()])
        rank_tensor = Variable(torch.FloatTensor(rank_list)).to(device)

    # model_ft = models.resnet18(weights='IMAGENET1K_V1')
    # num_ftrs = model_ft.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Start training
    model_trained, dict_result = train_hairiness_model(model,
                                                       device = device,
                                                       dataloaders = dataloaders,
                                                       rank_tensor = rank_tensor,
                                                       dataset_sizes = dataset_sizes,
                                                       criterion = criterion,
                                                       optimizer = optimizer,
                                                       scheduler = exp_lr_scheduler,
                                                       num_epochs=100)

    PATH = model_save_path
    torch.save(model_trained, PATH)


'''
load the model to get predicted ratings from the model
store ground truth ratings, predicted rating, entropy values to cvs
plot and find the correlation between hairiness ratings and entropy values
'''

def predicted_rating_entropy_values(csv_file, root_dir, model_save_path, predicted_rating_csv_path, data_transform):
    '''
    csv_file: the path to csv file that store the ground truth (manul) ratings for the artifical bees
    root_dir: the directory to artificial bees
    model_save_path: the path to save the model
    predicted_rating_csv_path: the path to csv file that stores the predicted ratings from the model and entropy values
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Configuration of the model
    OUTPUT_DIM = 20
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])
    model = ResNet(resnet50_config, OUTPUT_DIM)

    # model = torch.load(model_save_path, map_location=device)
    softmax = nn.Softmax(dim = -1).to(device) # I set dim = -1 and this seemed to actually allow the model to train. Could be a source of an issue though

    rating_frame = pd.read_csv(csv_file)
    rank_dict = rating_frame.set_index(['7'])['1.8'].to_dict()

    predicted_dict = defaultdict(list)

    idx_tensor = [idx for idx in range(20)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)
    rank_list = []
    for i in idx_tensor:
        rank_list.append(rank_dict[i.item()])

    rank_tensor = Variable(torch.FloatTensor(rank_list)).to(device)

    # Set up the dataset
    rating_dataset = HairinessRatingDataset(csv_file, root_dir,
                                            transform=data_transform)

    # Set the ratio of the length of training dataset and validation dataset to be 8:2
    train_len = int(len(rating_dataset) * 0.8)
    val_len = len(rating_dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(rating_dataset, [train_len, val_len])

    # Normalize the images by calculating the mean and std of the images in the dataset
    dataset_loader = DataLoader(rating_dataset,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0)  # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for inputs in dataset_loader:
        psum += inputs['image'].sum(axis=[0, 2, 3])
        psum_sq += (inputs['image'] ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(rating_dataset) * 256 * 256

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=total_mean,
                             std=total_std)
    ])

    with torch.no_grad():
        i = 0
        for imgname in os.listdir(root_dir):
            i += 1
            # im = imread(os.path.join(root_dir, imgname))
            image = Image.open(os.path.join(root_dir, imgname))
            image = data_transform(image).to(device)
            image = image[None, :]
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            softmaxed_output = softmax(outputs)
            expected_output = torch.sum(softmaxed_output * rank_tensor, 1)
            expected_output = expected_output[None]
            expected_output = torch.transpose(expected_output, 0, 1)
        # Save predicted ratings
        for idx in range(val_len):
            image_path = val_set[idx]['name']
            predicted_dict['image_name'].append(image_path)
            predicted_dict['ground_truth_rating'].append(val_set[idx]['rating'].item())
            predicted_dict['predicted_score'].append(expected_output.item())

            # Entropy Analysis
            im = imread(image_path)
            # convert image to grayscale
            im_gray = rgb2gray(im)
            entropy_image = entropy(im_gray, disk(6))
            # exclude zeros in the matrix
            entropy_image_1 = entropy_image[entropy_image != 0]
            predicted_dict['Entropy_mean'].append(np.mean(entropy_image_1))
            predicted_dict['Entropy_median'].append(np.median(entropy_image_1))
            predicted_dict['Entropy_std'].append(np.std(entropy_image_1))

        figure, axis = plt.subplots(2, 2)

        # create scatter plot, exploring the relationship with rating and entropy values
        axis[0, 0].scatter(predicted_dict['predicted_score'], predicted_dict['Entropy_mean'])
        axis[0, 0].set_xlabel('predicted')
        axis[0, 0].set_ylabel('Entropy_mean')

        axis[0, 1].scatter(predicted_dict['ground_truth_rating'], predicted_dict['Entropy_mean'])
        axis[0, 1].set_xlabel('ground_truth')
        axis[0, 1].set_ylabel('Entropy_mean')

        axis[1, 0].scatter(predicted_dict['predicted_score'], predicted_dict['Entropy_median'])
        axis[1, 0].set_xlabel('predicted')
        axis[1, 0].set_ylabel('Entropy_std')

        axis[1, 1].scatter(predicted_dict['ground_truth_rating'], predicted_dict['Entropy_median'])
        axis[1, 1].set_xlabel('ground_truth')
        axis[1, 1].set_ylabel('Entropy_std')

        figure.tight_layout()
        plt.show()

        # Save the results from above
        with open(predicted_rating_csv_path, "w") as outfile:

            # creating a csv writer object
            writerfile = csv.writer(outfile)

            # writing dictionary keys as headings of csv
            writerfile.writerow(predicted_dict.keys())

            # writing list of dictionary
            writerfile.writerows(zip(*predicted_dict.values()))


'''
Predicted ratings are influenced by surface area percentage of the bees compared to the whole bee
This functions takes that into account by dividing predicted scores by surface area percentage
'''


def predicted_rating_entropy_surface_area(csv_file, model_save_path, root_dir, crop_dir, surface_area_csv, data_transform):
    '''
    csv_file: the path to csv file that store the ground truth (manul) ratings for the artifical bees
    model_save_path: the path to save the model
    root_dir: the directory to artificial bees
    surface_area_csv: csv that stores the surface area percentage of bee hair compared to the whole bee
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_save_path, map_location=device)
    softmax = nn.Softmax(dim = -1).to(device)

    rating_frame = pd.read_csv(csv_file)
    rank_dict = rating_frame.set_index(['7'])['1.8'].to_dict()

    idx_tensor = [idx for idx in range(20)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)
    rank_list = []
    for i in idx_tensor:
        rank_list.append(rank_dict[i.item()])

    rank_tensor = Variable(torch.FloatTensor(rank_list)).to(device)

    surface_area_df = pd.read_csv(surface_area_csv)
    imgname_df = surface_area_df['name']
    surface_area_df = surface_area_df['percentage of pixels']

    whole_bee_dict = defaultdict(list)

    # Set up the dataset
    rating_dataset = HairinessRatingDataset(csv_file, crop_dir,
                                            transform=data_transform)

    # Normalize the images by calculating the mean and std of the images in the dataset
    dataset_loader = DataLoader(rating_dataset,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0)  # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for inputs in dataset_loader:
        psum += inputs['image'].sum(axis=[0, 2, 3])
        psum_sq += (inputs['image'] ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(rating_dataset) * 256 * 256

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    data_transform = transforms.Compose([
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=total_mean,
                             std=total_std)
    ])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=total_mean,
                             std=total_std)
    ])

    with torch.no_grad():
        for imgname, surf_percent in zip(imgname_df, surface_area_df):

            crop_dict = defaultdict(list)

            plt.figure(figsize=(4, 3), dpi=80)
            print(imgname)
            im = imread(os.path.join(root_dir, imgname))
            plt.imshow(im)
            plt.show()

            image = cv2.imread(os.path.join(root_dir, imgname))
            height, width = np.shape(image)[0], np.shape(image)[1]
            crop_height = 300
            crop_width = 300
            hcrop = height // crop_height
            wcrop = width // crop_width

            for i in range(hcrop):
                for j in range(wcrop):
                    crop = image[i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width]
                    crop = Image.fromarray(crop)
                    crop = data_transform(crop).to(device)
                    crop = crop[None, :]

                    outputs = model(crop)
                    _, preds = torch.max(outputs, 1)
                    softmaxed_output = softmax(outputs)
                    expected_output = torch.sum(softmaxed_output * rank_tensor, 1)
                    expected_output = expected_output[None]
                    expected_output = torch.transpose(expected_output, 0, 1)

                    crop_dict['predicted_score'].append(expected_output.item())

            whole_bee_dict['predicted_score'].append(np.mean(crop_dict['predicted_score']))

            print('Predicted rating w/o surface area: ', np.mean(crop_dict['predicted_score']))
            print('Predicted_rating w/ surface area: ',
                  np.mean(crop_dict['predicted_score']) / float(surf_percent.strip('%')) * 100)

            # convert image to grayscale
            im_gray = rgb2gray(im)
            entropy_image = entropy(im_gray, disk(6))
            # exclude zeros in the matrix
            entropy_image_1 = entropy_image[entropy_image != 0]
            whole_bee_dict['Entropy_mean'].append(np.mean(entropy_image_1) / float(surf_percent.strip('%')) * 100)

