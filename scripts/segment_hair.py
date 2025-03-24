'''
Name: Kathryn Chen
Date: June 23, 2023
'''

import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from paths import *
from classes import BeeInferenceDataset, CroppedDataset, GDL_BCE
from functions import train, predict, resize_predictions, display_image_grid, display_bees, calculate_accuracy, predict_crops, restitch_predictions

'''
to_crop determines if the images will be cropped before being segmented, then restitchced afterwards.
I would generally not recommend you set to_crop to False, as the model is pre-trained on image crops of 256 x 256 pixels.
to_train determines if the model will be trained on a set of images and masks before the model outputs predicted segmentations.
Only set to True if you have a set of images and corresponding masks for the model to train on.
Set the seed for not reproducibility.
Default loss option is "mixed," which uses the combined GDL and BCE loss. Set to "BCE" for only BCE loss.
'''
def segment_hair_main(to_crop = True, to_train = False, seed = 0, loss = "mixed"):
    results = os.path.join(parent, 'analysis_results')
    images_directory = os.path.join(parent, 'artificial_bees/')
    masks_directory = os.path.join(parent, 'original_hair_masks/')
    predicted_masks_directory = os.path.join(parent, 'predicted_hair_masks/')
    images = os.listdir(images_directory)
    masks = os.listdir(masks_directory)

    # shuffle images to increase likelihood of balanced train, val, and test datasets
    random.Random(seed).shuffle(images)

    train_count = int(0.6 * len(images))
    test_count = int(0.2 * len(images))
    val_count = len(images) - train_count - test_count
    train_images_filenames = images[:train_count]
    val_images_filenames = images[train_count:train_count + val_count]
    # test_images_filenames = images[train_count + val_count:]
    # If you are not training anything, test_images_filenames is the entire list of images
    test_images_filenames = images

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(parent + '/models/Hair_model', map_location=device).to(device)

    params = {
        "device": device,
        "lr": 0.001,
        "batch_size": 2,
        "num_workers": 0, # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.
        "epochs": 10,
    }

    # For the test loader and prediction functions specifically.
    test_params = {
        "device": device,
        "lr": 0.001,
        "batch_size": 16,
        "num_workers": 0, # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.
        "epochs": 10,
    }

    # The transformations done on the image dataset before training and prediction.
    image_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )

    # The transformations done on the mask dataset, if it exists, before training and prediction.
    mask_transform = A.Compose(
        [A.Resize(256, 256), ToTensorV2()]
    )

    if loss == "mixed":
        loss_fn = GDL_BCE()
    elif loss == "BCE":
        loss_fn = nn.BCELoss()
    else:
        raise ValueError("Loss parameter must be set to 'mixed' or 'BCE' only.")
    
    optimizer = optim.Adam(model.parameters(), lr = params['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 4)

    if to_train:
        # Creates Pytorch datasets of the training and validation images.
        train_dataset = BeeInferenceDataset(train_images_filenames, images_directory, masks_directory,
                                            image_transform=image_transform, mask_transform=mask_transform)
        val_dataset = BeeInferenceDataset(val_images_filenames, images_directory, masks_directory,
                                          image_transform=image_transform, mask_transform=mask_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, )

        train(model = model, batch_size = params['batch_size'], loss_fn = loss_fn,
              train_dataloader = train_dataloader, val_dataloader = val_dataloader,
              scheduler = scheduler, optimizer = optimizer, device = device, train_images = images, epochs = 15)

    if to_crop:
        # Creates a Pytorch dataset of the test images after dividing them into crops.
        bee_dataset = CroppedDataset(artificial_bees[:16], artificial_bees_directory, image_transform=image_transform)
        test_loader = DataLoader(bee_dataset, batch_size=test_params['batch_size'], shuffle=False, collate_fn=bee_dataset.collate_fn,
                                 pin_memory=True)

        # Crops the input images, segments the bee hair in the crops, then restitches the crops back into the original image.
        restitched_predictions = predict_crops(model, params, test_loader)

        # Resizes the 256 x 256 segmentations to their original size.
        restitched_predicted_masks = restitch_predictions(restitched_predictions, bee_dataset)

        # Slightly different grids are displayed based on whether there exist ground truth masks.
        if len(masks) == 0:
            display_bees(artificial_bees[:10], artificial_bees_directory, restitched_predicted_masks[:10], save = True, save_path = 'hair_predictions')
        else:
            display_image_grid(artificial_bees[:10], artificial_bees_directory, masks_directory, predicted_masks = restitched_predicted_masks[:10],
                               save = True, save_path = 'hair_predictions')
            # If the ground truth masks exist and you want the accuracy metrics
            prediction_accuracy_csv = os.path.join(results, 'hair_prediction_accuracies.csv')
            calculate_accuracy(restitched_predicted_masks, masks_directory=masks_directory, filenames=test_images_filenames,
                               csv_path = prediction_accuracy_csv)

    # to_crop = False
    else:
        # Creates a Pytorch dataset of the test images without dividing them into crops.
        test_dataset = BeeInferenceDataset(test_images_filenames, images_directory, image_transform=image_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)

        # Outputs the model's predicted segmentations of bee hair.
        predictions = predict(model, params, test_dataset, test_loader)

        # Resizes the 256 x 256 segmentations to their original size.
        predicted_masks = resize_predictions(predictions, test_dataset)

        # Slightly different grids are displayed based on whether there exist ground truth masks. Also accounts for if the entire test dataset is less than ten images.
        idx = min(10, len(test_images_filenames)) # Set the min to 10 so the predicted masks displayed are not too crowded with the number of masks shown
        if len(masks) == 0:
            display_bees(artificial_bees[:idx], artificial_bees_directory, predicted_masks[:idx], save = True, save_path = 'hair_predictions')
        else:
            display_image_grid(artificial_bees[:idx], artificial_bees_directory, masks_directory, predicted_masks = predicted_masks[:idx],
                               save = True, save_path = 'hair_predictions')
            # If the ground truth masks exist and you want the accuracy metrics
            prediction_accuracy_csv = os.path.join(results, 'hair_prediction_accuracies.csv')
            calculate_accuracy(predicted_masks, masks_directory=masks_directory, filenames=test_images_filenames,
                               csv_path = prediction_accuracy_csv)

