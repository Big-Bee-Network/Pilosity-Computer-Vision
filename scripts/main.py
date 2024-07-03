'''
Name: Kathryn Chen
Name: Luning Ding
Date: June 23, 2023
'''

import os
import sys
import torchvision.transforms as transforms
from classes import Bottleneck, ResNet
from remove_background import remove_background_main
from make_folders import make_folders_main
from segment_bee import segment_bee_main
from artificial_bees import artificial_bees_main, artificial_hair_main
from segment_hair import segment_hair_main
from entropy_analysis import entropy_analysis
from image_regression import image_regression, predicted_rating_entropy_values, predicted_rating_entropy_surface_area
from calculate_brightness import calculate_brightness_main
from calculate_surface_area import calculate_surface_area_main

# Set to True if you want to artificially remove the background of your images before using them for segmentation.
remove_background = False
if len(sys.argv) > 1:
    if sys.argv[1] == True or sys.argv[1] == False:
        remove_background = sys.argv[1]

if remove_background:
    remove_background_main()

root = os.getcwd()
parent = os.path.abspath(os.path.join(root, os.pardir))
data_transform = transforms.Compose([
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1, 1, 1])
    ])

make_folders_main()
segment_bee_main(background_removed = remove_background, to_train = False)
artificial_bees_main(remove_background)
segment_hair_main(to_crop = True, to_train = False)
artificial_hair_main()
calculate_brightness_main()
calculate_surface_area_main()

image_folder_path = os.path.join(parent, 'random_artificial_bee_crops')
# Defined for predicted_rating_entropy_surface_area() specifically
image_folder_path_2 = os.path.join(parent, 'bee_original')
entropy_output_path = os.path.join(parent, 'entropy_images')
entropy_values = os.path.join(parent, 'entropy_analysis', 'entropy_values_for_artificial_bees.csv')
ground_truth_hairiness_rating = os.path.join(parent, 'image_regression', 'ground_truth_rating.csv')
resnet_model_save = os.path.join(parent, 'models/state_dict_model_train_22bins_new.pt')
predicted_rating = os.path.join(parent, 'image_regression', 'predicted_rating.csv')
surface_area = os.path.join(parent, 'analysis_results', 'surface_areas.csv')

'''
Entropy Analysis

Optional functions to test on a single image:
entropy_mask_viz(image)
entropy_analysis_test(image_path)
'''

entropy_analysis(image_folder_path, entropy_output_path, entropy_values)

'''
Image regression
IMPORTANT NOTE
Do not run image_regression() unless you have a csv file that contains ground truth hairiness ratings of the bee images you are attempting to give hairiness scores to.
'''

image_regression(ground_truth_hairiness_rating, image_folder_path, resnet_model_save)
predicted_rating_entropy_values(ground_truth_hairiness_rating, image_folder_path, resnet_model_save, predicted_rating, data_transform = data_transform)
predicted_rating_entropy_surface_area(ground_truth_hairiness_rating, resnet_model_save, image_folder_path_2, image_folder_path, surface_area, data_transform = data_transform)
