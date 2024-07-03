'''
Name: Kathryn Chen
Date: October 6, 2023
'''

import albumentations as A
from albumentations.pytorch import ToTensorV2
from classes import *
from functions import preprocess_mask, predict, resize_predictions, count_surface_area


'''
Calculates the "surface area" of hair on a bee by counting the number of segmented bee pixels, the number of segmented 
hair pixels, and the ratio of segmented hair to segmented bee pixels to get an idea of what percent of the bee's surface area 
is covered by hair.
Keep in mind that since the bee masks segment out the eyes, wings, antennae, and tongues, the surface area of these parts of the bee
are not included in any surface area calculations. This is because we are not interested in measuring any hair that may exist on 
those regions, so we excluded them from the total area of the bee's body that we want to measure hair on.   
'''
def calculate_surface_area_main():
    root = os.getcwd()
    parent = os.path.abspath(os.path.join(root, os.pardir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = os.path.join(parent, 'analysis_results')
    bee_images_directory = os.path.join(parent, 'bee_original')
    artificial_bees_directory = os.path.join(parent, 'artificial_bees')
    bee_masks_directory = os.path.join(parent, 'predicted_bee_masks')
    hair_masks_directory = os.path.join(parent, 'predicted_hair_masks')

    bee_images = os.listdir(bee_images_directory)
    artificial_bees = os.listdir(artificial_bees_directory)
    bee_masks = os.listdir(bee_masks_directory)
    hair_masks = os.listdir(hair_masks_directory)

    bee_model = torch.load(parent + '/models/New_Bee_Model', map_location=device).to(device)
    hair_model = torch.load(parent + '/models/Hair_model', map_location=device).to(device)

    params = {
        "device": device,
        "lr": 0.001,
        "batch_size": 16,
        "num_workers": 4,
        "epochs": 15,
    }

    image_transform = A.Compose(
        [A.Resize(256, 256),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()]
    )

    bee_dataset = BeeInferenceDataset(bee_images, bee_images_directory, image_transform=image_transform)
    hair_dataset = BeeInferenceDataset(artificial_bees, artificial_bees_directory, image_transform=image_transform)

    # If there are no bee masks already made, then use the bee dataset to create a list of predicted masks.
    if len(bee_masks) == 0:
        bee_predictions = predict(bee_model, params, bee_dataset)
        bee_predicted_masks = resize_predictions(bee_predictions, bee_dataset)
    else:
        bee_predicted_masks = []
        for i in bee_masks:
            path = os.path.join(bee_masks_directory, i)
            image = cv2.imread(path)
            mask = preprocess_mask(image)
            bee_predicted_masks.append(mask)


    # If there are no hair masks already made, then use the hair dataset to create a list of predicted masks.
    if len(hair_masks) == 0:
        hair_predictions = predict(hair_model, params, hair_dataset)
        hair_predicted_masks = resize_predictions(hair_predictions, hair_dataset)
    else:
        hair_predicted_masks = []
        for i in hair_masks:
            path = os.path.join(hair_masks_directory, i)
            image = cv2.imread(path)
            mask = preprocess_mask(image)
            hair_predicted_masks.append(mask)

    # Set load = True if you want to add to an existing csv.
    count_surface_area(bee_predicted_masks, hair_predicted_masks, dataset = bee_dataset,
                       save_path = os.path.join(results, 'surface_areas.csv'),
                       load_path = os.path.join(results, 'surface_areas.csv'),
                       load=False)