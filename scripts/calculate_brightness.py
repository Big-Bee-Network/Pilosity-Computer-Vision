'''
Name: Kathryn Chen
Date: September 14, 2023
'''

from classes import *
from functions import predict, resize_predictions, calculate_brightness
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
IMPORTANT NOTE:
set load = True if you are adding more rows to an existing brightness_values csv.
Depending on your available hardware, this script may take up a lot of time and RAM, so it will likely be necessary to divide up your 
lists of images and masks and run the function several times. After running it the first time, change the load parameter from False to True.
This allows the calculate_brightness function to load and add on to the csv file created when the function was run the first time, 
instead of creating a new csv each time the script is run.
ex: calculate_brightness(images[40:80], predicted_masks[40:80], bee_images_directory, save = True, load = True)
The above calculates brightness for the 40th to 80th images in a folder. Adjust based on the number of images you have.

Calculates the brightness, which is measured by the average pixel value, of every pixel in an image that is considered to be a "bee" pixel.
This may mean that some pixels are inaccurately included or excluded from the calculations, but this is to help ensure that the background
pixels are overall left out of the calculations.
It is important to note that although the bee segmentation model used for most of the other scripts segments out the eyes, wings, 
antennae, and tongues, the model used for this script does not so that those parts of the bee can be included in the brightness calculation.
'''
def calculate_brightness_main():
    root = os.getcwd()
    parent = os.path.abspath(os.path.join(root, os.pardir))
    device = "cuda" if torch.cuda.is_available() else torch.device('cpu')
    results = os.path.join(parent, 'analysis_results')
    bee_images_directory = os.path.join(parent, 'bee_original/')
    images = os.listdir(bee_images_directory)
    model = torch.load(os.path.join(parent, 'models/New_Bee_Model'), map_location = device).to(device)
    params = {
        "device": device,
        "batch_size": 16,
        "num_workers": 0, # Change this value if you are running this on a computer/computing cluster that is capable of parallel processing.
    }

    image_transform = A.Compose(
        [A.Resize(256, 256),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()]
    )

    test_dataset = BeeInferenceDataset(images, bee_images_directory, image_transform=image_transform)
    predictions = predict(model, params, test_dataset)
    predicted_masks = resize_predictions(predictions, test_dataset, save=True, save_path = os.path.join(parent, 'predicted_bees_with_eyes_wings_antennae/'))

    calculate_brightness(images, predicted_masks, bee_images_directory,
                         csv_path=os.path.join(results, 'brightness_values.csv'),
                         load_csv_path=os.path.join(results, 'brightness_values.csv'),
                         save = True, load = False)