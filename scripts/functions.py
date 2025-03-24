'''
Name: Kathryn Chen
Name: Luning Ding
Name: Nicholas Alexander
Date: June 23, 2023
'''

from classes import *
from paths import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
import copy
import time
import os
import torch
import pandas as pd
from torchvision import utils
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from csv import writer
from collections import defaultdict
import glob
import tqdm.notebook as tqdm
from torch.autograd import Variable


'''
This file contains most of the functions that are used in this repository.
The functions that make the augmented images are located in make_augment_functions.py
'''

results = os.path.join(parent, 'analysis_results')

# Checks if each directory from the list of directories exists, and if not, creates the directory.
def make_directories(directory_list):
  for i in directory_list:
    path = os.path.join(parent, i)
    if not os.path.isdir(path):
      os.mkdir(path)
      print("Created new directory at: ", path)


'''
Calculates the average pixel value (brightness) of the bee pixels in an image, then saves each image name and corresponding pixel value to a csv.
Requires a folder or list of segmented masks to work. Does not include the pixel values of any non-bee pixel.
Set load = False if you are creating a new csv, and set load = True if you are adding more rows to an existing brightness_values csv.
This function is very time-intensive, so it may be necessary to divide up your lists of images and masks and run the function several times, with load set to True.  
'''
def calculate_brightness(images, masks, images_directory, csv_path = os.path.join(results, 'brightness_values.csv'),
                         load_csv_path = os.path.join(results, 'brightness_values.csv'), save = True, load = False):
  if load:
    try:
      df = pd.read_csv(load_csv_path, index_col = False)
    except:
      raise IOError("The csv filename does not exist. Make sure the filename of an existing csv is given for load_csv_path.")
  else:
    df = pd.DataFrame(columns = ["Filename", "Brightness Mean", "Brightness Median", "Brightness Standard Deviation"])

  for i in range(len(images)):
    image_path = os.path.join(images_directory, images[i])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = masks[i]
    mask = np.expand_dims(mask, axis = 2)

    artificial_image = image * mask
    non_black_pixels = (artificial_image.sum(axis = 2) > 0)
    final_image = artificial_image[non_black_pixels]

    mean = np.mean(final_image)
    median = np.median(final_image)
    sd = np.std(final_image)

    row = [[images[i], mean, median, sd]]
    row = pd.DataFrame(row, columns = ["Filename", "Brightness Mean", "Brightness Median", "Brightness Standard Deviation"])
    df = pd.concat([df, row], axis = 0, ignore_index = True)

  if save:
    df.to_csv(csv_path, index = False)

  return df


# Uses the predicted bee masks to artificially remove the eyes, wings, and antennae
# from the original bee images, then saves these modified images in a new folder.
def make_fake_bees(images_path, mask_names, masks_path, predicted_masks = None, save_path = os.path.join(parent, "artificial_bees")):
  if not os.path.isdir(save_path):
    os.mkdir(save_path)

  idx = 0

  for i in mask_names:
    im = plt.imread(os.path.join(images_path, i))
    if not masks_path:
      if not predicted_masks:
        raise TypeError("Either predicted_masks or masks_path must be not None.")
      else:
        mask = predicted_masks[idx]
        mask = mask[:, :, np.newaxis]
        mask = np.concatenate([mask] * 3, axis=-1)
    else:
      mask = plt.imread(os.path.join(masks_path, i))
      mask = preprocess_mask(mask)
      mask = mask[:, :, np.newaxis]
    if im.shape[:2] != mask.shape[:2]:
      print(np.shape(im), np.shape(mask))
      raise ShapeException("Images and masks are mismatched shapes: file " + i)
    fake_bee = im * mask
    fake_bee = fake_bee.astype('uint8')
    bee_reborn = cv2.cvtColor(fake_bee, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_path, i), bee_reborn)

    idx += 1


# Converts the transparent pixels in the bees with artificially removed backgrounds
# into black pixels, which are brings the images closer to what the model was pre-trained on.
def convert_to_black(image, pixel=130, degree=0.05):
  image = np.array(image)[:, :, :3]
  lower = pixel - pixel * degree
  upper = pixel + pixel * degree
  for i in range(len(image)):
    for j in range(len(image[i])):
      if all(lower <= k <= upper for k in image[i][j]):
        image[i][j] = [0, 0, 0]
      j += 1
    i += 1

  image = Image.fromarray(np.uint8(image))
  return image

# Creates a concatenated Pytorch Dataset of augmented values.
def make_augs(filenames, augs, image_transform, mask_transform):
  masterlist = []
  for x, y in augs:
    dset = BeeInferenceDataset(filenames, x, y, image_transform = image_transform, mask_transform = mask_transform)
    masterlist.append(dset)
  datasets = torch.utils.data.ConcatDataset(masterlist)
  return datasets


# Trains one epoch of a supervised segmentation model using the typical Pytorch training loop.
# Original code found here: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(model, loss_fn, train_dataloader, optimizer, device, batch_size, train_images):
    running_loss = 0
    batches = math.ceil(len(train_images) / batch_size)

    for i, data in enumerate(train_dataloader):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()

      outputs = model(inputs)
      outputs = torch.sigmoid(outputs)

      loss = loss_fn(outputs, labels)
      loss.backward()

      optimizer.step()
      running_loss += loss.item()

    last_loss = running_loss / batches

    return last_loss


# Trains a supervised segmentation model.
# Original code found here: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train(model, batch_size, loss_fn, train_dataloader, val_dataloader, scheduler, optimizer, device, train_images, epochs=20):
  epoch_number = 0
  for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(model, loss_fn, train_dataloader, optimizer, device, batch_size, train_images)

    model.eval()

    running_vloss = 0.0

    with torch.no_grad():
      for i, vdata in enumerate(val_dataloader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        voutputs = model(vinputs)
        voutputs = torch.sigmoid(voutputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    scheduler.step(avg_vloss)
    print('LOSS train {} val {}'.format(avg_loss, avg_vloss))

    epoch_number += 1


'''
This is a modified version of a function originally created by Nicholas Alexander.
Predicts the images of a test dataset using a model.
Zips the original image height and width along with the predicted crop, 
then adds that to a list of predictions that is returned at the end.
'''
def predict(model, params, dataset, dataloader = None):
  if dataloader == None:
    dataloader = DataLoader(dataset,
                            batch_size=params["batch_size"],
                            shuffle=False,
                            num_workers=params["num_workers"],
                            pin_memory=True)

  model.eval()
  predictions = []
  with torch.no_grad():
    i = 0
    for images, _ in dataloader:
      images = images.to(params["device"], non_blocking=True)
      output = model(images)
      probabilities = torch.sigmoid(output.squeeze(1))
      predicted_masks = (probabilities >= 0.5).float() * 1
      predicted_masks = predicted_masks.cpu().numpy()
      for predicted_mask in zip(
              predicted_masks
      ):
        original_heights, original_widths = dataset.getsize(i)
        predictions.append((predicted_mask, original_heights, original_widths))
        i += 1
  return predictions


'''
Only used if the hair segmentation process is done by cropping and restitching the images.
Predicts the images of a test dataset by cropping those images and segmenting the crops,
then re-appending those crops into a nested list in a way that allows the predicted crops
to be reconstructed into the original image.
'''
def predict_crops(model, params, dataloader):
  model.eval()
  predictions = []
  with torch.no_grad():
    for crops in dataloader:  # change to for crops, _ in test_loader if the test_loader has masks
      for i in crops:  # if batch_size is 1, this is the one crop list in the batch
        predicted_crops = []
        for j in i:  # one row in the crop list
          row = []
          for k in j:  # one image in a row in the crop list
            images = k.unsqueeze(0)
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_mask = (probabilities >= 0.5).float() * 1
            predicted_mask = predicted_mask.cpu().numpy()
            row.append(predicted_mask)
          predicted_crops.append(row)
        predictions.append(predicted_crops)

  return predictions


'''
This is a modified version of code originally written by Nicholas Alexander.
Used to resize the predicted images back to their original sizes.
If save = True, it also saves the predicted images to the designated save path.
'''
def resize_predictions(predictions, dataset, save=True, save_path=os.path.join(parent, 'predicted_bee_masks/')):
  i = 0
  predicted_masks = []
  for predicted_256x256_mask, original_height, original_width in predictions:
    mask = np.transpose(predicted_256x256_mask, (1, 2, 0))
    full_sized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    predicted_masks.append(full_sized_mask)
    if save:
      name = dataset.getname(i)
      save_name = os.path.join(save_path, name)
      if not os.path.isdir(save_path):
        os.mkdir(save_path)
      cv2.imwrite(save_name, full_sized_mask * 255)
    i += 1

  return predicted_masks

'''
Only used if the hair segmentation process is done by cropping and restitching the images.
Takes a nested list of predicted crops and restitches them back into a full mask of the original image.
If save = True, it also saves the restitched predicted images to the designated save path.
'''
def restitch_predictions(predictions, test_dataset, save=True,
                         save_path=os.path.join(parent, 'predicted_hair_masks/')):
  restitched_images = []
  idx = 0
  for image in predictions:
    height, width = test_dataset.getsize(idx)
    restitch = Image.new('L', (width, height))

    starth = 0
    for i in range(len(image)):
      startw = 0
      croph = height - 300 * (len(image) - 1) if i == len(image) - 1 else 300
      for j in range(len(image[i])):
        cropw = width - 300 * (len(image[i]) - 1) if j == len(image[i]) - 1 else 300
        mask = np.transpose(image[i][j], (1, 2, 0))
        mask = cv2.resize(mask, (cropw, croph), interpolation=cv2.INTER_NEAREST)
        mask_image = Image.fromarray(mask)
        restitch.paste(mask_image, (startw, starth, startw + cropw, starth + croph))
        startw += cropw
      starth += croph

    restitch = np.array(restitch)
    restitch = restitch * 255
    restitched_images.append(restitch)

    if save:
      name = test_dataset.getname(idx)
      if not os.path.isdir(save_path):
        os.mkdir(save_path)
      save_name = os.path.join(save_path, name)
      cv2.imwrite(save_name, restitch)

    idx += 1

  return restitched_images


'''
This is a modified version of a function originally created by Nicholas Alexander.
Used to display the original image, original mask, and predicted mask in a grid.
Displays two columns if no ground truth masks.
Otherwise displays three: one for original images, one for ground truth masks, one for predicted masks.
Ideally is used to display ten sets of images/masks.
If save = True, this image grid will be saved to the save_path.
'''
def display_image_grid(images_filenames, images_directory, masks_directory = None, predicted_masks=None,
                       save = False, save_path = 'whole_bee_predictions', filetype='.png'):
  cols = 3 if masks_directory else 2
  rows = len(images_filenames)
  figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, int(rows * 2.4))) # Change these dimensions depending on your resolution needs (x and y values of each picture box size)
  for i, image_filename in enumerate(images_filenames):
    image = cv2.imread(os.path.join(images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax[i, 0].imshow(image)
    ax[i, 0].set_title("Image")
    ax[i, 0].set_axis_off()

    if masks_directory:
      mask = cv2.imread(os.path.join(masks_directory, image_filename))
      mask = preprocess_mask(mask)
      mask = mask[:, :, 0]

      ax[i, 1].imshow(mask, interpolation="nearest")

      ax[i, 1].set_title("Ground truth mask")

      ax[i, 1].set_axis_off()

    if predicted_masks:
      predicted_mask = predicted_masks[i]
      ax[i, cols - 1].imshow(predicted_mask, interpolation="nearest")
      ax[i, cols - 1].set_title("Predicted mask")
      ax[i, cols - 1].set_axis_off()
  plt.tight_layout()
  if save:
    filename = os.path.join(results, save_path)
    plt.savefig(filename + filetype, bbox_inches='tight')
    plt.close(figure)
  else:
    plt.show()


'''
Same as display_image_grid, but does not display the ground truth masks.
Use this function if you don't have ground truth masks.
Ideally is used to display ten sets of images/masks.
If save = True, this image grid will be saved to the save_path.
'''
def display_bees(images_filenames, images_directory, predicted_masks,
                 save = False, save_path = 'whole_bee_predictions', filetype = '.png'):
  cols = 2
  rows = len(images_filenames)
  figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24)) # Change these dimensions depending on your resolution needs (x and y values of each picture box size)
  for i, image_filename in enumerate(images_filenames):
    image = cv2.imread(os.path.join(images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predicted_mask = predicted_masks[i]
    ax[i, 0].imshow(image)
    ax[i, 1].imshow(predicted_mask, interpolation="nearest")

    ax[i, 0].set_title("Image")
    ax[i, 1].set_title("Predicted mask")

    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()

  plt.tight_layout()
  if save:
    filename = os.path.join(results, save_path)
    plt.savefig(filename + filetype, bbox_inches='tight')
    plt.close(figure)
  else:
    plt.show()


'''
Requires a set of predicted bee and predicted hair masks for a set of images. Counts the number of predicted bee pixels and 
predicted hair pixels, then takes the ratio of those pixel counts and saves the bee pixel counts of each image into a csv.
dataset is just used to get the filename of the image. Either the bee or hair dataset should work, assuming they are using 
the same base images.
save_path is the path that the csv file will be saved to if save = True.
load_path is the path that loads the csv file that will be added onto if load = True.
'''
def count_surface_area(bee_masks, hair_masks, dataset, save_path, load_path, save=True, load=False):
  if load:
    df = pd.read_csv(load_path, index_col=False)
  else:
    df = pd.DataFrame(columns=['name', 'bee_surface area', 'hair_surface_area', 'percentage of pixels'])
  for i in range(len(bee_masks)):
    name = dataset.images_filenames[i]
    bee_unique = np.unique(bee_masks[i], return_counts=True)
    hair_unique = np.unique(hair_masks[i], return_counts=True)
    if bee_unique[1][0] == np.size(bee_masks[i]):
      df.loc[len(df)] = [name, 0, 0, '0.0%']
    elif len(bee_unique[1]) == 2:
      bee_pixels = bee_unique[1][1]
      hair_pixels = hair_unique[1][1]
      percentage = hair_pixels / bee_pixels
      percentage = f'{str(percentage * 100)}%'
      row = [[name, bee_pixels, hair_pixels, percentage]]
      row = pd.DataFrame(row, columns=['name', 'bee_surface area', 'hair_surface_area', 'percentage of pixels'])
      df = pd.concat([df, row], axis=0, ignore_index=True)

  if save:
    df.to_csv(save_path, index=False)


# Originally written by Nicholas Alexander.
# Calculates the jaccard (intersection over union) value.
def jaccard(y_true, y_pred):
  intersection = (y_true * y_pred).sum()
  union = y_true.sum() + y_pred.sum() - intersection
  return (intersection + 1e-15) / (union + 1e-15)


# Originally written by Nicholas Alexander.
# Calculates the dice coefficient.
def dice(y_true, y_pred):
  return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

# Calculates the dice loss using the dice coefficient
# Must ensure y_true and y_pred are Pytorch tensors
def dice_loss(y_true, y_pred):
  return 1 - dice(y_true, y_pred)

# Calculates the loss as a mix of dice and BCE
# Must ensure y_true and y_pred are Pytorch tensors
def dice_bce(y_true, y_pred):
  loss = nn.BCELoss()
  bce = loss(y_true, y_pred)
  dice_ = dice_loss(y_true, y_pred)

  return 0.5 * (dice_ + bce)

# Originally written by Nicholas Alexander.
# Calculates the raw accuracy score of accurate pixels divided by the total pixels.
def accuracy(y_true, y_pred):
  return (((y_pred + y_true) == 2).sum() + ((y_pred + y_true) == 0).sum()) / y_true.size


# Calculates the accuracy, dice, and IoU values, then saves them to a csv file.
def calculate_accuracy(predicted_masks, masks_directory, filenames, csv_path):
  samples = []

  for i, mask in enumerate(filenames):
    mask_orig = cv2.imread(os.path.join(masks_directory, mask))
    mask_orig = preprocess_mask(mask_orig)
    mask_orig = mask_orig[:, :, 0]
    mask_pred = predicted_masks[i]

    acc = accuracy(mask_orig, mask_pred)
    iou = jaccard(mask_orig, mask_pred)
    f1 = dice(mask_orig, mask_pred)

    sample = {'Name': mask, 'Accuracy': acc, 'F1': f1, 'IoU': iou}
    samples.append(pd.DataFrame(data=sample, index=[mask]))

  results = pd.concat(samples)
  results.to_csv(csv_path, index = False)


# Takes two csv files with the same metric data types, filenames, and column names, and compares their respective metrics.
# Mainly used to compare the same metrics of two different model's segmentations on the same set of images.
def calculate_metric_differences(csv_1, csv_2, csv_path):
  df_1 = pd.read_csv(csv_1)
  df_2 = pd.read_csv(csv_2)
  equals = df_1["Name"].equals(df_2['Name'])
  if not equals:
    print("Either you have the wrong column name, or there are name mismatches at the following:")
    neq = df_1.loc[~(df_1["Name"] == df_2["Name"])]
    return neq["Name"]

  new_df = pd.DataFrame()
  new_df["Accuracy"] = df_1["Accuracy"] - df_2["Accuracy"]
  new_df["F1"] = df_1["F1"] - df_2["F1"]
  new_df["IoU"] = df_1["IoU"] - df_2["IoU"]

  new_df.to_csv(csv_path)

'''
This is a modified version of the background removal function.
The original function can be found in the background_removal_helper.py file of this github:
https://github.com/Schachte/Background-Removal-Utility/blob/master/background_removal_helper.py
'''
def background_removal_updated(input_dir, output_dir, to_remove):
  SEGMENTATION_NETWORK = "tracer_b7"
  PREPROCESSING_METHOD = "stub"
  POSTPROCESSING_METHOD = "fba"
  SEGMENTATION_MASK_SIZE = 512
  TRIMAP_DILATION = 8
  TRIMAP_EROSION = 8
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  BACKGROUND_COLOR = (0, 0, 0)  # Black background

  config = MLConfig(
    segmentation_network=SEGMENTATION_NETWORK,
    preprocessing_method=PREPROCESSING_METHOD,
    postprocessing_method=POSTPROCESSING_METHOD,
    seg_mask_size=SEGMENTATION_MASK_SIZE,
    trimap_dilation=TRIMAP_DILATION,
    trimap_erosion=TRIMAP_EROSION,
    device=DEVICE,
    background_color=BACKGROUND_COLOR
  )

  interface = init_interface(config)

  os.makedirs(output_dir, exist_ok=True)
  for file_name in to_remove:
    image_path = os.path.join(input_dir, file_name)
    image = Image.open(image_path)

    if image.mode != "RGB":
      image = image.convert("RGB")

    print(f"Beginning removal on {file_name}...")

    result = interface([image])
    result = convert_to_black(result[0])
    result_path = os.path.join(output_dir, file_name)
    result.save(result_path)

    print(f"Saved {file_name} to {result_path}")


'''
Crops every image in a folder into squares of the crop height and crop width, with any remaining pixels
at the end of a row/column of images added to the last image in that row/column. The crops are then saved
to the path specified by crop_folder.
'''
def make_crops(folder, crop_folder, crop_height, crop_width):

  if not os.path.exists(crop_folder):
    os.makedirs(crop_folder)

  for img in os.listdir(folder):
    img_path = os.path.join(folder, img)
    if img.endswith(".jpg") or img.endswith(".JPG"):
      image = cv2.imread(img_path)
      height, width = np.shape(image)[0], np.shape(image)[1]
      hcrop = height // crop_height
      wcrop = width // crop_width

      for i in range(hcrop):
        for j in range(wcrop):
          crop = image[i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width]
          flat = np.ndarray.flatten(crop)
          sum = np.sum(flat)
          if sum > 0:
            name = img[:-4] + str(i) + str(j) + img[-4:]
            cv2.imwrite(os.path.join(crop_folder, name), crop)


'''
Picks random crops from the folder_path to save to the destination folder.
The number of crops is determined by size.
clear is set to True if you want to clear the random_crop folder first before adding a new set of random crops.
'''
def pick_random_crops(folder_path, destination_folder, size=250, clear=False):
  if clear:
    files = glob.glob(destination_folder)
    for f in files:
      os.remove(f)

  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  folder = os.listdir(folder_path)
  numbers = np.random.choice(len(folder), size=size, replace=False).tolist()
  for i in numbers:
    img_name = os.path.join(folder_path, folder[i])
    if not os.path.isfile(destination_folder + '/' + img_name):
      image = cv2.imread(img_name)
      cv2.imwrite(os.path.join(destination_folder, folder[i]), image)
  print("Random crop selection is complete")


'''
Divides an image into 300 x 300 crops, with the remaining ends 256 x 256 pixels at the smallest,
and appends the crops into a nested list that creates a new list for each row of crops, in which all
the crops in that row are stored in that list.
'''
def crop(image, crop_height = 300, crop_width = 300):
  crops = []
  crop_height = crop_height
  crop_width = crop_width

  height, width = np.shape(image)[0], np.shape(image)[1]
  hcrop = height // crop_height
  wcrop = width // crop_width

  lasth = height - crop_height * (hcrop - 1)
  if lasth >= 556:
    hcrop += 1

  lastw = width - crop_width * (wcrop - 1)
  if lastw >= 556:
    wcrop += 1

  for i in range(hcrop):
    hcrops = []
    starth = i * crop_height
    endh = height if i == hcrop - 1 else (i + 1) * crop_height

    for j in range(wcrop):
      startw = j * crop_width
      endw = width if j == wcrop - 1 else (j + 1) * crop_width

      crop = image[starth:endh, startw:endw]
      hcrops.append(crop)

    crops.append(hcrops)

  return crops


'''
Counts how many images and their corresponding masks have mismatched sizes, and saves the names of any mismatched pairs,
along with their respective sizes, to a csv. Then prints the numbers of pairs with matching and not matching shapes.
The csv created is saved in the same folder that is being checked.
'''
def entire_folder_check(folder, image_path, mask_path, csv='unmatched_shapes.csv'):
  root = os.getcwd()
  same_shape = 0
  different_shape = 0
  df = pd.DataFrame(columns=['Path', 'Image shape', 'Mask shape'])
  for i in folder:
    im_path = os.path.join(image_path, i)
    masks_path = os.path.join(mask_path, i)
    im = plt.imread(im_path)
    masks = plt.imread(masks_path)
    if np.shape(im) != np.shape(masks):
      different_shape += 1
      df.loc[len(df)] = [i, np.shape(im), np.shape(masks)]
    else:
      same_shape += 1

  df.to_csv(os.path.join(root, csv))

  print(same_shape, different_shape)


'''
Use this to find the RGB mean and standard deviation of a list of images, specified by the images parameter, in the specified directory.
This can be used if you are attempting to normalize the specified directory's image means and standard deviations during preprocessing. 
Example: 
image_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
        )
The mean and std values used in A.Normalize above are determined by the following function.
'''
def find_normalization(directory, images):
  idx = 0
  image_paths = [os.path.join(directory, i) for i in images]
  means = np.zeros(shape = (len(images), 3))
  stds = np.zeros(shape = (len(images), 3))
  for img in image_paths:
    image = cv2.imread(img)
    try:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = image / 255
      mean = np.mean(image, axis = (0, 1))
      std = np.std(image, axis = (0, 1))
      means[idx] = mean
      stds[idx] = std
      idx += 1
    except:
      print(img)

  mu_rgb = np.mean(means, axis=0)
  std_rgb = np.mean(stds, axis=0)

  return mu_rgb, std_rgb

'''
Based off of code from: https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
An optional function that can be used to standardize dark background images to all have "true" black backgrounds.
Use only if attempting to correct minor lighting differences among images that should otherwise have the same background color.
'''
def adjust_background(images, original_directory, adjusted_directory, pixel_range_x = 10, pixel_range_y = 10):
  if not os.path.exists(adjusted_directory):
    os.makedirs(adjusted_directory)

  for img in images:
    image = cv2.imread(os.path.join(original_directory, img))
    shape = image.shape
    x = int(shape[0] * 0.02)
    y = int(shape[1] * 0.02)
    pixel = image[x, y, :]
    value = pixel[0]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v[v < value] = 0
    v[v >= value] -= value

    final_hsv = cv2.merge((h, s, v))
    new_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(adjusted_directory, 'adjusted_' + img), new_image)


'''
Housekeeping function, used to add lists of new train, validation, and test filenames to their respective csv files.
Specify the lists of filenames before using them as function arguments.
This function will most likely be used if you acquire more data and want them to add them to train/validation/test datasets.
In such situations, make sure the older filenames are included in the lists as well, or else they will not be saved to the csv files. 
'''
def add_new_filenames(train, val, test, train_csv = os.path.join(matl, 'train_images.csv'),
                      val_csv = os.path.join(matl, 'val_images.csv'),
                      test_csv = os.path.join(matl, 'test_images.csv')):
  train_df = pd.DataFrame(columns=['0'])
  train_df['0'] = train

  val_df = pd.DataFrame(columns=['0'])
  val_df['0'] = val

  test_df = pd.DataFrame(columns=['0'])
  test_df['0'] = test

  train_df.to_csv(train_csv)
  val_df.to_csv(val_csv)
  test_df.to_csv(test_csv)


# Housekeeping function, used to check which images don't have corresponding masks
def maskless(images, masks, path, csv = 'maskless_whole_bees.csv'):
  j = 0
  no_mask = []
  df = pd.DataFrame(columns = ['path'])
  for i in images:
    if i not in masks:
      pth = os.path.join(path, i)
      oof = plt.imread(pth)
      no_mask.append(i)
      j += 1
  df['path'] = no_mask
  df.head()
  print(no_mask)
  df.to_csv(os.path.join(matl, csv))


# Housekeeping function, used if you want to copy all images from the start_path directory to the destination_path directory.
def copy_images(start_path, destination_path):
  os.chdir(destination_path)

  start = os.listdir(start_path)
  end = os.listdir(destination_path)

  for i in start:
    if i not in end:
      pth = os.path.join(start_path, i)
      im = cv2.imread(pth)
      cv2.imwrite(i, im)

'''
Name: Luning Ding
Date: July 3, 2023
'''

'''
Entropy Anlaysis
'''

'''
Changing the initial radius of the disk:
1 ---  sharpness
9 ---  dullness
'''
def disk_iterations(image):
  image_gray = rgb2gray(image)
  f_size = 20
  radi = list(range(1, 10))
  fig, ax = plt.subplots(3, 3, figsize=(15, 15))
  for n, ax in enumerate(ax.flatten()):
    ax.set_title(f'Radius at {radi[n]}', fontsize=f_size)
    ax.imshow(entropy(image_gray, disk(radi[n])), cmap='magma')
    ax.set_axis_off()
  fig.tight_layout()


# This returns a figure of the input image at various entropy thresholds.
def threshold_checker(image):
  thresholds = np.arange(0.1, 1.1, 0.1)
  image_gray = rgb2gray(image)
  entropy_image = entropy(image_gray, disk(6))
  scaled_entropy = entropy_image / entropy_image.max()
  fig, ax = plt.subplots(2, 5, figsize=(17, 10))
  for n, ax in enumerate(ax.flatten()):
    ax.set_title(f'Threshold  : {round(thresholds[n], 2)}',
                   fontsize=16)
    threshold = scaled_entropy > thresholds[n]
    ax.imshow(threshold, cmap='gist_stern_r')
    ax.axis('off')
  fig.tight_layout()


# Tests on single image and choose the radius of the disk for entropy analysis
def entropy_analysis_test(image_path):
  im = imread(image_path)
  plt.figure(num=None, figsize=(8, 6), dpi=80)
  imshow(im)
  # convert image to grayscale
  im_gray = rgb2gray(im)
  plt.figure(num=None, figsize=(8, 6), dpi=80)
  imshow(im_gray)
  disk_iterations(im_gray)
  threshold_checker(im_gray)


'''
Performs entropy analysis on the input image, then turns two images:
a version of the image with only the entropy values greater than 0.8, and
a version of the image with only the entropy values less than 0.8.
'''
def entropy_mask_viz(image):
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(6))
    scaled_entropy = entropy_image / entropy_image.max()
    f_size = 24
    fig, ax = plt.subplots(1, 2, figsize=(17, 10))

    ax[0].set_title('Greater Than Threshold',
                    fontsize=f_size)
    threshold = scaled_entropy > 0.8
    image_a = np.dstack([image[:, :, 0] * threshold,
                         image[:, :, 1] * threshold,
                         image[:, :, 2] * threshold])
    ax[0].imshow(image_a)
    ax[0].axis('off')

    ax[1].set_title('Less Than Threshold',
                    fontsize=f_size)
    threshold = scaled_entropy < 0.8
    image_b = np.dstack([image[:, :, 0] * threshold,
                         image[:, :, 1] * threshold,
                         image[:, :, 2] * threshold])
    ax[1].imshow(image_b)
    ax[1].axis('off')
    fig.tight_layout()
    return [image_a, image_b]


'''
Loop over the entire folder, perform entropy analysis on each image in the folder using disk 6,
find the mean, median, and standard deviation of the image's entropy values,
save all of those values to a csv file, and save the entropy analysis images to another folder.
'''
def entropy_analysis_images(masked_image_path, entropy_output_path, csv_path = 'Entropy_Analysis_for_Bees.csv', save = True):
  for image_name in os.listdir(masked_image_path):
    im = imread(os.path.join(masked_image_path, image_name))
    # convert image to grayscale
    im_gray = rgb2gray(im)
    entropy_image = entropy(im_gray, disk(6))
    # exclude zeros in the matrix
    entropy_image_1 = entropy_image[entropy_image != 0]
    # write mean/median/standard deviations to csv file
    with open(csv_path, 'a') as f_object:
      # Pass this file object to csv.writer() and get a writer object
      writer_object = writer(f_object)

      # Pass the list as an argument into the writerow()
      writer_object.writerow(
        [image_name, np.mean(entropy_image_1), np.median(entropy_image_1), np.std(entropy_image_1)])

      # Close the file object
      f_object.close()

      if save:
        # save entropy images to output_path
        plt.imsave(entropy_output_path + image_name, entropy_image, cmap = 'magma')



'''
image regression
'''

# Helper function to show a batch and check if images are successfully loaded
def show_landmarks_batch(sample_batched, landmarks_batch = None):
    """Show image with landmarks for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch, padding=5, normalize=True, pad_value=1)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    print('rating:', [float(rating) for rating in sample_batched['rating'].numpy()])
    print('rating bin class', [float(class_label) for class_label in sample_batched['class'].numpy()])
    if landmarks_batch:
      for i in range(batch_size):
          plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                     landmarks_batch[i, :, 1].numpy() + grid_border_size,
                      s=10, marker='.', c='r')

          plt.title('Batch from dataloader')



# train the model
def train_hairiness_model(model, device, dataloaders, rank_tensor, dataset_sizes,
                          criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  best_mse = 10000
  result_dict = defaultdict(list)
  softmax = nn.Softmax(dim = -1).to(device)

  for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      print('start', phase, 'phase')
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()  # Set model to evaluate mode
      running_loss = 0.0
      running_corrects = 0
      running_rmse_loss = 0.0
      # Iterate over data.
      for data in tqdm.tqdm(dataloaders[phase]):
        images, ratings, labels = data['image'], data['rating'], data['class']
        inputs = images.to(device)
        labels = labels.to(device)
        ratings = ratings.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)

          loss1 = criterion(outputs, labels)

          # revised loss function
          softmaxed_output = softmax(outputs)
          expected_output = torch.sum(softmaxed_output * rank_tensor, 1)
          expected_output = expected_output[None]
          expected_output = torch.transpose(expected_output, 0, 1)

          # RMSE loss
          reg_criterion = nn.MSELoss().to(device)
          loss2 = torch.sqrt(reg_criterion(expected_output, ratings))

          loss = loss1 + loss2

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        running_rmse_loss += loss2.item() * inputs.size(0)

      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_rmse_loss = running_rmse_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      result_dict['epoch_loss'].append(epoch_loss)
      result_dict['epoch_rmse_loss'].append(epoch_rmse_loss)
      result_dict['epoch_acc'].append(epoch_acc.item())

      print(f'{phase} RMSE Loss: {epoch_rmse_loss:.2f} Total Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
      if phase == 'val' and epoch_rmse_loss < best_mse:
        best_mse = epoch_rmse_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val Acc: {best_acc:4f}')
  print(f'Best val RMSE: {best_mse:4f}')

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model, result_dict



def hairiness_rating(root_dir, model, device, rank_dict, data_transform):
  csv_path = os.path.join(results, 'surface_areas.csv')
  idx_tensor = [idx for idx in range(20)]
  idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)
  rank_list = []
  for i in idx_tensor:
    rank_list.append(rank_dict[i.item()])
  rank_tensor = Variable(torch.FloatTensor(rank_list)).to(device)

  surface_area_df = pd.read_csv(csv_path)
  imgname_df = surface_area_df['name']
  surface_area_df = surface_area_df['percentage of pixels']

  whole_bee_dict = defaultdict(list)
  softmax = nn.Softmax(dim = -1).to(device)

  with torch.no_grad():
    for imgname, surf_percent in zip(imgname_df, surface_area_df):
      figure(figsize=(4, 3), dpi=80)
      print(imgname)
      im = imread(root_dir + imgname)
      plt.imshow(im)
      plt.show()
      image = Image.open(root_dir + imgname)
      image = data_transform(image).to(device)
      image = image[None, :]
      outputs = model(image)
      _, preds = torch.max(outputs, 1)
      softmaxed_output = softmax(outputs)
      expected_output = torch.sum(softmaxed_output * rank_tensor, 1)
      expected_output = expected_output[None]
      expected_output = torch.transpose(expected_output, 0, 1)
      whole_bee_dict['predicted_score'].append(expected_output.item() / float(surf_percent.strip('%')) * 100)
      print('Predicted_rating w/ surface area: ', expected_output.item() / float(surf_percent.strip('%')) * 100)
      print()
      print()

      # convert image to grayscale
      im_gray = rgb2gray(im)
      entropy_image = entropy(im_gray, disk(6))
      # exclude zeros in the matrix
      entropy_image_1 = entropy_image[entropy_image != 0]
      whole_bee_dict['Entropy_mean'].append(np.mean(entropy_image_1) / float(surf_percent.strip('%')) * 100)


