'''
Name: Kathryn Chen
Name: Luning Ding
Name: Nicholas Alexander
Date: June 23, 2023
'''

import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from paths import *
from PIL import Image
import torch.nn as nn

'''
By Nicholas Alexander.
Divides mask pixel values by 255 so the values are between 0 and 1,
then sets all pixel values greater than 0.5 to 1 and all values less  
than 0.5 to 0.
'''
def preprocess_mask(input_mask):
  test_mask = input_mask / 255
  test_mask[test_mask > 0.5] = 1
  test_mask[test_mask <= 0.5] = 0
  return test_mask


# An exception raised when the shapes of the image and corresponding mask do not match.
class ShapeException(Exception):
  pass

'''
Modified from a Pytorch dataset created by Nicholas Alexander.
Creates a custom Pytorch dataset for the bee and hair pipelines.
Class methods:
- getsize returns the original size of a specified image
- getname returns the original name of a specified image
- __getitem__ is overwritten to transform the images and masks into numpy arrays, 
  apply transformations if relevant, and return the following images and masks.
'''
class BeeInferenceDataset(Dataset):
  def __init__(self, images_filenames, images_directory, masks_directory=None, image_transform=None,
               mask_transform=None):
    self.images_filenames = images_filenames
    self.images_directory = images_directory
    self.masks_directory = masks_directory
    self.image_transform = image_transform
    self.mask_transform = mask_transform

  def __len__(self):
    return len(self.images_filenames)

  def getsize(self, idx):
    image_filename = self.images_filenames[idx]
    image = cv2.imread(os.path.join(self.images_directory, image_filename))
    original_size = tuple(image.shape[:2])
    return original_size

  def getname(self, idx):
    image_filename = self.images_filenames[idx]
    return image_filename

  def __getitem__(self, idx):
    image_filename = self.images_filenames[idx]
    try:
      image = cv2.imread(os.path.join(self.images_directory, image_filename))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if self.image_transform is not None:
        transformed_img = self.image_transform(image=image)
        image = transformed_img["image"]
        image = image.float()

      if self.masks_directory:
        mask = cv2.imread(os.path.join(self.masks_directory, image_filename))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.mask_transform is not None:
          transformed_mask = self.mask_transform(image=mask)
          mask = preprocess_mask(transformed_mask["image"])
          mask = mask.float()
      else:
        mask = torch.zeros((3, 256, 256))

      return image, mask
    except Exception as e:
      print(self.getname(idx))
      print("An exception occurred: ", e)
      return torch.zeros((3, 256, 256)), torch.zeros((3, 256, 256))

'''
This is used to create test datasets only, when crop = True.
Creates a custom Pytorch dataset for the hair pipelines.
Class methods:
- getsize returns the original size of a specified image
- getname returns the original name of a specified image
- collate_fn returns the collate function of the class
- crop divides an image into 300 x 300 crops, with the remaining ends 256 x 256 pixels at the smallest,
  and appends the crops into a nested list that creates a new list for each row of crops, in which all
  the crops in that row are stored in that list. 
- __getitem__ is overwritten to call the crop method on the input image and, if the masks exist, the mask,
  and applies the corresponding transformations on the lists of image and mask crops, before returning the
  lists of modified images and crops.
'''
class CroppedDataset(Dataset):
  def __init__(self, images_filenames, images_directory, masks_directory=None, image_transform=None,
               mask_transform=None, has_masks=False, crop_height=300, crop_width=300):
    self.images_filenames = images_filenames
    self.images_directory = images_directory
    self.masks_directory = masks_directory
    self.image_transform = image_transform
    self.mask_transform = mask_transform
    self.has_masks = has_masks
    self.crop_height = crop_height
    self.crop_width = crop_width

  def __len__(self):
    return len(self.images_filenames)

  def getsize(self, idx):
    image_filename = self.images_filenames[idx]
    image = cv2.imread(os.path.join(self.images_directory, image_filename))
    original_size = tuple(image.shape[:2])
    return original_size

  def getname(self, idx):
    image_filename = self.images_filenames[idx]
    return image_filename

  def collate_fn(self, batch):
    return batch

  def crop(self, image):
    crops = []
    crop_height = self.crop_height
    crop_width = self.crop_width

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

  def __getitem__(self, idx):
    image_filename = self.images_filenames[idx]
    image = cv2.imread(os.path.join(self.images_directory, image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crops = self.crop(image)
    croplist = []

    for i in crops:
      row = []
      for j in i:
        if self.image_transform is not None:
          transformed_img = self.image_transform(image=j)
          j = transformed_img["image"]
        row.append(j)
      croplist.append(row)

    if self.has_masks:
      mask = cv2.imread(os.path.join(self.masks_directory, image_filename))
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      maskcrops = self.crop(mask)
      maskcroplist = []

      for i in maskcrops:
        maskrow = []
        for j in i:
          if self.mask_transform is not None:
            transformed_mask = self.mask_transform(image=j)
            mask = preprocess_mask(transformed_mask["image"])
            j = mask.float()
          maskrow.append(j)
        maskcroplist.append(maskrow)

      return croplist, maskcroplist

    else:
      return croplist

'''
Name: Luning Ding
Date: July 5, 2023
'''

class HairinessRatingDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    # Arguments:
    #     csv_file (string): Path to the csv file with annotations.
    #     root_dir (string): Directory with all the images.
    #     transform (callable, optional): Optional transform to be applied
    #         on a sample.
    self.rating_frame = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.rating_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_name = os.path.join(self.root_dir,
                            self.rating_frame.iloc[idx, 0])
    image = Image.open(img_name)
    rating = self.rating_frame.iloc[idx, 6]
    rating = np.array([np.float32(rating)])
    bin_class_label = self.rating_frame.iloc[idx, 12]
    # bin_class_label = np.array([np.float32(bin_class_label)])
    # bin_class_label = np.int32(bin_class_label)

    if self.transform:
      image = self.transform(image)
      rating = torch.from_numpy(rating)
      # bin_class_label = torch.from_numpy(bin_class_label)
    # [0-1] is 0, (1-2] is 1, ... , (4-5] is 4
    # if rating != 0:
    #     bin_class_label = math.floor(rating - 0.000001)
    # else:
    #     bin_class_label = 0
    # image, rating, and rating class bin
    sample = {'name': img_name, 'image': image, 'rating': rating, 'class': bin_class_label}
    # sample = ToTensor(sample)
    return sample


# Define the ResNet model
class ResNet(nn.Module):
  def __init__(self, config, output_dim):
    super().__init__()

    block, n_blocks, channels = config
    self.in_channels = channels[0]

    assert len(n_blocks) == len(channels) == 4

    self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
    self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
    self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
    self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(self.in_channels, output_dim)

  def get_resnet_layer(self, block, n_blocks, channels, stride=1):

    layers = []

    if self.in_channels != block.expansion * channels:
      downsample = True
    else:
      downsample = False

    layers.append(block(self.in_channels, channels, stride, downsample))

    for i in range(1, n_blocks):
      layers.append(block(block.expansion * channels, channels))

    self.in_channels = block.expansion * channels

    return nn.Sequential(*layers)

  def forward(self, x):

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    h = x.view(x.shape[0], -1)
    x = self.fc(h)

    # return x, h
    return x


class BasicBlock(nn.Module):

  expansion = 1

  def __init__(self, in_channels, out_channels, stride=1, downsample=False):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                           stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.relu = nn.ReLU(inplace=True)

    if downsample:
      conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                       stride=stride, bias=False)
      bn = nn.BatchNorm2d(out_channels)
      downsample = nn.Sequential(conv, bn)
    else:
      downsample = None

    self.downsample = downsample

  def forward(self, x):

    i = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)

    if self.downsample is not None:
      i = self.downsample(i)

    x += i
    x = self.relu(x)

    return x


class Bottleneck(nn.Module):

  expansion = 4

  def __init__(self, in_channels, out_channels, stride=1, downsample=False):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                           stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                           stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1,
                           stride=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

    self.relu = nn.ReLU(inplace=True)

    if downsample:
      conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1,
                       stride=stride, bias=False)
      bn = nn.BatchNorm2d(self.expansion * out_channels)
      downsample = nn.Sequential(conv, bn)
    else:
      downsample = None

    self.downsample = downsample

  def forward(self, x):

    i = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.bn3(x)

    if self.downsample is not None:
      i = self.downsample(i)

    x += i
    x = self.relu(x)

    return x