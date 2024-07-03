'''
Name: Luning Ding
Date: July 23, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import os
import cv2
from csv import writer
from paths import *
from functions import disk_iterations, threshold_checker


'''
Loop over the entire folder for artificial_bees using disk 6
Get the matrix of means, median, and standard deviations of entropy values of each artificial bee image
Output the entropy images to entropy_output_path
'''
def entropy_analysis(image_folder_path, entropy_output_path, csv_file_name):
  '''
  image_folder_path: the directory to the artificial bee images
  entropy_output_path: the directory to store the output entropy images
  csv_file_name: the name of the csv file that you want to store the entropy values to
  '''

  for image_name in os.listdir(image_folder_path):
    im = cv2.imread(os.path.join(image_folder_path, image_name))
    # convert image to grayscale
    im_gray = rgb2gray(im)
    entropy_image = entropy(im_gray, disk(6))
    # exclude zeros in the matrix
    entropy_image_1 = entropy_image[entropy_image != 0]
    # write mean/median/standard deviations to csv file
    with open(csv_file_name, 'a') as f_object:

      # Pass this file object to csv.writer()
      # and get a writer object
      writer_object = writer(f_object)

      # Pass the list as an argument into
      # the writerow()
      writer_object.writerow([image_name,np.mean(entropy_image_1), np.median(entropy_image_1),np.std(entropy_image_1)])

      # Close the file object
      f_object.close()

    # save entropy images to output_path
    plt.imsave(os.path.join(entropy_output_path, image_name), entropy_image, cmap = 'magma')