import os
from functions import make_crops, pick_random_crops

root = os.getcwd()
image_path = os.path.join(root, 'artificial_bees')
crop_path = os.path.join(root, 'bee_crops')

make_crops(image_path, crop_path, 300, 300)

pick_random = True

# Used if you can not use all the bee crops, e.g. can not make that many masks,
# and only want a certain number of randomly chosen crops from that set of crops.
if pick_random:
    random_crops_path = os.path.join(root, 'random_bee_crops')
    pick_random_crops(crop_path, random_crops_path, size = 250)