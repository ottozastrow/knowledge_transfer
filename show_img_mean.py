import glob
import os
import random

import matplotlib
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tqdm


if('DISPLAY' not in os.environ.keys()):
    matplotlib.use('agg')

###
# output from train_dataset:
# mean r: 102.43793473080285
# mean_g: 170.1677151882308
# mean_b: 167.4679635012596

# ouput from test_dataset
# mean r: 79.2432085194694
# mean_g: 162.91863773144644
# mean_b: 159.5172805140004
###

# img_file_path = "../image-analysis/dataset
# _current/dataset_own_v9.1/images_train/"
img_file_path = "/home/zastrow-marcks/mag/barcodes/yolo/datasets/"\
    "localization/train_images/**.tif"
width, height = (224, 224)
filenames = list(glob.iglob(img_file_path, recursive=True))
means_r = []
means_g = []
means_b = []
counter = 0
min_r, max_r = ([], [])
for f in tqdm.tqdm(filenames):
    print(f)
    img = cv2.imread(f, 1)

    means_r.append(np.mean(img[:,:,0]))
    means_g.append(np.mean(img[:,:,1]))
    means_b.append(np.mean(img[:,:,2]))

    #sub_mean
    # img = cv2.resize(img, ( width , height ))

    img = img.astype(np.float32)
    # r= np.mean(img[:,:,0])
    # g= np.mean(img[:,:,1])
    # b= np.mean(img[:,:,2])
    # min_r.append(np.min(img))
    # max_r.append(np.max(img))
    # print(min_r[-1], max_r[-1])
    # print(means_r[-1])

    # print(means_g[-1])
    # print(means_b[-1])
    # img = img/255 + 0.5

    # min_r.append(np.min(img))
    # max_r.append(np.max(img))
    # print(min_r[-1], max_r[-1])
    # plt.imshow(img)
    # plt.show()
    counter += 1
    if counter > 300:
        break

mean_r = np.mean(np.array(means_r))
mean_g = np.mean(np.array(means_g))
mean_b = np.mean(np.array(means_b))
print("mean r:", mean_r)
print("mean_g:", mean_g)
print("mean_b:", mean_b)
