""" script takes json and tif file from a given directory and converts it for
neural network use.
The tifs are converted to jpgs and the jsons to ground truth masks in png."""
import argparse
import glob
import os

import cv2
import labelme
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", default="../tgb/annotate/raw_ann/",
                        help="input annotated directory")
    return parser.parse_args()


if __name__ == "__main__":
    # warning: only implemented for two class
    # case with class name a
    class_name_to_id = {'a': 1}
    counter = 0
    args = parse_args()
    output_dir = args.input_dir[:-1] + "_converted/"
    if os.path.exists(output_dir):
        raise Exception("output_dir already exists: " + output_dir)
    else:
        os.makedirs(output_dir + "images/")
        os.makedirs(output_dir + "annotations/")

    filenames = list(glob.iglob(args.input_dir + "**.json", recursive=True))
    for filename in tqdm.tqdm(filenames):
        imgfile = filename.replace(".json", ".tif")
        img = cv2.imread(imgfile)
        if img is None:
            print("image filename is:", imgfile)
            raise Exception("img not found")

        label_file = labelme.LabelFile(filename=filename)

        img = labelme.utils.img_data_to_arr(label_file.imageData)

        mask, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        mask = np.stack((mask,)*3, axis=-1)

        # check if valid images are being created (pixels with 0 or 1)
        # after 10 images stop checking as np.unique is slow
        if counter < 10:
            for c in range(3):
                un = np.unique(mask[:, :, c])
                assert(all(un == [0, 1]))
        counter += 1

        mask_filename = filename.replace(
            args.input_dir, output_dir + "images/")[:-4] + "png"
        image_filename = filename.replace(
            args.input_dir, output_dir + "annotations/")[:-4] + "jpg"
        cv2.imwrite(mask_filename, mask)
        cv2.imwrite(image_filename, img)
