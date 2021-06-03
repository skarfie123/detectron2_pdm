# from PIL import Image
import glob
import cv2
import os
import sys

input_folder = sys.argv[1]  # first commandline argument sets the original images folder
for input_mask in sys.argv[2:]:  # rest of command line input is list of mask images
    os.makedirs(input_folder + "\\" + input_mask.split(".")[0], exist_ok=True)

    for filename in glob.glob(input_folder + "\*.jpg"):  # assuming jpg

        image = cv2.imread(filename)
        mask = cv2.imread(input_mask)

        # Mask input image with binary mask
        result = cv2.bitwise_and(image, mask)
        # Color background white i.e. white mask
        # result[mask == 0] = 255  # Optional

        newFile = (
            input_folder
            + "\\"
            + input_mask.split(".")[0]
            + filename.split(input_folder)[1]
        )
        print(newFile)
        cv2.imwrite(newFile, result)
