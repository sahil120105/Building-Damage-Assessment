import cv2
import numpy as np

# Load one of your new masks
img = cv2.imread("../data/processed/train_masks/guatemala-volcano_00000000_post_disaster.png", cv2.IMREAD_GRAYSCALE)

# Print unique values. You should see something like [0, 1, 2] 
# (0=Background, 1=No Damage, 2=Minor...)
print(f"Classes found in this image: {np.unique(img)}")