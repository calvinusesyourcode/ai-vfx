import cv2
import numpy as np

def increase_alpha(image_path, alpha_increase):
    # Read an RGBA image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Increase alpha values for non-zero alpha pixels
    non_zero_alpha_indices = image[:, :, 3] > 1
    image[non_zero_alpha_indices, 3] = np.clip(image[non_zero_alpha_indices, 3] + alpha_increase, 0, 255)

    # Save the modified image
    cv2.imwrite(image_path, image)

# Usage
increase_alpha("out.png", 65)  # Increase alpha by 50