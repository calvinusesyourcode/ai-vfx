import os
import numpy as np
import cv2, time
from PIL import Image, ImageFont, ImageDraw

def ascii_filter_1(fontsize=8, image_path='', showResult=False, chroma_ref_path='', chroma_threshold=30):
    start_time = time.time()
    if fontsize < 5:
        print("WARNING: Font-Size is too small. This may result in an infinite loop. Enter any key to continue.")
        _ = input()

    CHAR_MAP = [c for c in reversed("""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|(1{[?-_+~<i!lI;:,"^`'.""")]

    OUTPUT_PATH = os.path.join(*os.path.split(image_path)[:-1], f"ascii_{os.path.split(image_path)[-1]}")

    FONT_PATH = os.path.join("secret_code.ttf")
    OUTPUT_WINDOW_NAME = "ASCIIfied"

    image = Image.open(image_path).convert('RGBA')
    image = np.array(image)
    image_height, image_width, _ = image.shape
    
    # Extracting the alpha channel
    alpha_channel = image[:,:,3]

    # Load the chroma reference image if provided
    if chroma_ref_path:
        chroma_ref_image = cv2.imread(chroma_ref_path)
        chroma_ref_pixel = chroma_ref_image[0, 0]  # Get the color of the first pixel
        chroma_ref_pixel = chroma_ref_pixel.astype(np.int32)  # Convert to int32 for calculations

    font = ImageFont.truetype(FONT_PATH, fontsize)
    l, t, font_width, font_height = font.getbbox("@")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    output = np.zeros((image_height, image_width, 4), dtype=np.uint16, order='C')
    output[..., 3] = 0

    pillow_output = Image.fromarray(output, 'RGBA')  # Note the 'RGBA'
    pillow_drawer = ImageDraw.Draw(pillow_output)

    for i in range(int(image_height / font_height)):
        for j in range(int(image_width / font_width)):
            y_start = i * font_height
            x_start = j * font_width
            x_end = x_start + font_width
            y_end = y_start + font_height

            # Check alpha channel to decide whether to skip this pixel
            avg_alpha = np.mean(alpha_channel[y_start:y_end, x_start:x_end])
            if avg_alpha == 0:
                continue

            i1 = np.mean(hsv_image[y_start:y_end, x_start:x_end, 1])
            i2 = np.mean(hsv_image[y_start:y_end, x_start:x_end, 2])
            intensity = (i1 + i2) / 2
            position = int(intensity * len(CHAR_MAP) / 360)

            color = np.mean(rgb_image[y_start:y_end, x_start:x_end], axis=(0, 1)).astype(np.uint16)
            if np.all(color <= [10, 10, 10]):
                continue  # Skip further processing for black regions

            # Check chroma threshold and ignore pixels that closely match the chroma reference color
            if chroma_ref_path and np.all(np.abs(chroma_ref_pixel - color) <= chroma_threshold):
                continue

            text = str(CHAR_MAP[position])
            if text == ",":
                print(color)
            pillow_drawer.text((x_start, y_start), text, font=font, fill=(color[0], color[1], color[2], 255))



    output = np.array(pillow_output)
    output_rgba = np.concatenate([output[:, :, :3], output[:, :, 3:4]], axis=2).astype(np.uint16)
    non_zero_alpha_indices = output_rgba[:, :, 3] > 0
    output_rgba[non_zero_alpha_indices, 3] = np.clip(output_rgba[non_zero_alpha_indices, 3] * (fontsize * 3/6 + 4), 0, 255)
    cv2.imwrite(OUTPUT_PATH, output_rgba.astype(np.uint8))
    if showResult:
        cv2.imshow("hi",output_rgba)
        time.sleep(3)
    # os.remove(image_path)
    # return output_rgba.astype(np.uint8)
    print(f"ascii took: {time.time()-start_time}s")

def ascii_filter_2(rgba_array, fontsize=8):
    start_time = time.time()
    if fontsize < 5:
        print("WARNING: Font-Size is too small. This may result in an infinite loop. Enter any key to continue.")
        _ = input()

    CHAR_MAP = [c for c in reversed("""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|(1{[?-_+~<i!lI;:,"^`'.""")]
    FONT_PATH = os.path.join("secret_code.ttf")

    image = rgba_array
    image_height, image_width, _ = image.shape
    
    alpha_channel = image[:,:,3]

    font = ImageFont.truetype(FONT_PATH, fontsize)
    l, t, font_width, font_height = font.getbbox("@")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    output = np.zeros((image_height, image_width, 4), dtype=np.uint16, order='C')
    output[..., 3] = 0

    pillow_output = Image.fromarray(output, 'RGBA')  # Note the 'RGBA'
    pillow_drawer = ImageDraw.Draw(pillow_output)

    for i in range(int(image_height / font_height)):
        for j in range(int(image_width / font_width)):
            y_start = i * font_height
            x_start = j * font_width
            x_end = x_start + font_width
            y_end = y_start + font_height

            # Check alpha channel to decide whether to skip this pixel
            avg_alpha = np.mean(alpha_channel[y_start:y_end, x_start:x_end])
            if avg_alpha == 0:
                continue

            i1 = np.mean(hsv_image[y_start:y_end, x_start:x_end, 1])
            i2 = np.mean(hsv_image[y_start:y_end, x_start:x_end, 2])
            intensity = (i1 + i2) / 2
            position = int(intensity * len(CHAR_MAP) / 360)

            color = np.mean(rgb_image[y_start:y_end, x_start:x_end], axis=(0, 1)).astype(np.uint16)
            if np.all(color <= [10, 10, 10]):
                continue  # Skip further processing for black regions


            text = str(CHAR_MAP[position])
            # if text == ",":
            #     print(color)
            pillow_drawer.text((x_start, y_start), text, font=font, fill=(color[0], color[1], color[2], 255))



    output = np.array(pillow_output)
    output_rgba = np.concatenate([output[:, :, :3], output[:, :, 3:4]], axis=2).astype(np.uint16)
    non_zero_alpha_indices = output_rgba[:, :, 3] > 0
    output_rgba[non_zero_alpha_indices, 3] = np.clip(output_rgba[non_zero_alpha_indices, 3] * (fontsize * 3/6 + 4), 0, 255)
    
    print(f"ascii took: {time.time()-start_time}s")
    return output_rgba.astype(np.uint8)

