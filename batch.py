import os, cv2, typing, subprocess
import numpy as np
import matplotlib.pyplot as plt
from ascii import ascii_filter_2, ascii_filter_1
from PIL import Image

def png_to_video(png_folder, output_video_name, fps=30):
    # Get all files from the folder
    images = [img for img in os.listdir(png_folder) if img.endswith(".png")]
    
    # Sort the images by name
    images.sort()

    # Read the first image to get the dimensions
    img_path = os.path.join(png_folder, images[0])
    frame = cv2.imread(img_path)
    height, width, layers = frame.shape

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for mp4 format
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # Loop through images and add to video
    for i in range(len(images)):
        img_path = os.path.join(png_folder, images[i])
        # frame = cv2.imread(img_path)
        frame = ascii_filter_1(fontsize=16, image_path=img_path)
        out.write(frame)
        os.remove(img_path)

    out.release()
    print(f"Video {output_video_name} has been created.")

# png_to_video("pre", "out.mp4", fps=30)

def apply_filter_to_masks(parent_folder: str, filter: typing.Callable):
    for dir in os.listdir(parent_folder):
        i = 0
        if dir == "mask":
            for file in os.listdir(os.path.join(parent_folder, dir)):
                img_path = os.path.join(parent_folder, dir, file)
                image = Image.open(img_path).convert("RGBA")
                rgba_array = np.array(image)
                filtered_array = filter(rgba_array)
                filter_name = filter.__name__
                pretty_name = filter_name.split("_")[0] if "_" in filter_name else filter_name
                # plt.figure(figsize=(10,10))
                # plt.imshow(filtered_array)
                # plt.axis('off')
                # plt.show()

                output_dir = os.path.join(parent_folder, pretty_name)
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(f'{output_dir}/{pretty_name}_{file}', cv2.cvtColor(filtered_array, cv2.COLOR_RGBA2BGRA))
                # break

        # for img in os.listdir(os.path.join(dir, parent_folder))

def convert_mov_to_mp4(folder_path):
    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".MOV") or filename.endswith(".mov"):
            # Prepare the output filename
            output_filename = os.path.splitext(filename)[0] + ".mp4"
            output_filepath = os.path.join(folder_path, output_filename)

            # Prepare the input filepath
            input_filepath = os.path.join(folder_path, filename)

            # Run ffmpeg command
            command = [
                'ffmpeg',
                '-i', input_filepath,
                '-vf', 'fps=30',
                '-y',  # Overwrite output file if it exists
                output_filepath
            ]
            subprocess.run(command)


# convert_mov_to_mp4("video_in")
apply_filter_to_masks("output/IMG_4764", ascii_filter_2)
