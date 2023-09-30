import cv2
import os

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
        frame = cv2.imread(img_path)
        out.write(frame)
        os.remove(img_path)

    out.release()
    print(f"Video {output_video_name} has been created.")

