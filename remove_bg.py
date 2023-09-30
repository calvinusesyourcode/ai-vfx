import torch
import torchvision
import sys
sys.path.append("..")
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import time
import subprocess
import json

def get_fps(input_path):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'json',
        input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"FFprobe command failed with error: {result.stderr}")
        return None
    
    info = json.loads(result.stdout)
    r_frame_rate = info['streams'][0]['r_frame_rate']
    fps = eval(r_frame_rate)
    return fps

def change_fps_with_ffmpeg(input_path, output_path, target_fps=30):
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-r', str(target_fps),
        output_path
    ]
    subprocess.run(cmd)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def mask_to_box(mask):
    """
    Convert a 2D boolean mask to a bounding box.
    The bounding box is defined by the top-left and bottom-right coordinates (x1, y1, x2, y2).
    """
    box_start = time.time()
    # Get the coordinates of True values in the mask
    coords = np.column_stack(np.where(mask))
    height = len(mask)
    width = len(mask[0])
    # print(f"width={width}, height={height}")

    if coords.shape[0] == 0:
        return None  # return None if the mask is empty
    
    # Find minimum and maximum coordinates
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Create the bounding box
    y1, x1 = min_coords
    y2, x2 = max_coords

    factor = 1/5
    xbuffer = np.floor((x2-x1)*factor)
    ybuffer = np.floor((y2-y1)*factor)

    y1 = np.max([0, int(y1-ybuffer)])
    y2 = np.min([height, int(y2+ybuffer)])
    x1 = np.max([0, int(x1-xbuffer)])
    x2 = np.min([width, int(x2+xbuffer)])

    print(f"box creation took {(time.time()-box_start):.3f} seconds")
    
    return np.array([x1, y1, x2, y2])

def show(mask, box, image, score):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_box(box, plt.gca())
    plt.title(f"score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

# init
from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# ##
video_file_path = 'video/frog2_30fps.mp4'
file_name = video_file_path.split("/")[-1].split(".")[0] if "/" in video_file_path else video_file_path.split(".")[0]
print(file_name)
start_frame = 75

input_video_path = video_file_path
output_video_path = 'video/frog2_30fps2.mp4'

if get_fps(input_video_path) != 30:
    change_fps_with_ffmpeg(input_video_path, output_video_path)
    video_file_path = output_video_path

capture = cv2.VideoCapture(video_file_path)




i = 0
cutout = None
setup_completed = False
if not capture.isOpened():
    print("Error: Couldn't open the video file.")
else:
    while True:
        frame_start = time.time()
        return_flag, current_frame = capture.read()

        if not return_flag:
            print("End of video.")
            break
        if i < start_frame:
            i += 1
            continue

        # Your image processing code here
        image = current_frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        if not setup_completed:
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.axis('off')
            plt.title('Select 2 foreground points')
            foreground = plt.ginput(2)
            plt.close()
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.axis('off')
            plt.title('Select 3 background points')
            background = plt.ginput(3)
            plt.close()
        
            input_points = np.array(foreground, dtype=int)
            bg_points = np.array(background, dtype=int)
            bg_labels = np.array([0,0,0])
            input_labels = np.array([1,1])
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # print(masks.shape)
            for j, (mask, score) in enumerate(zip(masks, scores)):
                if j == 2:
                    bbox = mask_to_box(mask)
                    show(mask,bbox,image,score)
                    # approved_mask = mask

            setup_completed = True
            last_mask = logits[np.argmin(scores), :, :]

        else:
            masks, scores, logits = predictor.predict(
                box=bbox,
                point_coords=bg_points,
                point_labels=bg_labels,
                mask_input=last_mask[None, :, :],
                multimask_output=False,
            )
            # print(masks.shape)
            for j, (mask, score) in enumerate(zip(masks, scores)):
                bbox = mask_to_box(mask)
                # show(mask, bbox, image, score)

            last_mask = logits[np.argmin(scores), :, :]

        output_masked_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        output_masked_image[mask == 1, :3] = image[mask == 1]
        output_masked_image[mask == 1, 3] = 255  # Setting alpha channel

        # Save this output image as PNG
        os.makedirs(f"output/{file_name}/mask", exist_ok=True)
        cv2.imwrite(f'output/{file_name}/mask/{file_name}_mask_{i}.png', cv2.cvtColor(output_masked_image, cv2.COLOR_RGBA2BGRA))

        # Now, let's cut out the mask part from the original image

        output_cutout_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        output_cutout_image[:, :, :3] = image
        output_cutout_image[:, :, 3] = 255  # Initialize alpha channel to fully visible
        output_cutout_image[mask == 1, 3] = 0  # Make the mask part transparent

        # Save this output cutout image as PNG
        os.makedirs(f"output/{file_name}/cutout", exist_ok=True)
        cv2.imwrite(f'output/{file_name}/cutout/{file_name}_cutout_{i}.png', cv2.cvtColor(output_cutout_image, cv2.COLOR_RGBA2BGRA))

        # time.sleep(0.1)
        print(f"frame {i} took {(time.time()-frame_start):.3f} seconds")
        i += 1

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the VideoCapture object
    capture.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()