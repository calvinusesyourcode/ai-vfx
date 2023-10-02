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
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    
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

def show(mask, box, image, score, ax):
    clear_axis(ax)
    ax.imshow(image)
    ax.set_title(f"score: {score:.3f}")
    plt.draw()
    show_mask(mask, plt.gca())
    show_box(box, ax)
    plt.draw()
    time.sleep(20)

def select_from_options(ax, options):
    print(f"selecting from {options}")
    rect_height = 0.1
    rect_width = 0.2
    colors = ['blue', 'green', 'red', 'yellow', 'purple']

    rects = []
    for i in range(len(options)):
        x, y, w, h = [0, 0.9-(i*rect_height), rect_width, rect_height]
        rects.append(plt.Rectangle((x, y), w, h, facecolor=colors[i % len(colors)], alpha=0.5, transform=ax.transAxes))

    for rect in rects:
        ax.add_patch(rect)

    for i, option in enumerate(options):
        ax.text(0.02, 0.9 - i * rect_height + rect_height / 3, option, color='white', transform=ax.transAxes)
    
    print(f"soliciting user input")
    plt.draw()
    userinput = plt.ginput(1)
    point = userinput[0]
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    point_x = (point[0] - xlim[0]) / (xlim[1] - xlim[0])
    point_y = (point[1] - ylim[0]) / (ylim[1] - ylim[0])
    
    for i, rect in enumerate(rects):
        x, y, w, h = rect.get_xy()[0], rect.get_xy()[1], rect.get_width(), rect.get_height()
        if x <= point_x <= x + w and y <= point_y <= y + h:
            return options[i]
    
def get_points(n, title, ax):
    clear_axis(ax)
    ax.set_title(f"select {n} {title}")
    plt.draw()
    userinput = plt.ginput(n)
    return userinput
    
def clear_axis(ax):
    for item in (ax.texts+ax.patches):
        item.remove()

# init
print(f"Loading SAM...")
from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print(f"Done.")

# config
video_folder = "video_in"
json_folder = "json"

# initialize figure and axis just once
fig, ax = plt.subplots(figsize=(10, 10))
plt.figure(fig)
plt.axis("off")

# pre-setup
modes = ["default", "middle-out", "reverse"]
os.makedirs(video_folder, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)
videos = os.listdir(video_folder)
if videos is None or len(videos) == 0:
    print(f"Critical failure: No files found in folder {video_folder}")
    exit

# setup videos for processing
v = 0
for video_file in videos: # setup
    video_shortname, ext = os.path.splitext(video_file)
    video_title = video_shortname

    if ext.lower() != ".mp4":
        print(f"Skipped, non mp4: {video_file}")
        videos.remove(video_file)
        continue
    
    if f"{video_shortname}.json" in os.listdir("json"):
        print(f"Skipped, already have setup info: {video_file}")
        continue

    video_fullpath = os.path.join(video_folder, video_file)
    capture = cv2.VideoCapture(video_fullpath)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {video_fullpath}")

    start_frame = 0
    chosen_frame = None

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return_flag, current_frame = capture.read()

    image = current_frame
    image_height, image_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)
    ax.set_title("mode select")
    clear_axis(ax)
    chosen_mode = select_from_options(ax, modes)

    look_frame = int(total_frames//2) if chosen_mode == "middle-out" else int(total_frames-1) if chosen_mode == "reverse" else 0 
    look_frame_skip = 120

    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, look_frame)
        return_flag, current_frame = capture.read()

        image = current_frame
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        clear_axis(ax)
        ax.imshow(image)
        ax.set_title(f'frame {look_frame}')
        choices = [f"< {look_frame_skip} frames",f"> {look_frame_skip} frames", "here"]
        selected_option = select_from_options(ax, choices)
        if selected_option is None:
            print("??")
            continue
        elif selected_option == "here":
            chosen_frame = look_frame
            pass
        elif ">" in selected_option:
            look_frame = min(total_frames-1, look_frame + look_frame_skip)
            continue
        elif "<" in selected_option:
            look_frame = max(0, look_frame - look_frame_skip)
            continue
        else:
            print(f"User error?: unrecognized option selected for init masking frame {video_fullpath}")
            continue
        print(f"selected option: {selected_option}")

        if chosen_frame is None:
            print(f"User error?: No chosen frame during {video_fullpath}")
            continue
        
        capture.set(cv2.CAP_PROP_POS_FRAMES, chosen_frame)
        return_flag, image = capture.read()

        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        foreground_points = get_points(3, "foreground points", ax)
        background_points = get_points(3, "persisting background points", ax)

        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=np.array((foreground_points+background_points), dtype=int),
            point_labels=np.array(([1]*len(foreground_points))+([0]*len(background_points)), dtype=int),
            multimask_output=True,
        )

        for j, (mask, score) in enumerate(reversed(list(zip(masks, scores)))):
            # plt init
            clear_axis(ax)
            ax.imshow(image)
            ax.set_title(f"mask {j} score: {score:.3f}")
            
            # show box
            box = mask_to_box(mask)
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

            # show mask
            color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

            plt.draw()
            mask_approved = True if select_from_options(ax, ["yes", "no"]) == "yes" else False

            if mask_approved:
                chosen_mask = logits[j, :, :]
                break
        
        if mask_approved:
            break
        else:
            print(f"Restarting mask creation for video {video_fullpath}")

    # This test was a success, chosen_mask[] uses weights, mask[] is boolean
    #####
    # for i, _ in enumerate(chosen_mask):
    #     if not (chosen_mask[i] == mask[i]):
    #         print(f"last mask {i}: {chosen_mask[i]}, mask {i}: {mask[i]}")
    # print("if no prints above then mask and chosen_mask are equal")
    #####

    # save video setup
    video_setup_data = {
        'video_fullpath': video_fullpath,
        'chosen_mode': chosen_mode,
        'chosen_startframe': chosen_frame,
        'chosen_box': box.tolist(),
        'chosen_bg_points': background_points,
        'chosen_mask': chosen_mask.tolist(),
    }

    with open(f"json/{video_shortname}.json", 'w') as f:
        json.dump(video_setup_data, f)
        print(f"Saved info for file {video_fullpath}")

    v += 1
    print(f"Success: processed {video_fullpath}")

print(f"{v} video(s) setup.")

# process videos
v = 0
for video_file in videos:
    video_shortname, ext = os.path.splitext(video_file)
    video_title = video_shortname

    if ext.lower() != ".mp4":
        print(f"Skipped, non mp4: {video_file}")
        continue
    
    if f"{video_shortname}.json" not in os.listdir("json"):
        print(f"No JSON setup data found: {video_file}")
        continue
    
    with open(f"{json_folder}/{video_shortname}.json", "r") as f:
        print(f"Retrieving JSON setup data: {video_file}")
        video_setup = json.load(f)
        video_fullpath = video_setup["video_fullpath"]
        chosen_mode = video_setup["chosen_mode"]
        chosen_startframe = video_setup["chosen_startframe"]
        chosen_box = np.array(video_setup["chosen_box"], dtype=int)
        chosen_bg_points = np.array(video_setup["chosen_bg_points"], dtype=float)
        chosen_mask = np.array(video_setup["chosen_mask"], dtype=float)

    video_fullpath = os.path.join(video_folder, video_file)
    capture = cv2.VideoCapture(video_fullpath)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {video_fullpath}")

    frames_processed = 0
    process_frame = chosen_startframe
    last_mask = chosen_mask
    last_box = chosen_box
    while frames_processed < total_frames:
        process_inittime = time.time()

        capture.set(cv2.CAP_PROP_POS_FRAMES, process_frame)
        return_flag, current_frame = capture.read()

        image_height, image_width, _ = current_frame.shape
        image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        if not capture.isOpened():
            print(f"Error: Couldn't open file: {video_file}")
            break
        
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            box=last_box,
            point_coords=chosen_bg_points,
            point_labels=([0]*len(chosen_bg_points)),
            mask_input=last_mask[None, :, :],
            multimask_output=False,
        )

        mask = masks[0]
        score = scores[0]
        last_mask = logits[0, :, :]
        box = mask_to_box(mask)

        # write files
        output_masked_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        output_masked_image[mask == 1, :3] = image[mask == 1]
        output_masked_image[mask == 1, 3] = 255  # setting alpha channel

        os.makedirs(f"output/{video_title}/mask", exist_ok=True)
        cv2.imwrite(f'output/{video_title}/mask/{video_title}_mask_{process_frame}.png', cv2.cvtColor(output_masked_image, cv2.COLOR_RGBA2BGRA))

        output_cutout_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        output_cutout_image[:, :, :3] = image
        output_cutout_image[:, :, 3] = 255 # init alpha channel 
        output_cutout_image[mask == 1, 3] = 0  # mask part transparent

        os.makedirs(f"output/{video_title}/cutout", exist_ok=True)
        cv2.imwrite(f'output/{video_title}/cutout/{video_title}_cutout_{process_frame}.png', cv2.cvtColor(output_cutout_image, cv2.COLOR_RGBA2BGRA))

        frames_processed += 1

        if process_frame < chosen_startframe:
            process_frame += -1
        elif (total_frames - chosen_startframe) == frames_processed:
            process_frame = chosen_startframe - 1
            last_mask = chosen_mask
            last_box = chosen_box
        else:
            process_frame += 1
        
        print(f"frame {process_frame} took {(time.time()-process_inittime):.3f} seconds")
    
    capture.release()