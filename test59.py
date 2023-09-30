import cv2
import os

from ascii import ascii_filter_1

def video_to_png(video_path, fps=None):
    output_dir = "output_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None:
        fps = video_fps
    else:
        fps = min(fps, video_fps)

    skip_frames = video_fps // fps
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % skip_frames == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
            cv2.imwrite(output_path, frame)

            # Create ASCII art from the PNG and save it
            print("%%%")
            ascii_filter_1(fontsize=16, image_path=output_path)
            print("___")

            # Remove the original PNG
            os.remove(output_path)

        frame_number += 1

    cap.release()
    print(f"Extracted and converted {frame_number // skip_frames} frames from the video.")
