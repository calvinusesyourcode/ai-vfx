import subprocess

# Assuming test59 and test60 are in the same directory as this script
from test59 import video_to_png
from test60 import png_to_video

# Step 1: Convert video to PNG images
# video_path = 'in.mp4'
output_frames_dir = 'pre'
# video_to_png(video_path, fps=None)  # Use None to keep original FPS

# Step 2: Convert PNG images back to video
output_video = 'out.mp4'
png_to_video(output_frames_dir, output_video, fps=10)  # Use 30 FPS for the new video

# # Step 3: Extract audio from the original video
# audio_file = 'audio.mp3'
# subprocess.run(f'ffmpeg -i {video_path} -q:a 0 -map a {audio_file}', shell=True)

# # Step 4: Add audio to the new video
# final_output = "ascii_" + video_path
# subprocess.run(f'ffmpeg -i {output_video} -i {audio_file} -c:v copy -c:a aac {final_output}', shell=True)

print("Video processing complete!")
