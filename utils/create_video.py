import cv2
import os
import numpy as np

def create_video_from_images(image_dir, output_video_path, fps=30, img_size=None):
    """
    Create a video from a directory of images.

    Parameters:
    - image_dir: Path to the directory containing the images.
    - output_video_path: Path where the output video will be saved.
    - fps: Frames per second for the output video.
    - img_size: (width, height) tuple specifying the size of each frame. If None, use the size of the first image.
    """
    # Get all image paths
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
    image_paths.sort()  # Sort the images by name

    # Read the first image to determine the size
    frame = cv2.imread(image_paths[0])
    if img_size is not None:
        frame = cv2.resize(frame, img_size)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if img_size is not None:
            frame = cv2.resize(frame, img_size)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage
DATASET = 'imagenet_winners'
image_directory = f'./results/Imagenet1k/{DATASET}'
output_video_file = f'./{DATASET}_RSC_video.mp4'
fps = 1  # Frames per second

first_image = cv2.imread(f'./results/Imagenet1k/{DATASET}/Avg RSC Iteration-0.png')
height, width, layers = first_image.shape
frame_size = (width, height)

create_video_from_images(image_directory, output_video_file, fps, img_size=frame_size)
