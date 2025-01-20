import os
import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder)[21800:22000]:
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = load_image(img_path).cuda()
            images.append(img)
    return images

# Function to draw matches and save frames to video
def draw_and_save_matches(img1, keypoints1, img2, keypoints2, matches, output_video_path):
    img1_cpu = (img1.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img2_cpu = (img2.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    keypoints1_cv2 = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints1.cpu().numpy()]
    keypoints2_cv2 = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints2.cpu().numpy()]

    dmatches = [cv2.DMatch(i, i, 0) for i in range(len(matches))]

    img_matches = cv2.drawMatches(img1_cpu, keypoints1_cv2, img2_cpu, keypoints2_cv2, dmatches, None, flags=2)

    return img_matches

def extract_matches(matches, scores, threshold = 0.99):
    selected_matches = []
    selected_scores = []

    for match, score in zip(matches, scores):
        if score >= threshold:
            selected_matches.append(match)
            selected_scores.append(score)

    return torch.stack(selected_matches)

# Path to the input image folders
folder1_path = r'G:\data\0903\cam0'
folder2_path = r'G:\data\0903\cam1'

# Load images from folders
images1 = load_images_from_folder(folder1_path)
images2 = load_images_from_folder(folder2_path)

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
matcher = LightGlue(features='superpoint').eval().cuda()

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'output_video.mp4'
video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (2048, 768))  # Adjust resolution and fps as needed

# Process each pair of images and write frames to video
for img1, img2 in zip(images1, images2):
    feats1 = extractor.extract(img1)
    feats2 = extractor.extract(img2)

    # match the features
    matches01 = matcher({'image0': feats1, 'image1': feats2})
    feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]  # remove batch dimension
    matchess = matches01['matches']  # indices with shape (K,2)
    matches = extract_matches(matchess,matches01["scores"])
    points1 = feats1['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points2 = feats2['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    # Draw matches and save frame to video
    img_matches = draw_and_save_matches(img1, points1,
                                        img2, points2,
                                        matches, output_video_path)

    # cv2.imshow('My Image',cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    video_writer.write(cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR))

# Release the video writer
video_writer.release()

print(f"Video saved to {output_video_path}")
