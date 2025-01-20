import cv2
import numpy as np
from matplotlib import pyplot as plt
from lightglue.utils import load_image, rbd


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

torch.set_grad_enabled(False)

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

extractor.save_model()

# or DISK+LightGlue, ALIKED+LightGlue, or SIFT+LightGlue
# extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
# image0 = load_image('E:/DeepLearn/Code/LightGlue/assets/sacre_coeur1.jpg').cuda()
# image1 = load_image('E:/DeepLearn/Code/LightGlue/assets/sacre_coeur2.jpg').cuda()
image0 = load_image('E:/DeepLearn/Code/LightGlue/assets/temp-07292023052835-0.jpg').cuda()
image1 = load_image('E:/DeepLearn/Code/LightGlue/assets/temp-07292023052835-1.jpg').cuda()

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

traced_superpoint = torch.jit.trace(extractor, image0)
traced_superpoint.save("extractor_model.pt")

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
# Trace LightGlue model and save to .pt file
# traced_lightglue = torch.jit.trace(matcher, {'image0': feats0, 'image1': feats1})
# traced_lightglue.save("matcher_model.pt")


feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

# Move GPU tensors to CPU before using them with OpenCV
img1_cpu = (image0.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # Convert to uint8
img2_cpu = (image1.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # Convert to uint8

# Convert keypoints to a list of cv2.KeyPoint objects
keypoints1 = [cv2.KeyPoint(point[0], point[1], 1) for point in points0.cpu().numpy()]
keypoints2 = [cv2.KeyPoint(point[0], point[1], 1) for point in points1.cpu().numpy()]

# Convert matches to a list of cv2.DMatch objects
dmatches = [cv2.DMatch(i, i, 0) for i in range(len(matches))]

# Draw matches
img31 = cv2.drawMatches(img1_cpu, keypoints1, img2_cpu, keypoints2, dmatches[:50], None, flags=2)

plt.figure(figsize=(10, 10))
plt.title('SuperPoint+LightGlue', fontsize=12)
plt.imshow(img31)
plt.axis('off')
plt.show()
