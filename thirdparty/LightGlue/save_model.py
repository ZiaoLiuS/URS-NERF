import torch
import torch.jit
import torchvision.transforms as transforms
from lightglue import LightGlue, SuperPoint, DISK

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# Example input
example_input = torch.randn(1, 1, 240, 320)  # Adjust the shape as per your model input requirements

# Trace SuperPoint model and save to .pt file
traced_superpoint = torch.jit.trace(extractor, example_input)
traced_superpoint.save("superpoint_model.pt")

# Trace LightGlue model and save to .pt file
traced_lightglue = torch.jit.trace(matcher, {'image0': example_input, 'image1': example_input})
traced_lightglue.save("lightglue_model.pt")
