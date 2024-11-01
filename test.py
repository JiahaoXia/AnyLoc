import os

os.environ["LD_LIBRARY_PATH"] = (
    "~/Downloads/home/rise/Documents/anaconda3/envs/anyloc/lib"
)

import torch
from utilities import DinoV2ExtractFeatures
from utilities import VLAD
from torchvision import transforms as tvf

import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
desc_layer = 31
desc_facet = "value"
extractor = DinoV2ExtractFeatures(
    "dinov2_vitg14", desc_layer, desc_facet, device=device
)

# Make image patchable (14, 14 patches)
img_pt = torch.rand(3, 300, 400).to(device)
c, h, w = img_pt.shape
h_new, w_new = (h // 14) * 14, (w // 14) * 14
img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
img_pt = img_pt.repeat(20, 1, 1, 1)
# Main extraction
t0 = time.time()
ret = extractor(img_pt)  # [1, num_patches, desc_dim]
print(f"Time taken: {time.time() - t0:.3f} s")
print(ret.shape)
print("Done")
