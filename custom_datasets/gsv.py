# %%
import os
import sys
from pathlib import Path

# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARN: __file__ not found, trying local")
    dir_name = os.path.abspath("")
lib_path = os.path.realpath(f"{Path(dir_name).parent}")
# Add to path
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from PIL import Image
from configs import prog_args


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


class GSV_Dataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.image_paths = natsorted(
            [
                os.path.join(dp, f)
                for dp, dn, filenames in os.walk(root)
                for f in filenames
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = path_to_pil_img(img_path)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)


if __name__ == "__main__":
    base_transform = transforms.Compose(
        [
            transforms.Resize((400, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gsv_ds = GSV_Dataset(
        "/home/rise/XJH/Geo-Loc/geo-loc-data/data/manville-panos/pov",
        transform=base_transform,
    )

    gsv_dl = DataLoader(gsv_ds, batch_size=10, shuffle=False)
    for i, (img, img_path) in enumerate(gsv_dl):
        print(i, img.shape, img_path)

    print("DONE")
