import os, sys
from os.path import dirname, abspath, join, exists, basename
import argparse
import yaml
from custom_datasets.gsv import GSV_Dataset
import torch
from torchvision import transforms as tvf
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import numpy as np

from omegaconf import DictConfig, OmegaConf
import hydra


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        help="input path to the image files",
        default="/home/rise/XJH/Geo-Loc/geo-loc-data/data/manville-panos/pov",
    )
    parser.add_argument(
        "--model_cfg_file",
        type=str,
        help="path to the model configuration file",
        default="/home/rise/XJH/Geo-Loc/config/AnyLoc-VLAD-DINOv2.yaml",
    )
    parser.add_argument(
        "--bs",
        type=int,
        help="batch size for feature extraction",
        default=10,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output path to save the extracted features",
        default="/home/rise/XJH/Geo-Loc/geo-loc-data/data/manville-panos/",
    )
    parser.add_argument(
        "--torch_hub_dir",
        type=str,
        help="path to the torch hub directory",
        default="/home/rise/XJH/Geo-Loc/geo-loc-data/model-zoos/torch-hub",
    )

    return parser.parse_args()


# @hydra.main(
#     version_base=None,
#     config_path="/home/rise/XJH/Geo-Loc/config/DINOv2/satellite/gt",
#     config_name="cranford",
# )
@hydra.main(version_base=None)
def main(cfg):
    print("============START============")
    print(OmegaConf.to_yaml(cfg))

    torch.hub.set_dir(cfg.torch_hub_dir)

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # Extract parameters from the configuration
    domain = cfg.get("domain")
    backbone = cfg.get("backbone")
    vit_model = cfg.get("vit_model")
    vit_layer = cfg.get("vit_layer")
    vit_facet = cfg.get("vit_facet")
    num_c = cfg.get("num_c")
    if "basename" not in cfg:
        output_folder = f"{basename(cfg.input_dir)}---{domain}---{backbone}---{vit_model}---layer[{vit_layer}]---facet[{vit_facet}]---num_c[{num_c}]"
    else:
        output_folder = f"{cfg.basename}---{domain}---{backbone}---{vit_model}---layer[{vit_layer}]---facet[{vit_facet}]---num_c[{num_c}]"
    if not exists(join(cfg.output_dir, output_folder)):
        os.makedirs(join(cfg.output_dir, output_folder))
    output_path = join(cfg.output_dir, output_folder)

    # Save the YAML configuration to the output path
    with open(join(output_path, "model-cfg.yaml"), "w") as file:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), file)

    # Load the dataset
    base_transform = tvf.Compose(
        [
            tvf.Resize(tuple(cfg.resize)),
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    gsv_ds = GSV_Dataset(cfg.input_dir, transform=base_transform)
    gsv_dl = DataLoader(gsv_ds, batch_size=cfg.bs, shuffle=False)

    # Load the DINOv2 model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    extractor = torch.hub.load(
        "AnyLoc/DINO",
        "get_vlad_model",
        domain=domain,
        backbone=backbone,
        vit_model=vit_model,
        vit_layer=vit_layer,
        vit_facet=vit_facet,
        num_c=num_c,
        device=device,
    )
    # HDF5 file path
    hdf5_path = os.path.join(output_path, "features_and_paths.h5")
    pbar = tqdm(gsv_dl)

    # Open the HDF5 file in append mode
    with h5py.File(hdf5_path, "a") as f:
        # Create datasets if they don't already exist
        if "image_paths" not in f:
            img_paths_dataset = f.create_dataset(
                "image_paths", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )
            features_dataset = f.create_dataset(
                "features",
                (
                    0,
                    0,
                ),
                maxshape=(
                    None,
                    None,
                ),
                dtype=np.float16,
            )
        else:
            img_paths_dataset = f["image_paths"]
            features_dataset = f["features"]

        # Determine the starting index by checking the current size of `image_paths`
        start_idx = img_paths_dataset.shape[0]
        print(f"Resuming from index {start_idx}")

        for idx, (img, img_path) in enumerate(pbar):
            if idx < start_idx:
                continue  # Skip already processed entries
            pbar.set_postfix({"batch": idx + 1})

            try:
                img = img.to(device)
                bn, c, h, w = img.shape
                h_new, w_new = (h // 14) * 14, (w // 14) * 14
                img = tvf.CenterCrop((h_new, w_new))(img)
                ret = extractor(img).cpu().numpy()  # [bs, desc_dim]
                # Get desc_dim dynamically from the current batch
                desc_dim = ret.shape[1]

                # Resize datasets to add new data
                img_paths_dataset.resize((img_paths_dataset.shape[0] + len(img_path),))
                img_paths_dataset[-len(img_path) :] = img_path

                features_dataset.resize(
                    (features_dataset.shape[0] + ret.shape[0], desc_dim)
                )
                features_dataset[-ret.shape[0] :] = ret
            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                break

    print("------------done------------")


if __name__ == "__main__":
    main()
    print("============DONE============")
