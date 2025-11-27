import os
import random
random.seed(224)
import sys
import glob
import time
import threading
import argparse
from typing import List, Optional
import torch.nn.functional as F
import numpy as np
import torch
from tqdm.auto import tqdm
import torchvision
from g2vlm_utils import load_model_and_tokenizer, save_ply_visualization

parser = argparse.ArgumentParser(description="Demo for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/dl3dv/", help="Path to folder containing images"
)
# parser.add_argument(
#     "--image_folder", type=str, default="examples/arkitscenes/", help="Path to folder containing images"
# )
parser.add_argument("--model_path",type=str, default="InternRobotics/G2VLM-2B-MoT")
parser.add_argument("--save_path",type=str, default="results/arkitscenes_results.ply")

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_folder = args.image_folder
    image_names = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    print(image_names)
  
    model, tokenizer, new_token_ids , vit_image_transform, dino_transform = load_model_and_tokenizer(args)
    pred = model.recon(
        tokenizer,
        new_token_ids,
        dino_transform,
        image_names, 
    )
    save_ply_visualization(pred, args.save_path)

if __name__ == "__main__":
    main()