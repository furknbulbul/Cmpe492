from utils import *
import dataloader.dataset
import torch

args = get_args()


assert args.model == "multimodal", "Only multimodal model is supported for pretraining"

