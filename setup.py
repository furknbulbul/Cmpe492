from utils import *
import time
import json
import os


def setup():
    args = get_args()    
    path = create_checkpoint_path(args)
    logger = Logger(logfile=args.log_file, use_wandb=args.use_wandb)
    wandb_config = create_wandb_config(args)
    return args, path, logger, wandb_config

def create_checkpoint_path(args):
    checkpoint_path = "checkpoints/" + args.model + "_" + args.vgg_config +  "_" + str(int(time.time()))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    return checkpoint_path


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", default=64, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    parser.add_argument("-dr", "--data_root", default="Data/FER2013_small", type=str)
    parser.add_argument("-sf", "--save_freq", default=50, type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-r", "--ratio", default=60, type=int)
    parser.add_argument("-l", "--l2regularization", default=0, type=float)
    parser.add_argument("--log_file", default="output.log", type=str)
    parser.add_argument("-o", "--optimizer", default="adam", type=str)
    parser.add_argument("-m", "--vgg_config", default="vgg11", type=str)
    parser.add_argument("-d", "--dropout", default=0.2, type=float)
    parser.add_argument("-a", "--augmentation", action= "store_true", help = "use data augmentation")
    parser.add_argument("-sc", "--scheduler", default = None, type=str, help="[reduce, cos]")
    parser.add_argument("-M", "--model", default = 'vgg', type=str)
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", default="emotion-recognition", type=str)
    parser.add_argument("--classification", action="store_true", help="use classifier")
    parser.add_argument("--freeze_cnn", action="store_true", help="freeze cnn")
    parser.add_argument("--mlp_hidden_dim", default=256, type=int)
    parser.add_argument("--mlp_output_dim", default=64, type=int)
    parser.add_argument("--ntxnet_alpha", default=0.75, type=float, help="alpha for ntxnet loss")
    parser.add_argument("--ntxnet_temp", default=0.1, type=float, help="temperature for ntxnet loss")
    parser.add_argument("--word_embedding_dim", default=100, type=int)  
    parser.add_argument("--contrastive_loss", action="store_true", help="use contrastive loss") # use with vgg for now
    parser.add_argument("--image_embedding", default="vgg11", type=str)
    parser.add_argument("--resnet_config", default="resnet50", type=str)
    get_pipeline_args(parser)

    return parser.parse_args()

def get_pipeline_args(parser):
    parser.add_argument("--pretrain_epoch", default=1, type=int)
    parser.add_argument("--classifier_epoch", default=1, type=int)
    parser.add_argument("--finetune_epoch", default=1, type=int)
    parser.add_argument("--train_pipeline", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--classifier", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    


def create_wandb_config(args):
    config = {}
    config["batch"] = args.batch
    config["learning_rate"] = args.learning_rate
    config["data_root"] = args.data_root
    config["l2regularization"] = args.l2regularization
    config["optimizer"] = args.optimizer
    config["vgg_config"] = args.vgg_config
    config["dropout"] = args.dropout
    config["scheduler"] = args.scheduler
    config["augmentation"] = args.augmentation
    config["mlp_hidden_dim"] = args.mlp_hidden_dim
    config["mlp_output_dim"] = args.mlp_output_dim
    config["ntxnet_alpha"] = args.ntxnet_alpha
    config["image_embedding"] = args.image_embedding
    return config

