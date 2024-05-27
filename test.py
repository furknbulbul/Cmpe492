
import torch
import argparse
from dataloader.multimodal_dataset import *
from onnx2pytorch import ConvertModel
import onnx
import wandb
from models.VGGNet import VGGNet
from models.multimodal.multimodal import Multimodal
from torch.utils.data import DataLoader, random_split
from dataloader.multimodal_dataset import *
from dataloader.dataset import *

# Set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="model.pth")
parser.add_argument("--data_root", type=str, default="data/FER2013")
parser.add_argument("--model", type=str, default="vgg")
parser.add_argument("--use_wandb", action="store_true")
args = parser.parse_args()




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
onnx_model = onnx.load(args.model_path)


pytorch_model = ConvertModel(onnx_model)


if args.model == "vgg":
    model = VGGNet()
    model.load_state_dict(torch.load(pytorch_model))
    dataset_test = dataset(args.data_root, phase="test")

elif args.model == "multimodal":
    model = Multimodal(hidden_dim=256, output_dim=64)
    model.load_state_dict(torch.load(pytorch_model))
    dataset_test = MultimodalDataset(args.data_root, phase="test")


test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True, pin_memory=True)

cross_entropy = torch.nn.CrossEntropyLoss()

def test_model():
    
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name="vgg_test")
    
    
    model.eval()
    
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # No gradient updates needed for testing
    with torch.no_grad():
        for data, _, targets in test_loader:
            # Move data to the appropriate device
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Calculate the batch loss
            loss = cross_entropy(outputs, targets)
            test_loss += loss.item() * data.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculate and print average loss and accuracy
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%')
    
    # Log to wandb
    if args.use_wandb:
        wandb.log({"Test": { "Loss": test_loss, "Test Accuracy": test_accuracy}})
        wandb.finish()
    
    return test_loss, test_accuracy


test_model()