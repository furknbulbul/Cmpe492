from setup import *
from utils import *
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from models.multimodal import *
import torch
from torchvision.transforms import transforms
from dataloader.dataset import *
from dataloader.multimodal_dataset import *
from trainer.multimodal_trainer import *
from torch.utils.data import DataLoader, random_split
from models.VGGNet import VGGNet
from models.multimodal.multimodal import Multimodal
import torch.nn as nn
from loss.nt_xent import NTXentLoss



args, path, logger, wandb_config = setup()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info("Using device: %s", device) 
transform = None
if args.augmentation:
        print("Using augmentation")
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p = 0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p = 0.5),
            ]
        )



cross_entropy = nn.CrossEntropyLoss()
nt_xent = NTXentLoss(device, args.batch, args.ntxnet_temp, use_cosine_similarity = True, alpha_weight = args.ntxnet_alpha)

if args.model == "vgg":
    dataset_train = dataset(root = args.data_root, phase = 'train', transform = transform)
    dataset_test = dataset(root = args.data_root, phase = 'test', transform = transform)
    model = VGGNet(config = args.vgg_config, num_classes = 7, dropout = args.dropout)
    trainer = ImageTrainer()

if args.model == "multimodal":
    print("Using multimodal model")
    dataset_train = MultimodalDataset(root = args.data_root, phase = 'train', transform = transform)
    dataset_test = MultimodalDataset(root = args.data_root, phase = 'test', transform = transform)
    model = Multimodal(hidden_dim = args.mlp_hidden_dim, output_dim = args.mlp_output_dim, num_classes = 7, dropout = args.dropout)
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Testing dataset size: {len(dataset_test)}")
    trainer = MultimodalTrainer()


optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.l2regularization) if args.optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
if args.scheduler == "cos":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epoch)
if args.scheduler == "reduce":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5)




model.to(device)
train_length = int(len(dataset_train) * args.ratio * 1.2 / 100)
val_length = len(dataset_train) - train_length
test_length = len(dataset_test)


dataset_train, dataset_val = random_split(dataset_train, [train_length, val_length],
                                              generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset_val, batch_size=args.batch, shuffle=True,  pin_memory=True)
test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, pin_memory=True)


logger.info("Train set length: %d", len(dataset_train))
logger.info("Validation set length: %d", len(dataset_val))
logger.info("Test set length: %d", len(dataset_test))




def train_vgg():
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = "vgg")
        wandb.watch(model)
    example_image = torch.rand(1, 1, 48, 48).to(device)
    
    for epoch in range(1, args.pretrain_epoch+1):
        train_acc, train_loss = trainer.train_model(model, train_loader, cross_entropy, optimizer)
        val_acc, val_loss = trainer.val_model(model, val_loader, cross_entropy)
        if args.scheduler:
            scheduler.step(val_acc)
        logger.info("Epoch: %d, Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f", epoch, train_loss, train_acc, val_loss, val_acc)
        
        logger.train_log_wandb(epoch, train_loss, train_acc)
        logger.val_log_wandb(epoch, val_loss, val_acc)

        if epoch % args.save_freq == 0:
            #torch.save(model.state_dict(), path + f"/vgg_{epoch}.pth")
            if args.use_wandb:
                try:
                    #wandb.save(path + f"/vgg_{epoch}.pth")
                    torch.onnx.export(model, example_image, f"vgg{epoch}.onnx")
                    wandb.save(f"vgg{epoch}.onnx")
                    artifact = wandb.Artifact('vgg-model-weights', type='model')
                    artifact.add_file(f"vgg{epoch}.onnx")
                    wandb.log_artifact(artifact)
                except:
                    pass


    
    if args.use_wandb:
        
        torch.onnx.export(model, example_image, f"vgg-trained.onnx")
        wandb.save(f"vgg-trained.onnx")

    wandb.finish()


     
def train_pretrain():
    example_image = torch.rand(1, 1, 48, 48).to(device)
    example_text = torch.randint(0, 100, (1, 100)).to(device)
    if args.use_wandb:
        print(wandb_config)
        wandb.init(project=args.wandb_project, config=wandb_config, name = "pretrain")
        wandb.watch(model)

    for epoch in range(1, args.pretrain_epoch+1):
        train_loss = trainer.pretrain_model(model, train_loader, nt_xent, optimizer)
        val_loss = trainer.pre_val_model(model, val_loader, nt_xent)

        logger.info("Epoch: %d, Train Loss: %f, Val Loss: %f", epoch, train_loss, val_loss)
        logger.train_log_wandb(epoch, train_loss)
        logger.val_log_wandb(epoch, val_loss)

        if epoch % args.save_freq == 0:
            #torch.save(model.state_dict(), path + f"/vgg_{epoch}.pth")
            if args.use_wandb:
                
                #wandb.save(path + f"/vgg_{epoch}.pth")
                torch.onnx.export(model, (example_image,example_text), f"pretrain{epoch}.onnx")
                wandb.save(f"pretrain{epoch}.onnx")
                artifact = wandb.Artifact('pretrain-model-weights', type='model')
                artifact.add_file(f"pretrain{epoch}.onnx")
                wandb.log_artifact(artifact)
    if args.use_wandb:
        torch.onnx.export(model, (example_image,example_text), f"pretrain-done.onnx")
        wandb.save(f"pretrain-done.onnx")
        artifact = wandb.Artifact('pretrain-model-weights', type='model')
        artifact.add_file(f"pretrain-done.onnx")
        wandb.log_artifact(artifact)

    wandb.finish()
                
def train_classifier():
    example_image = torch.rand(1, 1, 48, 48).to(device)
    example_text = torch.randint(0, 100, (1, 100)).to(device)
    model.use_classifier = True
    model.freeze_cnn = True
    model.reset_classifier()
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = "classifier")
        wandb.watch(model)

    for epoch in range(1, args.classifier_epoch+1):
        train_acc, train_loss = trainer.train_classifier(model, train_loader, cross_entropy, optimizer)
        val_acc, val_loss = trainer.val_model(model, val_loader, cross_entropy)

        logger.info("Epoch: %d, Train Loss: %f, Val Loss: %f, Train Acc: %f, Val Acc: %f", epoch, train_loss, val_loss, train_acc, val_acc)
        logger.train_log_wandb(epoch, train_loss, train_acc)
        logger.val_log_wandb(epoch, val_loss, val_acc)



        if epoch % args.save_freq == 0:
            #torch.save(model.state_dict(), path + f"/vgg_{epoch}.pth")
            if args.use_wandb:
                
                #wandb.save(path + f"/vgg_{epoch}.pth")
                torch.onnx.export(model, (example_image,example_text), f"classifer{epoch}.onnx")
                wandb.save(f"classifier{epoch}.onnx")
                artifact = wandb.Artifact('classifier-model-weights', type='model')
                artifact.add_file(f"classifier{epoch}.onnx")
                wandb.log_artifact(artifact)

    if args.use_wandb:
        torch.onnx.export(model, (example_image,example_text), f"classifier-done.onnx")
        wandb.save(f"classifier-done.onnx")
        artifact = wandb.Artifact('classifier-model-weights', type='model')
        artifact.add_file(f"classifier-done.onnx")
        wandb.log_artifact(artifact)


    wandb.finish()




def train_finetune():
    example_image = torch.rand(1, 1, 48, 48).to(device)
    example_text = torch.randint(0, 100, (1, 100)).to(device)
    model.use_classifier = True
    model.freeze_cnn = False
    model.reset_classifier()
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = "finetune")
        wandb.watch(model)
    
    for epoch in range(1, args.finetune_epoch+1):
        train_acc, train_loss = trainer.train_classifier(model, train_loader, cross_entropy, optimizer)
        val_acc, val_loss = trainer.val_model(model, val_loader, cross_entropy)

        logger.info("Epoch: %d, Train Loss: %f, Val Loss: %f, Train Acc: %f, Val Acc: %f", epoch, train_loss, val_loss, train_acc, val_acc)
        logger.train_log_wandb(epoch, train_loss, train_acc)
        logger.val_log_wandb(epoch, val_loss, val_acc)

        if epoch % args.save_freq == 0:
            #torch.save(model.state_dict(), path + f"/vgg_{epoch}.pth")
            if args.use_wandb:
                
                #wandb.save(path + f"/vgg_{epoch}.pth")
                torch.onnx.export(model, (example_image,example_text), f"finetune{epoch}.onnx")
                wandb.save(f"finetune{epoch}.onnx")
                artifact = wandb.Artifact('finetune-model-weights', type='model')
                artifact.add_file(f"finetune{epoch}.onnx")
                wandb.log_artifact(artifact)

    if args.use_wandb:
        torch.onnx.export(model, (example_image,example_text), f"finetune-done.onnx")
        wandb.save(f"finetune-done.onnx")
        artifact = wandb.Artifact('finetune-model-weights', type='model')
        artifact.add_file(f"finetune-done.onnx")
        wandb.log_artifact(artifact)
        
    wandb.finish()


if args.train_pipeline:
    train_pretrain()
    train_classifier()
    train_finetune()
elif args.pretrain:
    train_pretrain()

elif args.classifier:
    # need load
    pass
elif args.finetune:
    # need load
    pass