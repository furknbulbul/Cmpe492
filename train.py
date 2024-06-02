from setup import *
from utils import *
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch
from torchvision.transforms import transforms
from dataloader.image_dataset import *
from dataloader.multimodal_dataset import *
from trainer.multimodal_trainer import *
from torch.utils.data import DataLoader, random_split
from models.VGGNet import VGGNet
from models.multimodal.multimodal import Multimodal
import torch.nn as nn
from loss.nt_xent import NTXentLoss
from sklearn.metrics import confusion_matrix
import seaborn as sns  
from loss.contrastive_loss import ContrastiveLoss
from loss.triplet_loss import TripletLoss
from dataloader.siamase_dataset import SiamaseDataset
from models.VGGNet import SiamaseNetVGG
import torchvision.models as models
from models.resnet import ResNet50 
from models.ViT.ViT import ViT

# TODO: hyperparameter tuning
# more augmentation, maybe noise
# decrease the dim of the mlp layers
# increase batch size



args, path, logger, wandb_config = setup()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info("Using device: %s", device) 
transform = None
if args.augmentation:
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p = 0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p = 0.5),
            ]
        )



cross_entropy = nn.CrossEntropyLoss()
nt_xent = NTXentLoss(device, args.batch, args.ntxnet_temp, use_cosine_similarity = True, alpha_weight = args.ntxnet_alpha)
contrastive_loss = ContrastiveLoss(margin=1.0)
triplet_loss = TripletLoss(margin=1.0)
print(args.model)

if args.model == "vgg":
    if args.contrastive_loss:
        dataset_train = SiamaseDataset(root = args.data_root, phase = 'train', transform = transform)
        dataset_test = ImageDataset(root = args.data_root, phase = 'test', transform = None)
        model = SiamaseNetVGG(VGGNet(config = args.vgg_config, num_classes = 7, dropout = args.dropout), use_classifier = False, freeze_cnn=True)
        trainer = ImageTrainer()
    else:
        dataset_train = ImageDataset(root = args.data_root, phase = 'train', transform = transform)
        dataset_test = ImageDataset(root = args.data_root, phase = 'test', transform = None)
        model = VGGNet(config = args.vgg_config, num_classes = 7, dropout = args.dropout)
        trainer = ImageTrainer()

if args.model == "resnet":
    dataset_train = ImageDataset(root = args.data_root, phase = 'train', transform = transform)
    dataset_test = ImageDataset(root = args.data_root, phase = 'test', transform = None)
    model = ResNet50(num_classes = 7, is_classifier = True, dropout = args.dropout)
    trainer = ImageTrainer()

if args.model == "vit":
    dataset_train = ImageDataset(root = args.data_root, phase = 'train', transform = transform)
    dataset_test = ImageDataset(root = args.data_root, phase = 'test', transform = None)
    model = ViT()
    trainer = ImageTrainer()
   

if args.model == "multimodal":
    print("Using multimodal model")
    image_embedding_dim = 512 if args.image_embedding == "vgg11" or args.image_embedding == "vgg16" else 0
    image_embedding_dim = 2048 if args.image_embedding == "resnet" else 0
    dataset_train = MultimodalDataset(root = args.data_root, phase = 'train', transform = transform, text_embedding_dim=args.word_embedding_dim)
    dataset_test = MultimodalDataset(root = args.data_root, phase = 'test', transform = None, text_embedding_dim=args.word_embedding_dim)
    model = Multimodal(image_embedding_dim = image_embedding_dim, hidden_dim = args.mlp_hidden_dim, output_dim = args.mlp_output_dim, text_embedding_dim=args.word_embedding_dim, image_embedding= args.image_embedding, num_classes = 7, dropout = args.dropout)
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
dataset_val.transform = None
train_loader = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset_val, batch_size=args.batch, shuffle=True,  pin_memory=True)
test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, pin_memory=True)


logger.info("Train set length: %d", len(dataset_train))
logger.info("Validation set length: %d", len(dataset_val))
logger.info("Test set length: %d", len(dataset_test))




def train_vgg_resnet_vit(name = "vgg"):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = name)
        wandb.watch(model)
    
    for epoch in range(1, args.pretrain_epoch+1):
        train_acc, train_loss = trainer.train_model(model, train_loader, cross_entropy, optimizer)
        val_acc, val_loss = trainer.val_model(model, val_loader, cross_entropy)
        if args.scheduler:
            scheduler.step(val_acc)
        logger.info("Epoch: %d, Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f", epoch, train_loss, train_acc, val_loss, val_acc)
        
        logger.train_log_wandb(epoch, train_loss, train_acc)
        logger.val_log_wandb(epoch, val_loss, val_acc)

        if epoch % args.save_freq == 0 and epoch != args.pretrain_epoch:
            upload_wandb(f"{name}_{epoch}", model, args)


    
    upload_wandb(f"{name}-done", model,  args)
    pred, probs, corrects, test_acc = trainer.test_model(model, test_loader)
    logger.info("Test Accuracy: %f", test_acc)
    logger.test_log_wandb(test_acc)
    corrects = torch.tensor(corrects).cpu()
    pred = torch.tensor(pred).cpu()
    cm = confusion_matrix(corrects, pred)
    save_confusion_matrix(cm, args)
    wandb.finish()


     
def train_pretrain():
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = "pretrain")
        wandb.watch(model)

    for epoch in range(1, args.pretrain_epoch+1):
        train_loss = trainer.pretrain_model(model, train_loader, nt_xent, optimizer)
        val_loss = trainer.pre_val_model(model, val_loader, nt_xent)

        logger.info("Epoch: %d, Train Loss: %f, Val Loss: %f", epoch, train_loss, val_loss)
        logger.train_log_wandb(epoch, train_loss)
        logger.val_log_wandb(epoch, val_loss)

        if epoch % args.save_freq == 0 and epoch != args.pretrain_epoch:
            upload_wandb(f"pretrain_{epoch}", model,  args)
            
    
    upload_wandb("pretrain-done", model,  args)
    wandb.finish()
                
def train_classifier(type = "classifier"):
   
    model.use_classifier = True
    model.freeze_cnn = True
    model.reset_classifier()
    print("Using classifer of the model:", model.use_classifier)
    print("Freezeing cnn:", model.freeze_cnn)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = type)
        wandb.watch(model)

    for epoch in range(1, args.classifier_epoch+1):
        train_acc, train_loss = trainer.train_classifier(model, train_loader, cross_entropy, optimizer)
        val_acc, val_loss = trainer.val_model(model, val_loader, cross_entropy)

        logger.info("Epoch: %d, Train Loss: %f, Val Loss: %f, Train Acc: %f, Val Acc: %f", epoch, train_loss, val_loss, train_acc, val_acc)
        logger.train_log_wandb(epoch, train_loss, train_acc)
        logger.val_log_wandb(epoch, val_loss, val_acc)

        if epoch % args.save_freq == 0:
            upload_wandb(f"{type}_{epoch}", model,  args)
        

    upload_wandb(f"{type}-done", model, args)

    # test 
    pred, probs, corrects, test_acc = trainer.test_model(model, test_loader)
    logger.info("Test Accuracy: %f", test_acc)
    logger.test_log_wandb(test_acc)
    corrects = torch.tensor(corrects).cpu()
    pred = torch.tensor(pred).cpu()
    cm = confusion_matrix(corrects, pred)
    save_confusion_matrix(cm, args)
    wandb.finish()




def train_finetune(type = "finetune"):
    model.use_classifier = True
    model.freeze_cnn = False
    model.reset_classifier()
    print("Using classifer of the model:", model.use_classifier)
    print("Freezeing cnn:", model.freeze_cnn)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = type)
        wandb.watch(model)
    
    for epoch in range(1, args.finetune_epoch+1):
        train_acc, train_loss = trainer.train_classifier(model, train_loader, cross_entropy, optimizer)
        val_acc, val_loss = trainer.val_model(model, val_loader, cross_entropy)

        logger.info("Epoch: %d, Train Loss: %f, Val Loss: %f, Train Acc: %f, Val Acc: %f", epoch, train_loss, val_loss, train_acc, val_acc)
        logger.train_log_wandb(epoch, train_loss, train_acc)
        logger.val_log_wandb(epoch, val_loss, val_acc)

        if epoch % args.save_freq == 0 and epoch != args.finetune_epoch:
            upload_wandb(f"{type}_{epoch}", model,  args)
        
    
    upload_wandb(f"{type}-done", model,  args)
    pred, probs, corrects, test_acc = trainer.test_model(model, test_loader)
    logger.info("Test Accuracy: %f", test_acc)
    logger.test_log_wandb(test_acc)
    corrects = torch.tensor(corrects).cpu()
    pred = torch.tensor(pred).cpu()
    cm = confusion_matrix(corrects, pred)
    save_confusion_matrix(cm, args)
    wandb.finish()

def train_contrastive():
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=wandb_config, name = "contrastive")
        wandb.watch(model)
    for epoch in range(1, args.pretrain_epoch+1):
        train_loss = trainer.train_contrastive(model, train_loader, contrastive_loss, optimizer)
        val_loss = trainer.val_contrastive(model, val_loader, contrastive_loss)
        logger.info("Epoch: %d, Train Loss: %f, Val Loss: %f", epoch, train_loss, val_loss)
        logger.train_log_wandb(epoch, train_loss)
        logger.val_log_wandb(epoch, val_loss)

        if epoch % args.save_freq == 0:
            upload_wandb(f"contrastive_{epoch}", model,  args)
    
    upload_wandb("contrastive-img-img-done", model,  args)
    wandb.finish()
    train_classifier("contrastive-img-img-classifier")
    train_finetune("contrastive-img-img-finetune")



        

if args.model == "vgg":
    if not args.contrastive_loss:
        train_vgg_resnet_vit(name = "vgg")
    else :
        train_contrastive()

if args.model == "resnet":
    train_vgg_resnet_vit(name = "resnet")
if args.model == "vit":
    train_vgg_resnet_vit(name = "vit")

if args.model == "multimodal" and args.train_pipeline:
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