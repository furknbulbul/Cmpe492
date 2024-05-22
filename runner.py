from utils import *
from dataset import *
from models.VGGNet import *
from training import *
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
from timeit import default_timer as timer
from torchvision.transforms import transforms
from models.multimodal import Multimodal
from multimodal_trainer import MultimodalTrainer
from multimodal_dataset import MultimodalDataset


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", default=120, type=int)
parser.add_argument("-b", "--batch", default=64, type=int)
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
parser.add_argument("-dr", "--data_root", default="Data/FER2013_small", type=str)
parser.add_argument("-sf", "--save_freq", default=20, type=int)
parser.add_argument("-s", "--seed", default=42, type=int)
parser.add_argument("-r", "--ratio", default=60, type=int)
parser.add_argument("-l", "--l2regularization", default=0, type=float)
parser.add_argument("--log_dir", default="tensorboard_logs", type=str)
parser.add_argument("--log_file", default="output.log", type=str)
parser.add_argument("-o", "--optimizer", default="adam", type=str)
parser.add_argument("-m", "--config", default="vgg11", type=str)
parser.add_argument("-d", "--dropout", default=0.2, type=float)
parser.add_argument("-a", "--augmentation", action= "store_true", help = "use data augmentation")
parser.add_argument("-sc", "--scheduler", default = "none", type=str, help="[reduce, cos]")
parser.add_argument("-M", "--model", default = 'vgg', type=str)

best_accuracy = 0


def main():
    global best_accuracy

    args = parser.parse_args()
    ROOT = args.data_root
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch
    LEARNING_RATE = args.learning_rate
    SAVE_FREQ = args.save_freq
    SEED = args.seed
    TRAIN_RATIO = args.ratio / 100
    L2 = args.l2regularization
    LOG_DIR = args.log_dir
    LOG_FILE = args.log_file
    OPTIMIZER = args.optimizer
    CONFIG = args.config
    DROPOUT = args.dropout
    AUGMENTATION = args.augmentation
    SCHEDULER = args.scheduler
    MODEL = args.model
    

    writer = SummaryWriter(log_dir=LOG_DIR)
    logger = Logger(logfile=LOG_FILE)

    checkpoint_path = "checkpoints/" + "batch_size" +  str(BATCH_SIZE) + "_learning_rate" +str(LEARNING_RATE) + "_l2" + str(L2) + "_seed" + str(SEED) + "_optimizer" + str(OPTIMIZER) + "config" + str(CONFIG) + "_dropout" + str(DROPOUT)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Using device: %s", device)
    logger.info("Loading data from %s", args.data_root)

    transform = None
    if AUGMENTATION:
        print("Using augmentation")
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p = 0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p = 0.5),
            ]
        )

    if MODEL == 'vgg':
        dataset_train = dataset(root=ROOT, phase='train', transform=None)
        dataset_test = dataset(root=ROOT, phase='test', transform=None)
    if MODEL == 'multimodal':
        dataset_train = MultimodalDataset(root=ROOT, phase='train', transform=None)
        dataset_test = MultimodalDataset(root=ROOT, phase='test', transform=None)
    


    train_length = int(len(dataset_train) * TRAIN_RATIO * 1.2)
    val_length = len(dataset_train) - train_length
    test_length = len(dataset_test)
    dataset_train, dataset_val = random_split(dataset_train, [train_length, val_length],
                                              generator=torch.Generator().manual_seed(SEED))
    
    dataset_train.transform = transform
    dataset_val.transform = None
    
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    logger.info("Train set length: %d", len(dataset_train))
    logger.info("Validation set length: %d", len(dataset_val))
    logger.info("Test set length: %d", len(dataset_test))

    if MODEL == 'vgg':
        model = VGGNet(config=CONFIG, dropout=DROPOUT)
    if MODEL == 'multimodal':
        model = Multimodal(hidden_dim= 64, output_dim = 16) # TODO: think about dimensions

    cross_entropy = torch.nn.CrossEntropyLoss()
    margin_loss = torch.nn.MarginRankingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = L2) if OPTIMIZER == "adam" else torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    #write_graph(writer, model, input_tensor=torch.rand(1, 1, 48, 48))
    model.to(device)

    


    if SCHEDULER == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5)
    if SCHEDULER == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if MODEL == 'vgg':
        trainer = ImageTrainer()
    if MODEL == 'multimodal':
        trainer = MultimodalTrainer()
        criterion = margin_loss


    start = timer()
    pre = start
    for epoch in range(1, EPOCHS + 1):
        is_best = False
        logger.info("Epoch %d", epoch)

        train_acc, train_loss = trainer.train_model(model, train_loader, criterion, optimizer)
        logger.info("Training accuracy: %f", train_acc)
        logger.info("Training loss: %f", train_loss)

        val_acc, val_loss = trainer.val_model(model, val_loader, criterion)
        logger.info("Validation accuracy: %f", val_acc)
        logger.info("Validation loss: %f", val_loss)

        if SCHEDULER == "reduce":
            scheduler.step(val_acc)
        if SCHEDULER == "cos":
            scheduler.step()

        write_accuracy_loss(writer, train_loss, train_acc, epoch, is_training=True)
        write_accuracy_loss(writer, val_loss, val_acc, epoch, is_training=False)


        now = timer()
        epoch_time = now - pre
        pre = now
        logger.info("Epoch time: %f", epoch_time)
        logger.info("Current time: %f", now - start)

        write_best_acc(writer, best_accuracy, epoch)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            is_best = True

        save_checkpoint(model.state_dict(), epoch, is_best, checkpoint_path, SAVE_FREQ)

    logger.info("Best accuracy: %f", best_accuracy)
    logger.info("Total time: %f", timer() - start)

    #pred, probs, corrects = trainer.test_model(model, data_loader=test_loader)

    #write_pr_curve(writer, corrects, pred)
    writer.close()


if __name__ == '__main__':
    main()
