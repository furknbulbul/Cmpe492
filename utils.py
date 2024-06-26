import logging
import torch
import wandb
import os


class Logger():
    def __init__(self, logfile='output.log', use_wandb=False):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        self.use_wandb = use_wandb
        
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=self.logfile
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


    def train_log_wandb(self, epoch, loss, acc=None):
        if self.use_wandb:
            if acc is not None:
                wandb.log({"train": {"loss": loss, "acc": acc}}, step=epoch)
            else:
                wandb.log({"train": {"loss": loss}}, step=epoch)
    
    def val_log_wandb(self, epoch, loss, acc=None):
        if self.use_wandb:
            if acc is not None:
                wandb.log({"val": {"loss": loss, "acc": acc}}, step=epoch)
            else:
                wandb.log({"val": {"loss": loss}}, step=epoch)
    
    def test_log_wandb(self, accuracy):
        if self.use_wandb:
            wandb.log({"test": {"test_accuracy": accuracy}})
    
    


def save_checkpoint(state, epoch, is_best, save_path, save_freq=10):
    filename = os.path.join(save_path, 'checkpoint_' + str(epoch) + '.pt')
    if epoch % save_freq == 0:
        torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, "_" +str(epoch)+ 'best_checkpoint.pt')
        torch.save(state, best_filename)

    
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)



def upload_wandb(name, model, args):
    if args.use_wandb:
        torch.save(model.state_dict(), name + ".pth")
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(name + ".pth")
        wandb.log_artifact(artifact)


def save_confusion_matrix(cm, args):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if args.use_wandb:
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        wandb.log({"Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)