import logging
import torch
import os


def write_accuracy_loss(summary_writer, loss, correct, epoch, is_training=True):
    loss_text = "training loss" if is_training else "validation loss"
    accuracy_text = "training accuracy" if is_training else "validation accuracy"
    summary_writer.add_scalar(loss_text, loss, epoch)
    summary_writer.add_scalar(accuracy_text, correct, epoch)


def write_best_acc(summary_writer, best_acc, epoch):
    summary_writer.add_scalar('best_accuracy', best_acc, epoch)


def write_graph(summary_writer, model, input_tensor):
    summary_writer.add_graph(model, input_tensor)


def write_pr_curve(summary_writer, labels, predictions, num_classes=7):
    for i in range(num_classes):
        summary_writer.add_pr_curve('pr_curve', labels, predictions, i)


class Logger():
    def __init__(self, logfile='output.log'):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
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


def save_checkpoint(state, epoch, is_best, save_path, save_freq=10):
    filename = os.path.join(save_path, 'checkpoint_' + str(epoch) + '.tar')
    if epoch % save_freq == 0:
        torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, 'best_checkpoint.tar')
        torch.save(state, best_filename)
