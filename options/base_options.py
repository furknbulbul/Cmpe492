import argparse

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        # dataset
        parser.add_argument("-dr", "--data_root", default="Data/FER2013_small", type=str)

        # logging
        parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
        parser.add_argument("--wandb_project", default="emotion-recognition", type=str)

        parser.add_argument("--use_tensorboard", action="store_true", help="use tensorboard for logging")
        parser.add_argument("--tensorboard_dir", default="tensorboard_logs", type=str)
        parser.add_argument("--log_file", default="output.log", type=str)


        
        parser.add_argument("--seed", default=42, type=int)

        # model 
        parser.add_argument("--model", default="vgg", type=str)
        parser.add_argument("--config", default="vgg11", type=str)
        parser.add_argument("--no_dropout", action="store_true", help="no dropout")
        

        self.initialized = True


    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            self.initialize(parser)
        opt, _ = parser.parse_known_args()
        return opt
    


    def print_options(parser, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)



