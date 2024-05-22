from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("-e", "--epoch", default=120, type=int)
        parser.add_argument("-b", "--batch", default=64, type=int)
        parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
        parser.add_argument("-sf", "--save_freq", default=20, type=int)
        parser.add_argument("-l2", "--l2lambda", default=0, type=float)
        parser.add_argument("-o", "optimizer", default="adam", type=str)
        parser.add_argument("-d", "--dropout", default=0.2, type=float)
        parser.add_argument("-a", "--augmentation", action="store_true", help="use data augmentation")
        parser.add_argument("-sc", "--scheduler", default = "none", type=str, help="[reduce, cos]")

        self.isTrain = True