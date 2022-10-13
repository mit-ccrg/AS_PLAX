from options.options import Options

class TrainOptions(Options):
    """
    Includes basic options and other train options
    """
    def initialize(self, parser):
        parser = Options.initialize(self, parser)
        parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs for training')
        parser.add_argument('--print_freq', type=int, default=1, help='number of epochs to print loss')
        parser.add_argument('--lr', type=float, default=0.0001, help='inital learning rate for adam')
        parser.add_argument('--optimizer', type=str, default="adam", help='type of optimizer')
        parser.add_argument('--weight_decay', type=float, default=.0001, help='regularization factor')
        parser.add_argument('--lr_strategy', type=str, default='linear', help='strategy of learning rate decay. [linear | step | plateau]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='decay learning rate for every lr_decay_iters')
        parser.add_argument('--train_size', type=float, default=.8, help='percent of dataset used for training dataset in decimal formal')
        parser.add_argument('--pre_train', type=bool, default=False, help='if pre-train')
        parser.add_argument('--momentum', type=float, default=0, help='momentum')
        parser.add_argument('--is_continue', type=str, default="False", help='whether or not continue training from a checkpoint')
        parser.add_argument('--continue_check_point_dir', type=str, default="None", help='which checkpoint should model load if continue training')
        parser.add_argument('--continue_epoch', type=int, default="0", help='which checkpoint epoch should model load if continue training')


        self.isTrain = True
        return parser