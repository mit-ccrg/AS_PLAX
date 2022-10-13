import argparse
import torch
from options.loadJson import LoadJson

class Options():
    """
    Defines options used for both and test time.
    Gather information from arg parse and print them.
    """

    def __init__(self):
        self.is_initialized = False # indicate the options haven't been initialized

    def initialize(self, parser):
        """
        Defines options to model training and testing
        """
        # basic parameters
        parser.add_argument('--is_train', default='training', help='training or testing')
        parser.add_argument('--data_dir', required=False, help='directory where video data sit')
        parser.add_argument('--label_dir', required=False, help='direcotry where tabular data sit')
        parser.add_argument('--label_file', required=False, help='file name of tabular data')
        parser.add_argument('--save_dir', type=str, required=False, help='directory to save model checkpoints')
        parser.add_argument('--name', type=str, default='random_experiment',help='name of the experiment')
        parser.add_argument('--gpu_ids', type=str, default='', help='speicify which gpus to use.')
        parser.add_argument('--verbose', type=int, default=1, help='Verbosity mode. 0 = silent, 1 = results from epoch, 2 = print all ')
        parser.add_argument('--date', type=str, help='date of experiment')
        # model parameters
        parser.add_argument('--seed', type=int, default=1234, help='manual seed for reproducibility purposes')
        parser.add_argument('--model',type=str, default='cnn_3d', help='choose which model to train')
        parser.add_argument('--kernel_size', type=str, default='2-4-4', help='kernel size for 3d CNN, e.g. 2-4-4')
        parser.add_argument('--num_outcomes', type=int, default=1, help='number of outcomes')
        parser.add_argument('--loss_function', type=str, default="BCE", help='loss fuction [BCE | MSE]')
        # dataset parameters
        parser.add_argument('--dataset_name',type=str, default='echo_200', help='choose which dataset to train')
        parser.add_argument('--max_frame', type=int, default=18, help='maximum number of frames allowed')
        parser.add_argument('--if_split', type=str, default="asdf", help='if train test split during training')
        parser.add_argument('--input_size', type=str, default='18-480-480', help='input size, height-width-frames, e.g. 100-100-18')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='input device')
        parser.add_argument('--num_frames', type=int, default=10, help='number of frames used from each video')
        parser.add_argument('--transform', type=str, default="False", help='whether or not the transform should be applied')
        parser.add_argument('--ecg_aligned', type=str, default="False", help='whether or not the videos are aligned with ECG')
        parser.add_argument('--view', type=str, default='plax_plax', help='view of echo video')
        parser.add_argument('--view_model', type=str, default='supervised', help='which model generated the view result')
        parser.add_argument('--masked', type=str, default="False", help='whether or not the inputs should be padded')
        # file parameters
        parser.add_argument('--json_file', type=str, required=False, action=LoadJson, help='path to Json file with options')
        self.is_initialized = True
        return parser

    def gather_options(self):
        """
        Initialize parser with basic options
        """

        if not self.is_initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # get possible more options related to model
        # model_name = opt.model
        # to_be_implemented

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """
        print and save options in a file
        """
        if not self.is_initialized:
            self.parser = argparse.ArgumentParser()
            self.parser = self.initialize(self.parser)
        message = ''
        message += '----------------Options----------------\n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value!=default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>15}:{:<25}{}\n'.format(str(key), str(value), comment)
        message += '---------------End Options---------------\n'
        print(message)

        # save to a txt file
        # to_be_implemented

    def parse(self):
        """
        Parse options and set up gpu devices
        """

        opt = self.gather_options()
#         self.print_options(opt)
        return opt
