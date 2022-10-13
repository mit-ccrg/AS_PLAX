from options.options import Options

class TestOptions(Options):
	"""
	Includes testing options
	"""
	def initialize(self, parser):
		parser = Options.initialize(self, parser)
		parser.add_argument('--test_epoch', type=str, help='epoch of checkpoint for testing')
		parser.add_argument('--test_result_dir', type=str, help='dir to save testing result')
		self.isTrain = False
		return parser