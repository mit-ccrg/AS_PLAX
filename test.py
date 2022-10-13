import torch
import pandas as pd
from options.test_options import TestOptions
from datasets import create_dataset
from models import create_model
from sklearn.metrics import roc_auc_score

def save_output(label_test, model_output_test, file_name, save_dir):
	label_df = pd.DataFrame(columns=['FileName', 'TrueLabel', 'Output'])
	label_df['TrueLabel'] = label_test
	label_df['Output'] = model_output_test
	label_df['FileName'] = file_name
	label_df.to_csv(save_dir)

def main():
	opt = TestOptions().parse()
	opt.if_split =  (opt.if_split =="True")
	opt.masked = (opt.masked =="True")
	opt.transform = (opt.transform =="True")

	data_loaders = create_dataset(opt)
	test_loader = data_loaders.test_loader
	check_point_dir = opt.save_dir
	model = create_model(opt)
	model.network.load_state_dict(torch.load(check_point_dir + "checkpoint_epoch_" + str(opt.test_epoch)))
	print('Loaded model weights from ', check_point_dir + "checkpoint_epoch_" + str(opt.test_epoch))
	model.network.eval()
	# print('Test datasize = ', len(test_loader))
	with torch.no_grad():
		for j, data in enumerate(test_loader):
			# print(j)
			model.set_input(data)
			model.forward()
			model.test()
			if j == 0:
				model_output_test = model.output1.view(-1)
				label_test = model.input_label
				file_name = data['file_name']
				# print(file_name, type(file_name))
			else:
				model_output_test = torch.cat((model_output_test, model.output1.view(-1)))
				label_test = torch.cat((label_test,model.input_label))
				file_name = file_name+data['file_name']
	label_test = label_test.cpu()
	model_output_test = model_output_test.cpu()
	if opt.loss_function == 'BCE':
		auc_valid = roc_auc_score(label_test, model_output_test)
		print('Test Finished: AUC = ', auc_valid)
	save_dir = opt.test_result_dir + opt.name + '_epoch_' + str(opt.test_epoch) + '.csv'
	save_output(label_test, model_output_test, file_name, save_dir)
if __name__=="__main__":
	main()