from tabnanny import verbose
import time
import torch
from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets import create_dataset
from models import create_model
from utils.visualizer import display_current_results
import os
import shutil 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score



    
def main():
    opt = TrainOptions().parse()
    opt.if_split =  (opt.if_split =="True")
    opt.masked = (opt.masked =="True")
    opt.transform = (opt.transform =="True")
    opt.is_continue = (opt.is_continue =="True")
    
    if opt.verbose > 0:
        TrainOptions().print_options(opt)
    
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.set_num_threads(10)
    data_loaders = create_dataset(opt)
    train_loader = data_loaders.train_loader
    valid_loader = data_loaders.valid_loader
    check_point_dir = opt.save_dir + '_' + opt.date + '/'
    print('Check points saved to ', check_point_dir)

    model = create_model(opt)
    start_epoch = 0
    if opt.is_continue:
        load_epoch = opt.continue_epoch
        if opt.continue_check_point_dir == 'None': # same task continue training
            start_epoch = load_epoch + 1
            model.network.load_state_dict(torch.load(check_point_dir + "checkpoint_epoch_" + str(load_epoch)))
        else: # transfer exisiting model to another task
            model.network.load_state_dict(torch.load(opt.continue_check_point_dir + "checkpoint_epoch_" + str(load_epoch)))
        print('Loaded model weights from epoch ', load_epoch)

    print("Size of training", len(train_loader), "size of testing", len(valid_loader))
    # real training
    total_iter = 0
    total_start_time = time.time()
    if not opt.is_continue:
        if os.path.exists(check_point_dir):
            shutil.rmtree(check_point_dir)
        os.mkdir(check_point_dir)
    tb = SummaryWriter(check_point_dir)
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=5, gamma=0.3)
    for epoch in range(start_epoch, opt.n_epochs):
        if opt.verbose == 2:
            print("----------------Training----------------")
        print("Epoch:",epoch)
        running_losses1 = 0
        model.network.train()
        for i, data in enumerate(train_loader):
            if i%200 == 0 and opt.verbose == 2:
                print(i)
            total_iter += opt.batch_size
            # update model
            model.set_input(data)
            model.optimize_parameters()
            running_losses1 += model.get_current_losses()[0]
            tb.add_scalar("Running Loss 1", running_losses1/(i+1), i)
        scheduler.step()
        if opt.verbose >= 1:
            print('Epoch {}, lr {}'.format(
            epoch, model.optimizer.param_groups[0]['lr']))
            
            
        if epoch % opt.print_freq == 0:
            time_passed = time.time() - total_start_time
            display_current_results(time_passed, epoch, running_losses1/len(train_loader))
            model.save_model(epoch)
            if  opt.verbose == 2:
                print("----------------Evaluating on Validation Set----------------")
            model.network.eval()
            valid_losses1 = 0
            model_output_valid = torch.empty(1)
            label_valid = torch.empty(1)
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    if  opt.verbose == 2:
                        print(j)
                    total_iter += opt.batch_size
                    # update model
                    model.set_input(data)
                    model.forward()
                    model.test()
                    valid_losses1 += model.get_current_losses()[0]
                    if j == 0:
                        model_output_valid = model.output1.view(-1)
                        label_valid = model.input_label
                    else:
                        model_output_valid = torch.cat((model_output_valid, model.output1.view(-1)))
                        label_valid = torch.cat((label_valid,model.input_label))
            time_passed = time.time() - total_start_time
            display_current_results(time_passed, epoch, valid_losses1/len(valid_loader))
            if opt.loss_function == 'BCE':
                auc_valid = roc_auc_score(label_valid.cpu(), model_output_valid.cpu())
            train_loss1=running_losses1/len(train_loader)
            valid_loss1 = valid_losses1/len(valid_loader)

            if opt.verbose >= 1:
                print("Epoch", epoch) 
                print("- train_loss1:", train_loss1.item())
                print("- valid_loss1:", valid_loss1.item())
                if opt.loss_function == 'BCE':
                    print("- valid_auc1:", auc_valid)


     #       print("Epoch", epoch)  
            tb.add_scalar("Train Loss", running_losses1/len(train_loader), epoch)
            tb.add_scalar("Valid Loss", valid_losses1/len(valid_loader), epoch)
            if opt.loss_function == 'BCE':
                tb.add_scalar("Valid AUC", auc_valid, epoch)
    tb.close()


if __name__=="__main__":
	main()