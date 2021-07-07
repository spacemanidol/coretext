import random
import os
import argparse
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn

from utils import CustomDataset, set_seed

def save_checkpoint(args, model, model_optimizer, epoch):
    print("Saving model")
    state = {'epoch': epoch,
             'model': model, 
             'model_optimizer' : model_optimizer}
    torch.save(state, args.model_path +str(epoch))

def main(args):
    set_seed(args.seed)
    # Load Checkpoint if exists
    start_epoch = 0
    if args.load:
        try:
            print("Loading models: {}".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            start_epoch = checkpoint['epoch'] + 1
            model = checkpoint['model']
            model_optimizer = checkpoint['model_optimizer']
            print("Model Loaded")
        except: 
            print("Model couldn't be loaded. Aborting")
            exit(0)
    else:
        model = model()
        model_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        print("Model Loaded")
   
    # Deal With CUDA
    if args.cuda:
        device = args.cuda_device
        cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print("There are", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    else:
        device = 'cpu'
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.do_eval:
        print("Loading Validation Dataset")
        val_dataset =  CustomDataset(args.data_dir, 'validation')
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print("There are {} validation data examples".format(len(val_dataset)))
        print("Evaluating Model")
        validation_score = validate(args)
        print("Model performance is {}".format(validation_score))
        
    if args.do_train:
        validation_score, is_best = 0, False #in case we arent evaluating        
        print("Loading Training Dataset")
        train_dataset = CustomDataset(args.data_dir, 'training')
        print("There are {} training data examples".format(len(train_dataset)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        print("Traing model")
        for epoch in range(start_epoch, args.epochs):
            print("Starting epoch {}".format(epoch))    
            train(args)
            if args.do_eval:
                validation_score = validate(args)
                is_best= False
                if validation_score > best_validation_score:
                    is_best = True
                    best_validation_score = validation_score
            save_checkpoint(args, epoch, validation_score, is_best)

def train(args):
    return 0
                
def validate(args):
    return 0
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and validate model')
    parser.add_argument('--do_train', action='store_true', help='train the model')
    parser.add_argyment('--do_eval', action='store_true', help='evaluate the model')
    parser.add_argument('--data_dir', default='data/', type=str, help='directory of data to be processed')
    parser.add_argument('--epochs', default=10, type=int, help='Train epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of sampled batch')
    parser.add_argument('--dropout', default=0.5, type=float, help='Rate of dropout')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device to use. aka gpu')
    parser.add_argument('--model_path', default='models/', type=str, help='model path')
    parser.add_argument('--load', action='store_true', help='load existing model')
    parser.add_argument('--print_freq', default=5000, type=int, help="print model performance every n batches")
    parser.add_argument('--seed', default=42, type=int, help='Set random seed')
    parser.add_argument('--checkpoint_freq', default=1000, type=int, help='how often to checkpoint model')
    args = parser.parse_args()
    main(args)
