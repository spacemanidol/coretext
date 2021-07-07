import os
import argparse
import numpy as np
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

def main(args):
    # Load model
    try:
        print("Loading models: {}".format(args.model_path))
        checkpoint = torch.load(args.model_path)
    except: 
        print("Models couldn't be loaded. Aborting")
        exit(0)
    
    # Deal With CUDA
    if args.cuda:
        device = args.cuda_device
        cudnn.benchmark = True
    else:
        device = 'cpu'
    with torch.no_grad():
        
    print("Done Predicting")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict ...')
    parser.add_argument('--output_file', type=str, default='output.txt', help='output file for model prediction.')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device to use')
    parser.add_argument('--model_path', default='models/', type=str, help='model path')
    args = parser.parse_args()
    main(args)    
