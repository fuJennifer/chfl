#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#mnist & cifar

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random
from collections import deque
import datetime

import torch
torch.cuda.empty_cache()
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CharLSTM
from utils import get_dataset, average_weights, exp_details
from shkp import ShakeSpeare



if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.gpu:
        torch.cuda.set_device(args.gpu) # if error here, it means you have no suitable torch installed, please refer https://pytorch.org/get-started/locally/
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    if args.dataset in ['mnist', 'cifar']:
        train_dataset, test_dataset, user_groups = get_dataset(args)
    elif args.dataset == 'shakespeare':
        train_dataset = ShakeSpeare(train=True)
        test_dataset = ShakeSpeare(train=False)
        user_groups = train_dataset.get_client_dic()
        args.num_users = len(user_groups) # 139 users
        print(args.num_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    
    # For MNIST and CIFAR10, each edge has 5 clients
    # For Shakespeare, each edge has 14 clients and the last edge has 13
    num_edges = args.num_edges # 10 
    step = args.step  # clients/num_edges 5  

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,dim_out=args.num_classes)
    
    elif args.model == 'lstm':
        global_model = CharLSTM(args=args)
        args.num_users = 139
        # step = 14 # 139 clients / 10 edge servers
        step = 12 # 12 edges, first 11 edge have 12 clients, last edge has 7 clients
        args.num_edges = 12
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training Parameters Initialization
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    test_accuracy = []
    
    # write the result to file
    result_fn = args.pattern +"_"+ args.dataset +"_"+ str(args.hety) +"_"+ str(args.pg) + "_" + str(args.pc) + "_" + str(args.select_edges) + "_" + str(args.local_ep)+"_" +str(args.edge_ep)+"_" + str(args.epochs) +"_" + str(args.num_users)+ "_"+str(datetime.datetime.now())
    # result_fn = args.dataset +"_"+ str(args.local_ep)+"_" + str(args.epochs) +"_" + str(args.num_users)+"_" + "_"+str(datetime.datetime.now())
    with open('result.txt','a') as sr:
        sr.write(result_fn)
        sr.write('\n')
        sr.close()
    
    # JJ inital queue for global model candidates
    global_models_q = deque()
    global_models_q.append(copy.deepcopy(global_model))
    
    # JJ inital models for all edge servers
    edge_models = [copy.deepcopy(global_model)]*num_edges
    
    if args.pattern == 'e2e':
        edge_model = copy.deepcopy(global_model)
        e2e_weights = global_model.state_dict()
        
    # Start training
    for epoch in tqdm(range(args.epochs)):
        global_model.train()
        selected_edges_weights = []
        
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # JJ select some edges to do training
        selected_edges = random.sample(range(num_edges),args.select_edges) # range creates sequence from 0-9 for edge servers
        if args.pattern == 'e2e':
            selected_edges = range(num_edges)
        #print(selected_edges)
        if args.pattern == 'e2s':
            # JJ initilized edge server models with global models
            global_models_list = list(global_models_q)
            for index, edge_index in enumerate(selected_edges):
                edge_models[edge_index] = copy.deepcopy(global_models_list[min(index,len(global_models_list)-1)])
        for edge_index in selected_edges:
            # get clients for each selected edge server
            if args.pattern == 'e2s':
                idxs_users = range(edge_index*step,min(len(user_groups),(edge_index+1)*step))
                edge_model = copy.deepcopy(edge_models[edge_index])
            else:
             # for e2e, it needs to sampling the clients in each edge
                if args.model == 'lstm':
                    idxs_users = np.random.randint(edge_index*step,min(len(user_groups),(edge_index+1)*step),size = 1)
                else:
                    idxs_users = np.random.randint(edge_index*step,min(len(user_groups),(edge_index+1)*step),size = int(step*args.frac))
                # for e2e, it needs to sampling the clients in each edge
                #idxs_users = np.random.randint(edge_index*step,min(len(user_groups),(edge_index+1)*step),size = int(step*args.frac))
                edge_model.load_state_dict(e2e_weights)
            edge_model.train().to(device)
            # FedAvg for all clients in edge server
            for ed_ep in range(args.edge_ep):
                local_weights = []
                for idx in idxs_users:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(model=copy.deepcopy(edge_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    #local_losses.append(copy.deepcopy(loss))
                edge_weights = average_weights(local_weights) # FedAvg weights
                edge_model.load_state_dict(edge_weights) # update edge server
            if args.pattern == 'e2e':
                e2e_weights = edge_weights
            else:
                selected_edges_weights.append(copy.deepcopy(edge_weights))
            print('finish training for one edge!')
        
        if args.pattern == 'e2s':   
            global_weights = average_weights(selected_edges_weights)
            # update global weights
            global_model.load_state_dict(global_weights)
            global_models_q.append(copy.deepcopy(global_model))
            # Maintain a dqueue with a size same as selected edges, and pop the oldest global model if needed
            if len(global_models_q) >= args.select_edges:
                global_models_q.popleft()
        else:
            global_model.load_state_dict(e2e_weights)
            
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        
        #print("|---- Test Loss: {}".format(test_loss))
        print('| Global Round : {}|---- Test Accuracy: {:.2f}%'.format(epoch,100*test_acc))
        test_accuracy.append(test_acc)
    del global_model
    with open('result.txt','a') as sr:
        for item in test_accuracy:
            acc_str = str(item)
            sr.write(acc_str)
            sr.write(',')
        sr.write('\n\n')
       
    
