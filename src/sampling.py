#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

# Non-IID for edge heterogeneity, IID for user heterogeneity.
def mnist_noniid_edge(dataset, args):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_edges = args.num_edges
    step = args.step
    pg = args.pg
    num_users = args.num_users
    
    num_shards, num_imgs = 200, 300
    
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    #labels = np.array(dataset.targets)
    uniq_labels = np.unique(labels)
    #samples_per_label = 6000
    
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # construct (index,label)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :] # Now the label is from 0-9, but index is not in order.
    
    # Save the first index for each label
    unique_labels, first_appearance_indices = np.unique(idxs_labels[1,:], return_index=True)
    #print(":", unique_labels)
    #print("Indices of First Appearance:", first_appearance_indices)    
    
    # How many samples should one user has?
    num_samples_user = num_shards * num_imgs/(num_edges*step)
    # How many samples should one label has in one user? Find the average label samples and total samples of one label
    num_samples_ulabel = int(np.minimum(num_samples_user/pg,num_shards * num_imgs/len(uniq_labels)))
    
    # divide and assign for each edge
    for i in range(num_edges):
        # random select pg labels for each edge
       edge_labels_set = set(np.random.choice(len(uniq_labels), pg, replace=False))
       # users index in current edge
       idxs_users_edge = range(i*step,min(num_users,(i+1)*step))
       # based on selected labels in this edge, assign label sample to this edge.
       for j in idxs_users_edge:
            for edge_label in edge_labels_set:
                if edge_label == 9:
                    end_indice = 60000
                else:
                    end_indice = first_appearance_indices[edge_label+1]
                rand_idx = np.random.randint(first_appearance_indices[edge_label],end_indice-num_samples_ulabel) 
                dict_users[j] = np.concatenate((dict_users[j], idxs[rand_idx:(rand_idx+num_samples_ulabel)]), axis=0)
    return dict_users

# Non-IID user data heterogeneity, IID for edge heterogeneity.
def mnist_noniid_user(dataset,args):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_edges = args.num_edges
    step = args.step
    pc = args.pc
    num_users = args.num_users
    
    num_shards, num_imgs = 200, 300
    
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    uniq_labels = np.unique(labels)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # construct (index,label)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :] # Now the label is from 0-9, but index is not in order.
    
    # Save the first index for each label
    unique_labels, first_appearance_indices = np.unique(idxs_labels[1,:], return_index=True)
    #print("Unique Labels:", unique_labels)
    #print("Indices of First Appearance:", first_appearance_indices)    
    
    # How many samples should one user have?
    num_samples_user = num_shards * num_imgs/(num_edges*step)
    # How many samples should one label have in one user? Find the average label samples and total samples of one label
    num_samples_ulabel = int(np.minimum(num_samples_user/pc,num_shards * num_imgs/len(uniq_labels)))
    
    # divide and assign for each edge
    for i in range(num_edges):
        # For each edge, it should include samples with all labels.
       # users index in current edge
       idxs_users_edge = range(i*step,min(num_users,(i+1)*step))
       # For the user in each client, it should includes sample with random pc labels
       for j in idxs_users_edge:
            # random select pc labels for each user
            user_labels_set = set(np.random.choice(len(uniq_labels), pc, replace=False))
            for user_label in user_labels_set:
                if user_label == 9:
                    end_indice = int(num_shards*num_imgs)
                else:
                    end_indice = first_appearance_indices[user_label+1]
                rand_idx = np.random.randint(first_appearance_indices[user_label],end_indice-num_samples_ulabel) 
                dict_users[j] = np.concatenate((dict_users[j], idxs[rand_idx:(rand_idx+num_samples_ulabel)]), axis=0)      
    return dict_users

# Non-IID for edge heterogeneity, IID for user heterogeneity.
def cifar_noniid_edge(dataset, args):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """ 
    num_edges = args.num_edges
    step = args.step
    pg = args.pg
    num_users = args.num_users
    
    num_shards, num_imgs = 200, 250
    
    idx_shard = [i for i in range(num_shards)]
    #dict_edges = {i: np.array([]) for i in range(num_edges)}
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    num_labels = np.unique(labels)
    samples_per_label = 5000
    
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # construct (index,label)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :] # Now the label is from 0-9, but index is not in order.
    
    # Save the first index for each label
    unique_labels, first_appearance_indices = np.unique(idxs_labels[1,:], return_index=True)
    #print("Unique Labels:", unique_labels)
    #print("Indices of First Appearance:", first_appearance_indices)    
    
    # How many samples should one user has?
    num_samples_user = int(num_shards * num_imgs/(num_edges*step))
    print("How many samples should one user has:",num_samples_user)
    # How many samples should one label have in one user? Find the average label samples and total samples of one label
    num_samples_ulabel = int(np.minimum(num_samples_user/pg,num_shards * num_imgs/len(num_labels)))
    print("How many samples should one label have in one user:",num_samples_ulabel)
    
    # divide and assign for each edge
    for i in range(num_edges):
        # random select pg labels for each edge
        edge_labels_set = set(np.random.choice(len(num_labels), pg, replace=False))
        #users index in current edge
        idxs_users_edge = range(i*step,min(num_users,(i+1)*step))
        #based on selected labels in this edge, assign all selected label sample to users in this edge.
        for j in idxs_users_edge:
            #print("user in current edge:", j)
            for edge_label in edge_labels_set:
                rand_idx = np.random.randint(first_appearance_indices[edge_label],first_appearance_indices[edge_label]+samples_per_label-num_samples_ulabel) 
                dict_users[j] = np.concatenate((dict_users[j], idxs[rand_idx:(rand_idx+num_samples_ulabel)]), axis=0)
    return dict_users


# Non-IID user data heterogeneity, IID for edge heterogeneity.
def cifar_noniid_user(dataset, args):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_edges = args.num_edges
    step = args.step
    pc = args.pc
    num_users = args.num_users
    
    num_shards, num_imgs = 200, 250
    samples_per_label = 5000
    
    idx_shard = [i for i in range(num_shards)]
    #dict_edges = {i: np.array([]) for i in range(num_edges)}
    idxs = np.arange(num_shards*num_imgs)
    dict_users = {i: np.array([]) for i in range(num_users)}
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    uniq_labels = np.unique(labels)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # construct (index,label)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :] # Now the label is from 0-9, but index is not in order.
    
    # Save the first index for each label
    unique_labels, first_appearance_indices = np.unique(idxs_labels[1,:], return_index=True)
    #print("Unique Labels:", unique_labels)
    #print("Indices of First Appearance:", first_appearance_indices)    
    
    # How many samples should one user have?
    num_samples_user = num_shards * num_imgs/(num_edges*step)
    print("num_samples_user:",num_samples_user)
    # How many samples should one label have in one user? Find the average label samples and total samples of one label
    num_samples_ulabel = int(np.minimum(num_samples_user/pc,num_shards * num_imgs/len(uniq_labels)))
    print("num_samples_ulabel:",num_samples_ulabel)
    
    # divide and assign for each edge
    for i in range(num_edges):
        # For each edge, it should include samples with all labels.
       # users index in current edge
       idxs_users_edge = range(i*step,min(num_users,(i+1)*step))
       # For the user in each client, it should includes sample with random pc labels
       for j in idxs_users_edge:
            # random select pc labels for each user
            user_labels_set = set(np.random.choice(len(uniq_labels), pc, replace=False))
            for user_label in user_labels_set:
                rand_idx = np.random.randint(first_appearance_indices[user_label],first_appearance_indices[user_label]+samples_per_label-num_samples_ulabel) 
                dict_users[j] = np.concatenate((dict_users[j], idxs[rand_idx:(rand_idx+num_samples_ulabel)]), axis=0)      
    return dict_users

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,size=num_users)
    random_shard_size = np.around(random_shard_size /sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


    
    

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


    


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
