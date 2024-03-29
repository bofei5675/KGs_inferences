import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle
from tools import save_mbeddings

seed = 141

# %%
# %%from torchviz import make_dot, make_dot_from_trace

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/WN18RR/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=8, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=2, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=str2bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-4)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/wn/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=5000, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[1, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=64, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")
    args.add_argument("-load_conv", "--load_conv", type=str, default=None,
                      dest='load_conv')
    args.add_argument("-load_gat", "--load_gat", type=str, default=None,
                      dest='load_gat')
    args.add_argument("-tanh", "--tanh", type=str2bool, default='yes', dest='tanh')

    args.add_argument('--debug', default=False, action='store_true',
                        help='debug mode')

    args = args.parse_args()
    return args


args = parse_args()
# %%
print('Using pretrained:', args.pretrained_emb)

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, args.id2entity, args.id2relation, headTailSelector, unique_entities_train, unique_relations_train = build_data(
        args.data, is_unweigted=False, directed=False)
    print('Training size', len(train_data), 'Val size', len(validation_data), 'Test size', len(test_data))
    if args.pretrained_emb:
        # no relation embedding for us now
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 None)
        if entity_embeddings.shape[0] == 0:
            entity_embeddings = np.random.randn(
                len(entity2id), args.embedding_size)
        if relation_embeddings.shape[0] == 0:
            relation_embeddings = np.random.randn(
                len(relation2id), args.embedding_size)
        print("Initialised relations and entities from SSP")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, unique_relations_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


Corpus_, entity_embeddings, relation_embeddings = load_data(args)


if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)
    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples)
    if CUDA:
        y = y.cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def write_to_file(args, line):
    output_folder = args.output_folder
    with open(output_folder + 'log.txt', 'a+') as f:
        f.write(line + '\n')


def train_gat(args):

    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    if args.tanh:
        model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                    args.drop_GAT, args.alpha, args.nheads_GAT, 'tanh')
    else:
        model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                    args.drop_GAT, args.alpha, args.nheads_GAT, 'leakyrelu')

    model_gat = nn.DataParallel(model_gat)


    if CUDA:
        model_gat.cuda()

    if args.load_gat is not None and args.epochs_gat == 0:
        model_gat.load_state_dict(torch.load(
            '{0}gat/trained_{1}.pth'.format(args.output_folder, args.epochs_gat - 1)))
    elif args.load_gat is not None and args.epochs_gat > 0:
        model_gat.load_state_dict(torch.load(args.load_gat))

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([], dtype=torch.long)
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train, node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1
        if args.debug:
            pass
        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)
            # print('Forward pass', entity_embed.shape, relation_embed.shape)
            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            line = "Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item())
            print(line)
            write_to_file(args, line)

        scheduler.step()
        line = "Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time)
        print(line)
        write_to_file(args, line)

        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        save_model(model_gat, args.data, epoch,
                   args.output_folder + 'gat/')
    if CUDA:
        final_entity_embeddings = model_gat.module.final_entity_embeddings.cpu().detach().numpy()
        final_relation_embeddings = model_gat.module.final_relation_embeddings.cpu().detach().numpy()

    else:
        final_entity_embeddings = model_gat.module.final_entity_embeddings.detach().numpy()
        final_relation_embeddings = model_gat.module.final_relation_embeddings.detach().numpy()
    save_mbeddings(args, final_entity_embeddings, final_relation_embeddings)

def train_conv(args):

    # Creating convolution model here.
    ####################################

    print("Defining model")
    if args.tanh:
        model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                    args.drop_GAT, args.alpha, args.nheads_GAT, 'tanh')
    else:
        model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                    args.drop_GAT, args.alpha, args.nheads_GAT, 'leakyrelu')
    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    model_gat = nn.DataParallel(model_gat)
    model_conv = nn.DataParallel(model_conv)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()
    # load gat weights given pretrained
    if (args.load_gat is None) or (args.load_gat is not None and args.epochs_gat > 0):
        model_gat.load_state_dict(torch.load(
            '{0}gat/trained_{1}.pth'.format(args.output_folder, args.epochs_gat - 1)))
    else:
        model_gat.load_state_dict(torch.load(args.load_gat))

    if isinstance(model_conv, nn.DataParallel):
        if args.load_conv is None:
            model_conv.module.final_entity_embeddings = model_gat.module.final_entity_embeddings
            model_conv.module.final_relation_embeddings = model_gat.module.final_relation_embeddings
        else:
            model_conv.load_state_dict(torch.load(args.load_conv))
    else:
        model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
        model_conv.final_relation_embeddings = model_gat.final_relation_embeddings

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            for param in model_conv.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()
            line = "Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item())
            print(line)
            write_to_file(args, line)

        scheduler.step()
        line = "Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time)
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
        print(line)
        write_to_file(args, line)

        save_model(model_conv, args.data, epoch,
                   args.output_folder + "conv/")


def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv = nn.DataParallel(model_conv)

    if CUDA:
        model_conv.load_state_dict(torch.load(
            '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)))
        model_conv.cuda()
    else:
        model_conv.load_state_dict(torch.load(
            '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1), map_location=torch.device('cpu')))
    model_conv.eval()
    with torch.no_grad():
        if isinstance(model_conv, nn.DataParallel):
            ### original code is get_validation_pred
            Corpus_.get_validation_pred_relation(args, model_conv.module, unique_entities)
        else:
            Corpus_.get_validation_pred_relation(args, model_conv, unique_entities)

        if isinstance(model_conv, nn.DataParallel):
            Corpus_.get_validation_pred(args, model_conv.module, unique_entities)
        else:
            Corpus_.get_validation_pred(args, model_conv, unique_entities)


if (args.load_gat is None) or (args.load_gat is not None and args.epochs_gat > 0):
    train_gat(args)

train_conv(args)
evaluate_conv(args, Corpus_.unique_entities_train)
# evaluate_gat(args)
