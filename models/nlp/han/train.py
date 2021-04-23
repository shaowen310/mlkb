# Adapted from https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/train.py
import os
import argparse
import shutil
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics
# from tensorboardX import SummaryWriter

from dataset import HANDataset
from han import HierAttNet
import loggingutil
from paramstore import ParamStore


def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []

    mapped_doc = None
    with open(data_path, 'rb') as f:
        mapped_doc = pickle.load(f)

    for (_, doc) in mapped_doc:
        sent_length_list.append(len(doc))
        for sent in doc:
            word_length_list.append(len(sent))

    sorted_word_length = sorted(word_length_list)
    sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8 * len(sorted_word_length))], sorted_sent_length[int(
        0.8 * len(sorted_sent_length))]


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification"""
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=0.0,
        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
        help=
        "Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique."
    )
    parser.add_argument("--train_set", type=str, default="data_/ag_news/train.pkl")
    parser.add_argument("--test_set", type=str, default="data_/ag_news/test.pkl")
    parser.add_argument("--test_interval",
                        type=int,
                        default=1,
                        help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data_/ag_news/i2w.pkl")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="model_")
    args = parser.parse_args()
    return args


def train(opt):
    opt['saved_path'] = os.path.join(opt['saved_path'], opt['id_'])
    logger = loggingutil.get_logger(opt['id_'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if not os.path.isdir(opt['saved_path']):
        os.makedirs(opt['saved_path'])
    output_file = open(opt['saved_path'] + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(opt))
    training_params = {"batch_size": opt['batch_size'], "shuffle": True, "drop_last": True}
    test_params = {"batch_size": opt['batch_size'], "shuffle": False, "drop_last": False}

    max_word_length, max_sent_length = get_max_lengths(opt['train_set'])
    training_set = HANDataset(opt['train_set'], opt['word2vec_path'], max_sent_length,
                              max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    test_set = HANDataset(opt['test_set'], opt['word2vec_path'], max_sent_length, max_word_length)
    test_generator = DataLoader(test_set, **test_params)

    model = HierAttNet(opt['word_hidden_size'],
                       opt['sent_hidden_size'], opt['batch_size'], training_set.num_classes,
                       len(training_set.index_to_word), max_sent_length, max_word_length)

    if os.path.isdir(opt['log_path']):
        shutil.rmtree(opt['log_path'])
    os.makedirs(opt['log_path'])
    # writer = SummaryWriter(opt['log_path)
    # writer.add_graph(model, torch.zeros(opt['batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=opt['lr'],
                                momentum=opt['momentum'])
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt['num_epoches']):
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(),
                                              predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])
            if not ((iter + 1) % 100):
                logger.debug(
                    "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                        epoch + 1, opt['num_epoches'], iter + 1, num_iter_per_epoch,
                        optimizer.param_groups[0]['lr'], loss, training_metrics["accuracy"]))
            # writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            # writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        if epoch % opt['test_interval'] == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label,
                                          te_pred.numpy(),
                                          list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".
                format(epoch + 1, opt['num_epoches'], te_loss, test_metrics["accuracy"],
                       test_metrics["confusion_matrix"]))

            logger.info("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1, opt['num_epoches'], optimizer.param_groups[0]['lr'], te_loss,
                test_metrics["accuracy"]))
            # writer.add_scalar('Test/Loss', te_loss, epoch)
            # writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if te_loss + opt['es_min_delta'] < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt['saved_path'] + os.sep + "whole_model_han.pt")

            # Early stopping
            if epoch - best_epoch > opt['es_patience'] > 0:
                logger.info("Stop training at epoch {}. The lowest loss achieved is {}".format(
                    epoch, te_loss))
                break

            output_file.close()


if __name__ == '__main__':
    opt = vars(get_args())

    model_name = 'HAN'

    pstore = ParamStore()
    pstore.add(model_name, opt)

    train(opt)
