import numpy as np
import torch
from utils.utils import *
import os
# from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import matplotlib.pyplot as plt

from focal_loss.focal_loss import FocalLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

saved_pt_name = "breast_step2.pt"


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, stop_epoch=30, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, args, writer=None):
    cur = 1
    print('\nTraining!')
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn = FocalLoss(gamma=2)
        loss_fn = [0,0]
        loss_fn[0] = FocalLoss(gamma=1.25)
        loss_fn[1] = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            # model.load_state_dict(torch.load("/mnt/NAS_AI/ahmed/CLAM_2class/results/task_1_tumor_vs_normal_CLAM_50_s1/breast_histai_125_best.pt"))
            # print("model loaded: /mnt/NAS_AI/ahmed/CLAM_2class/results/task_1_tumor_vs_normal_CLAM_50_s1/breast_histai_125_best.pt")
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

#     print('\nSetup EarlyStopping...', end=' ')
#     if args.early_stopping:
#         early_stopping = EarlyStopping(patience = 10, stop_epoch=20, verbose = True)

#     else:
#         early_stopping = None
#     print('Done!')

#     for epoch in range(args.max_epochs):
#         if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
#             train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
#             stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
#                 early_stopping, writer, loss_fn, args.results_dir)
        
#         else:
#             train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
#             stop = validate(cur, epoch, model, val_loader, args.n_classes, 
#                 early_stopping, writer, loss_fn, args.results_dir)
        
#         if stop: 
#             break

#     if args.early_stopping:
#         model.load_state_dict(torch.load(os.path.join(args.results_dir, saved_pt_name)))
#     else:
#         torch.save(model.state_dict(), os.path.join(results_dir, saved_pt_name))
    
    
    # model.load_state_dict(torch.load("/mnt/NAS_AI/ahmed/CLAM_2class/results/task_1_tumor_vs_normal_CLAM_50_s1/breast_histai_125_best.pt"))
    # print("model loaded: /mnt/NAS_AI/ahmed/CLAM_2class/results/task_1_tumor_vs_normal_CLAM_50_s1/breast_histai_125_best.pt")
    
    # model.load_state_dict(torch.load("/mnt/NAS_AI/ahmed/CLAM/results/task_1_tumor_vs_normal_CLAM_50_s1/colorectal_new_withpretrain.pt"))
    # model.load_state_dict(torch.load("/mnt/NAS_AI/ahmed/CLAM/results/task_1_tumor_vs_normal_CLAM_50_s1/gastric_v3.pt"))

    # model.load_state_dict(torch.load("/mnt/NAS_AI/ahmed/CLAM_2class/results/task_1_tumor_vs_normal_CLAM_50_s1/breast_histai_125_best.pt"))
    model.load_state_dict(torch.load("/mnt/NAS_AI/ahmed/CLAM_2class/results/task_1_tumor_vs_normal_CLAM_50_s1/"+saved_pt_name))


    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    # results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    # print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.log_metric('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.log_metric('final/val_error', val_error, 0)
        writer.log_metric('final/val_auc', val_auc, 0)
        writer.log_metric('final/test_error', test_error, 0)
        writer.log_metric('final/test_auc', test_auc, 0)
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):

        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)


        acc_logger.log(Y_hat, label)
        loss = loss_fn[0](Y_prob, label)
        loss += loss_fn[1](logits, label)
        
        # loss = loss_fn(Y_prob, label)
        # loss = loss_fn(logits, label)

        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
        #         'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()


    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.log_metric('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.log_metric('train/loss', train_loss, epoch)
        writer.log_metric('train/error', train_error, epoch)
        writer.log_metric('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.log_metric('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.log_metric('train/loss', train_loss, epoch)
        writer.log_metric('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(Y_prob, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.log_metric('val/loss', val_loss, epoch)
        writer.log_metric('val/auc', auc, epoch)
        writer.log_metric('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, saved_pt_name))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            # loss = loss_fn(logits, label)
            # loss = loss_fn(Y_prob, label)
            
            loss = loss_fn[0](Y_prob, label)
            loss += loss_fn[1](logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.log_metric('val/loss', val_loss, epoch)
        writer.log_metric('val/auc', auc, epoch)
        writer.log_metric('val/error', val_error, epoch)
        writer.log_metric('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.log_metric('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, saved_pt_name))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    
    all_labels_list = []
    all_probs_list = []
    
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        Y = int(label)
        # all_labels[batch_idx] = Y
        # all_probs[batch_idx] = Y_prob
        prob = Y_prob[0][1].item()  # Assuming class 1 is positive

        all_labels_list.append(Y)
        all_probs_list.append(prob)

        if prob > 0.5:
            if Y == 1:
                tp += 1
            else:
                fp += 1
                print("FP:", slide_id, " prob: ", Y_prob)
        else:
            if Y == 0:
                tn += 1
            else:
                fn += 1
                print("FN:", slide_id, " prob: ", Y_prob)
                
    

#     fpr, tpr, thresholds = roc_curve(all_labels_list, all_probs_list)
#     roc_auc = calc_auc(fpr, tpr)

#     plt.figure(figsize=(6, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     plt.grid()
#     plt.tight_layout()

#     # Save the figure
#     plt.savefig('roc_curve_split_1_val.png', dpi=300)

#     # Optional: close or show the plot
#     # plt.show()  # Only if you want to view it interactively
#     plt.close()

# fp = 0
# fn = 0
# tp = 0
# tn = 0
# for batch_idx, (data, label) in enumerate(loader):
#     data, label = data.to(device), label.to(device)
#     slide_id = slide_ids.iloc[batch_idx]
#     with torch.no_grad():
#         logits, Y_prob, Y_hat, _, _ = model(data)
#     Y = int(label)    
#     if Y_prob[0][0] > 0.5:
#         if Y == 0:
#             tn += 1
#         elif Y == 1:
#             fn += 1
#             print("FN:", slide_id, " prob: ", Y_prob)
#     else:
#         if Y == 0:
#             fp += 1
#             print("FP:", slide_id, " prob: ", Y_prob)
#         elif Y == 1:
#             tp += 1

    # Y_pred = int(Y_hat)
    # Y = int(label)
    # if Y_pred != Y and Y == 1:
    #     print("FN:", slide_id)
    #     print(Y_prob)
    #     print(Y_hat)
    # if Y_pred != Y and Y == 0:
    #     print("FP:", slide_id)

    acc_logger.log(Y_hat, label)
    probs = Y_prob.cpu().numpy()
    all_probs[batch_idx] = probs
    all_labels[batch_idx] = label.item()

    patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
    # error = calculate_error(Y_hat, label)
    # test_error += error

    test_error /= len(loader)
    print("fp: ", fp)
    print("tp: ", tp)
    print("fn: ", fn)
    print("tn: ", tn)

    if n_classes == 2:
        # auc = roc_auc_score(all_labels, all_probs[:, 1])
        # aucs = []
        # print(all_labels)
        fpr, tpr, thresholds = roc_curve(all_labels_list, all_probs_list)
        roc_auc = calc_auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()

        # Save the figure
        plt.savefig('roc_curve_split_5_val.png', dpi=300)

        # Optional: close or show the plot
        # plt.show()  # Only if you want to view it interactively
        plt.close()
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
