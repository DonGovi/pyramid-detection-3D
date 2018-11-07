from __future__ import print_function
import argparse
import pandas as pd
import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from loss import FocalLoss
from retinanet3d import RetinaNet
from data_generator import SPHDataset
import config


def get_data_list(label_df, ratio, split=True):
    '''
    Get train and val set samples' list

    Args:
      label_df: (DataFrame) label dataframe
      ratio: (int) train_num : val_num
      split: (boolean) Whether split trainset and valset or not 

    Returns:
      train_list: (list)
      val_list: (list)
    '''
    sample_ids = label_df["sample_id"].drop_duplicates().tolist()
    if split:
        num = len(sample_ids)
        train_num = int(num * (ratio/(1.+ratio)))
        train_idx = np.random.choice(num, train_num, replace=False)
        train_ids = [sample_ids[i] for i in train_idx]
        val_ids = [i for i in sample_ids if not i in train_ids]
        print("Train set number: %d, Val set number: %d" % (len(train_ids), len(val_ids)))
        return train_ids, val_ids
    else:
        return sample_ids


parser = argparse.ArgumentParser(description="Pytorch RetinaNet Training")
parser.add_argument("--exp", required=True, help="experiment name")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
args = parser.parse_args()

cfg = config.Config()

assert torch.cuda.is_available(), "Error: CUDA not found!"
train_best_loss = float("inf")      # best loss
train_best_loc_loss = float("inf")     # best loc loss
train_best_cls_loss = float("inf")     # best cls loss
val_best_loss = float("inf")      # best loss
val_best_loc_loss = float("inf")     # best loc loss
val_best_cls_loss = float("inf")     # best cls loss
start_epoch = 0              # start from epoch 0 or last epoch
lr = cfg.learning_rate_start 
val_count = 0

# Data
print("==> Preparing data...")
label_df = pd.read_csv(cfg.crop_64_label)
#print(label_df.shape)
trainlist, vallist = get_data_list(label_df=label_df, ratio=cfg.trainval_ratio, split=True)

trainset = SPHDataset(scan_path=cfg.crop_64_samples, scan_list=trainlist, label_df=label_df, 
                      threshold=cfg.norm_threshold, transform=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, 
                                          num_workers=2, collate_fn=trainset.collate_fn)
#print(len(trainloader))
valset = SPHDataset(scan_path=cfg.crop_64_samples, scan_list=vallist, label_df=label_df,
                    threshold=cfg.norm_threshold, transform=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, 
                                        num_workers=2, collate_fn=valset.collate_fn)

# Model
net = RetinaNet(backbone="resnet34", num_classes=cfg.num_cls)
if args.resume:
    print("Resuming from checkpoint...")
    checkpoint = torch.load(os.path.join(cfg.checkpoints_path, "ckpts_v27_change_anchor_strategy_val/", "ckpt.pth"))
    net.load_state_dict(checkpoint["net"])
    best_cls_loss = float("inf")
    best_loc_loss = float("inf")
    start_epoch = checkpoint["epoch"]-2
    lr = checkpoint["lr"]
    print("Resuming finish")
    print("best loc and cls loss:%f, %f" % (best_loc_loss, best_cls_loss))
    print("start_epoch: %d" % start_epoch)
    print("learning rate: %f" % lr)
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

criterion = FocalLoss(gamma=0, alpha=None, num_classes=cfg.num_cls)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, 
                                        weight_decay=cfg.weight_decay)
writer = SummaryWriter()

def train(epoch, num_step):

    net.train()
    global train_loss, train_loc_loss, train_cls_loss


    for batch_idx, (dict_list, inputs, loc_targets, cls_targets) in enumerate(trainloader):
        step = ((epoch-1)*len(trainloader) + batch_idx)
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss, loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), max_norm=1.2)
        optimizer.step()
        train_loss += loss.data[0]
        train_loc_loss += loc_loss.data[0]
        train_cls_loss += cls_loss.data[0]

        if epoch < start_epoch+2 and step%num_step == 0:
            print("train_loss: %.6f | train_loc_loss: %.6f | train_cls_loss: %.6f" % (train_loss/num_step,
                                                                                  train_loc_loss/num_step,
                                                                                  train_cls_loss/num_step))
            train_loc_loss = 0
            train_cls_loss = 0
            train_loss = 0
            

        elif epoch >= start_epoch+2:
            if step%num_step == 0 and step > 0:
                print("\nTraining epoch %d step %d" % (epoch, step))
                train_loss /= num_step
                train_loc_loss /= num_step
                train_cls_loss /= num_step
                writer.add_scalars("v27_data/loss", {"train_loss": train_loss}, step)
                writer.add_scalars("v27_data/loc_loss", {"train_loc_loss":train_loc_loss}, step)
                writer.add_scalars("v27_data/cls_loss", {"train_cls_loss": train_cls_loss}, step)
            
                print("train_loss: %.6f | train_loc_loss: %.6f | train_cls_loss: %.6f" % (train_loss,
                                                                                 	  train_loc_loss,
                                                                                 	  train_cls_loss))
            
                save_checkpoint(train_loc_loss, train_cls_loss, stage="train")
                train_loc_loss = 0
                train_cls_loss = 0
                train_loss = 0
            if step%val_step == 0 and step > 0:
                print("Val...")
                val_loss, val_loc_loss, val_cls_loss = val(epoch)	
                writer.add_scalars("v27_data/loss", {"val_loss": val_loss}, step)
                writer.add_scalars("v27_data/loc_loss", {"val_loc_loss":val_loc_loss}, step)
                writer.add_scalars("v27_data/cls_loss", {"val_cls_loss":val_cls_loss}, step)

def val(epoch):
    net.eval()
    val_loc_loss = 0
    val_cls_loss = 0
    val_loss = 0
    for batch_size, (dict_list, inputs, loc_targets, cls_targets) in enumerate(valloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss, loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        val_loc_loss += loc_loss.data[0]
        val_cls_loss += cls_loss.data[0]
        val_loss += loss.data[0]
        
    val_loss /= len(valloader)
    val_loc_loss /= len(valloader)
    val_cls_loss /= len(valloader)
    save_checkpoint(val_loc_loss, val_cls_loss, stage="val")
    print("val_loss: %.6f | val_loc_loss: %.6f | val_cls_loss: %.6f" % (val_loss,
                                                                        val_loc_loss,
                                                                        val_cls_loss))
    return val_loss, val_loc_loss, val_cls_loss


def save_checkpoint(loc_loss, cls_loss, stage="train"):
    global train_best_loss, train_best_loc_loss, train_best_cls_loss
    global val_best_loss, val_best_loc_loss, val_best_cls_loss
    
    if stage == "train":
        best_loss = train_best_loss
        best_loc_loss = train_best_loc_loss
        best_cls_loss = train_best_cls_loss
    else:
        best_loss = val_best_loss
        best_loc_loss = val_best_loc_loss
        best_cls_loss = val_best_cls_loss

    if cls_loss+loc_loss < best_loss:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "loc_loss": loc_loss,
            "cls_loss": cls_loss,
            "epoch": epoch,
            "lr": lr
        }
        ckpt_path = os.path.join(cfg.checkpoints_path,"ckpts_"+ args.exp + "_" + stage)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(state, os.path.join(ckpt_path+"/", "ckpt.pth"))
        if stage == "train":
            train_best_loss = loc_loss + cls_loss
            train_best_loc_loss = loc_loss
            train_best_cls_loss = cls_loss
        elif stage == "val":
            val_best_loss = loc_loss + cls_loss
            val_best_loc_loss = loc_loss
            val_best_cls_loss = cls_loss
        print("best_loss: %.4f" % (cls_loss+loc_loss))



train_loss = 0
train_loc_loss = 0
train_cls_loss = 0
num_step = 50
val_step = 500

if __name__ == '__main__':
     
    for epoch in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
        print(epoch)
        if epoch in cfg.lr_decay_epochs:
            lr *= 0.1
            print("Decay Learning Rate: %f" % lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if epoch == 100:
            print("Finetuning box subnet...")
            optimizer = optim.SGD(net.subnet_boxes.parameters(), lr=lr, momentum=cfg.momentum, 
                                        weight_decay=cfg.weight_decay)

        train(epoch, num_step)
    '''
    for epoch in range(start_epoch+1, start_epoch+cfg.pre_train_epochs+1):
        print(epoch)
        train(epoch, num_step)
        if epoch == (start_epoch+cfg.pre_train_epochs):
            state = {"net": net.state_dict(),
                     "loc_loss": float("inf"),
                     "cls_loss": float("inf"),
                     "epoch": epoch,
                     "lr": lr}    
            ckpt_path = os.path.join(cfg.checkpoints_path,"ckpts_"+ args.exp)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            torch.save(state, os.path.join(ckpt_path+"/", "ckpt.pth"))
            print("Saved pre-trained network")
    ''' 



