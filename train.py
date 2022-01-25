# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2021/10/30 8:36
# @Software: PyCharm
# @Brief: Training script

from utils.parser import args
from utils.helpers import *
from utils.metrics import generate_matrix, get_mean_iou, get_recall, get_precision
from utils.dataloaders import get_data_loader

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_torch(seed=777)

    # Create model weights and summary save directory
    remove_dir_and_create_dir(args.log_dir)
    writer = [SummaryWriter(args.log_dir)]
    for i in range(args.num_classes):
        writer.append(SummaryWriter(args.log_dir + "/class_{}".format(i)))

    train_loader = get_data_loader(args.dataset_train_dir, args.batch_size, aug=True)
    val_loader = get_data_loader(args.dataset_val_dir, args.batch_size, aug=False)

    model = load_model(args, dev)

    # Get loss function
    criterion = get_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learn_rate_init)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs * len(train_loader),
                                                           eta_min=args.learn_rate_end)

    step = 0
    best_iou = 0.
    best_recall = 0.
    best_precision = 0.
    image_write_step = len(train_loader) // 2

    for epoch in range(args.epochs):

        # training
        model.train()
        tbar = tqdm(train_loader)
        total_cm = np.zeros((args.num_classes,)*2)
        train_loss = []

        for images_l, images_r, labels in tbar:
            tbar.set_description("epoch {}".format(epoch))

            # Set variables for training
            images_l = images_l.float().to(dev)
            images_r = images_r.float().to(dev)
            labels = labels.long().to(dev)

            # Zero the gradient
            optimizer.zero_grad()

            # Get model predictions, calculate loss
            cd_preds = model(images_l, images_r)

            cd_loss = criterion(cd_preds, labels)
            cd_loss.backward()
            optimizer.step()

            cd_preds = torch.argmax(cd_preds, 1)
            labels = labels.squeeze(1)

            images_l = images_l.cpu().numpy()
            images_r = images_r.cpu().numpy()
            labels = labels.cpu().numpy()
            cd_preds = cd_preds.cpu().numpy()
            cd_loss = cd_loss.data.cpu().numpy()

            # Calculate confusion matrix get recall, precision and mean iou
            cm = generate_matrix(labels, cd_preds,
                                 num_classes=args.num_classes)

            # Recorde model metrics
            total_cm += cm
            mean_iou, ious = get_mean_iou(total_cm)
            mean_recall, recalls = get_recall(total_cm)
            mean_precision, precisions = get_precision(total_cm)
            train_loss.append(cd_loss)

            # Write training information to tensorboard
            writer[0].add_scalar("loss", cd_loss, step)
            writer[0].add_scalar("mean_iou", mean_iou, step)
            writer[0].add_scalar("mean_recall", mean_recall, step)
            writer[0].add_scalar("mean_precision", mean_precision, step)
            writer[0].add_scalar("learning_rate", scheduler.get_lr(), step)
            for i in range(args.num_classes):
                writer[i+1].add_scalar("iou", ious[i], step)
                writer[i+1].add_scalar("recall", recalls[i], step)
                writer[i+1].add_scalar("precisions", precisions[i], step)

            if step % image_write_step == 0:
                masks = np.where(labels == 255, True, False) | np.where(labels == 0, True, False)
                labels = np.where(masks, 0, 1)

                for i in range(len(labels)):
                    image_l = images_l[i].transpose(1, 2, 0) * 255
                    image_r = images_r[i].transpose(1, 2, 0) * 255
                    label = labels[i].astype(np.uint8)

                    cd_pred = cd_preds[i].astype(np.uint8)

                    label = label_to_color_image(label)
                    cd_pred = label_to_color_image(cd_pred)

                    image = np.vstack((np.hstack((image_l, image_r)),
                                       np.hstack((label, cd_pred)))).astype(np.uint8)

                    writer[0].add_image('images_{}'.format(i), image, global_step=step, dataformats="HWC")

            # Clear batch variables from memory
            del images_l, images_r, labels

            scheduler.step()
            step += 1
            tbar.set_postfix(loss="{:.4f}".format(cd_loss),
                             m_iou="{:.4f}".format(mean_iou),
                             m_recall="{:.4f}".format(mean_recall),
                             m_precision="{:.4f}".format(mean_precision))

        # evaluate
        model.eval()
        total_cm = np.zeros((args.num_classes,)*2)
        test_loss = []
        write_image = True
        with torch.no_grad():
            for images_l, images_r, labels in val_loader:
                # Set variables for training
                images_l = images_l.float().to(dev)
                images_r = images_r.float().to(dev)
                labels = labels.long().to(dev)

                # Get predictions and calculate loss
                cd_preds = model(images_l, images_r)
                cd_loss = criterion(cd_preds, labels)

                cd_preds = torch.argmax(cd_preds, 1)
                labels = labels.squeeze(1)

                images_l = images_l.cpu().numpy()
                images_r = images_r.cpu().numpy()
                labels = labels.cpu().numpy()
                cd_preds = cd_preds.cpu().numpy()
                cd_loss = cd_loss.data.cpu().numpy()

                # Calculate confusion matrix get recall, precision and mean iou
                cm = generate_matrix(labels, cd_preds,
                                     num_classes=args.num_classes)

                total_cm += cm
                test_loss.append(cd_loss)
                if write_image:
                    write_image = False
                    masks = np.where(labels == 255, True, False) | np.where(labels == 0, True, False)
                    labels = np.where(masks, 0, 1)
                    for i in range(len(labels)):
                        image_l = images_l[i].transpose(1, 2, 0) * 255
                        image_r = images_r[i].transpose(1, 2, 0) * 255
                        label = labels[i].astype(np.uint8)
                        cd_pred = cd_preds[i].astype(np.uint8)

                        label = label_to_color_image(label)
                        cd_pred = label_to_color_image(cd_pred)

                        image = np.vstack((np.hstack((image_l, image_r)),
                                           np.hstack((label, cd_pred)))).astype(np.uint8)

                        writer[0].add_image('val_images_{}'.format(i), image, global_step=epoch, dataformats="HWC")

                # Clear batch variables from memory
                del images_l, images_r, labels

            # Recorde model metrics
            mean_iou, ious = get_mean_iou(total_cm)
            mean_recall, recalls = get_recall(total_cm)
            mean_precision, precisions = get_precision(total_cm)

            # Write validation information to tensorboard
            writer[0].add_scalar("val_loss", np.mean(test_loss), epoch)
            writer[0].add_scalar("val_mean_iou", mean_iou, epoch)
            writer[0].add_scalar("val_mean_recall", mean_recall, epoch)
            writer[0].add_scalar("val_mean_precision", mean_precision, epoch)
            for i in range(args.num_classes):
                writer[i+1].add_scalar("val_iou", ious[i], epoch)
                writer[i+1].add_scalar("val_recall", recalls[i], epoch)
                writer[i+1].add_scalar("val_precisions", precisions[i], epoch)

            # Output loss and metrics
            print("=> loss: {:.4f}   val_loss: {:.4f}   mean_iou: {:.4f}   mean_recall: {:.4f}   mean_precision: {:.4f}".
                  format(np.mean(train_loss), np.mean(test_loss), mean_iou, mean_recall, mean_precision))

            # Save best model
            if (mean_iou > best_iou) or (mean_recall > best_recall) or (mean_precision > best_precision):
                best_iou = mean_iou
                best_recall = mean_recall
                best_precision = mean_precision

                torch.save(model,
                           '{}/epoch={}_miou={:.4f}_recall={:.4f}_precision={:.4f}.pt'.
                           format(args.weights_dir, epoch, mean_iou, mean_recall, mean_precision))

    # Close tensorboard
    for w in writer:
        w.close()
