import argparse
import os
import random
from collections import Counter
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch import autograd
from model import MobileNetV1
import torch.nn.functional as F
from dataloader import TrainData,TestData



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="CUMT", help="name of the dataset")
    parser.add_argument("--model_name", type=str, default="mobilenetV2", help="name of the model")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--classes", type=int, default=293, help="numbers of the class")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
    parser.add_argument("--lambda_id", type=float, default=0.001, help="identity loss weight")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    net =MobileNetV1(num_classes=opt.classes)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=1e-5)


    dataset = TrainData("D:/data/{}-Multimodal/print-train".format(opt.dataset_name))
    dataset_test = TestData("D:/data/{}-Multimodal/print-test".format(opt.dataset_name))
    CELoss = nn.CrossEntropyLoss()  # 实例化损失函数

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    train_steps = len(data_loader) * opt.batch_size

    best_acc = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        net.train()
        acc = 0.0
        running_loss = 0.0
        train_bar = tqdm(data_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            p_img, person_name = data
            p_img = p_img.to(device)

            person_labels = [int(_) - 1 for _ in person_name]
            person_labels = torch.tensor(person_labels).to(device)

            # 前向传播
            output = net(p_img)
            predict = torch.max(output, dim=1)[1]
            acc += torch.eq(predict, person_labels.to(device)).sum().item()

            # 计算总损失
            loss = CELoss(output, person_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, opt.n_epochs,
                                                                     running_loss / (step + 1))
            accurate = acc / train_steps
        print('num:{},train_accuracy:{:.4f},acc:{}'.format(train_steps, accurate, acc))

        # 仅在训练到60轮之后进行验证
        if (epoch + 1) >= 0:
            net.eval()
            acc = 0.0
            test_bar = tqdm(test_loader, file=sys.stdout)
            with torch.no_grad():
                for step, data in enumerate(test_bar):
                    p_img,  person_name = data
                    p_img = p_img.to(device)

                    output = net(p_img )

                    predict = torch.max(output, dim=1)[1]
                    label = [int(_) - 1 for _ in person_name]
                    label = torch.tensor(label).to(device)
                    acc += torch.eq(predict, label.to(device)).sum().item()
                    test_steps = len(test_loader) * opt.batch_size
                    accurate = acc / test_steps

            print("num:{}, test_accuracy:{:.4f},acc:{}".format(test_steps, accurate, acc))

        if best_acc < accurate:
            best_acc = accurate
            best_batch = epoch + 1


            save_path = 'model_best_mobilenetV1.pth'
            torch.save(net.state_dict(), save_path)

        print("best_acc = ", best_acc)
        print("best_batch=", best_batch)


if __name__ == '__main__':
    main()