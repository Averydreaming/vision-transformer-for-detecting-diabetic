import os
import math
import argparse
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from my_dataset import APTOSDataset 
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:#判断括号里的文件是否存在的意思
        os.makedirs("./weights")#不存在就创建一个

    tb_writer = SummaryWriter()

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    
    train_csv_root = './train.csv'
    test_csv_root = './test.csv'
    train_img_root = './train_images'
    test_img_root = './test_images'
    
    # 划分训练集和验证集
    train_labels_frame = pd.read_csv(train_csv_root)
    train_df, val_df = train_test_split(train_labels_frame, test_size=0.1)

    #数据增强
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),#随即裁剪，然后缩放
                                     transforms.RandomHorizontalFlip(),#以给定的概率随机水平旋转给定的PIL的图像
                                     transforms.ToTensor(),#将给定图像转为Tensor
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),#归一化处理#串联多个transform操作
        "val": transforms.Compose([transforms.Resize(256),#调整尺寸
                                   transforms.CenterCrop(224),#从图像中心裁剪图片
                                   transforms.ToTensor(),#将给定图像转为Tensor
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    #train_data_set = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])

    # 实例化验证数据集
    #val_data_set = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])
    
    # 数据集和数据加载器
    train_dataset = APTOSDataset(dataframe=train_df, root_dir=train_img_root, transform=data_transform["train"])
    val_dataset = APTOSDataset(dataframe=val_df, root_dir=train_img_root, transform=data_transform["val"])
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=5, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))#model.load_state_dict(weights_dict, strict=False)加载的模型

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(false)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)#随机梯度下降算法
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine 余弦退火降低学习率
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)#调整学习率
    # 训练模型
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


    # 绘制损失曲线和准确率曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Training loss')
    ax1.plot(val_losses, label='Validation loss')
    ax1.legend(frameon=False)
    ax1.set_title("Loss curves")

    ax2.plot(train_accuracies, label='Training accuracy')
    ax2.plot(val_accuracies, label='Validation accuracy')
    ax2.legend(frameon=False)
    ax2.set_title("Accuracy curves")

    # 保存图像到文件
    plt.savefig('loss_and_accuracy_curves.png', dpi=300)

    # 显示图像
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    #parser.add_argument('--data-path', type=str, default="./data/flower_photos")
    #parser.add_argument('--model-name', default='', help='create model name')
    # 预训练权重路径，如果不想载入就设置为空字符
    #parser.add_argument('--weights', type=str, default='./weights/jx_vit_base_patch16_224_in21k.pth',help='initial weights path')
    parser.add_argument('--weights', type=str, default='',help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

