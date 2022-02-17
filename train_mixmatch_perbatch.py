import argparse
import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torch_ema import ExponentialMovingAverage
from randaugment import RandAugment
from models.wideresnet import WideResNet
from utils.misc import load_full_checkpoint, SharpenSoftmax


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="stl10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_arch", default="wideresnet")
    parser.add_argument("--model_name", default="wideresnet-28-2")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--results_dir", type=str, default="/data/weights/hayoung/mixmatch/t1")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--datadir", type=str, default="/data/data/stl10")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--tensorboard_path", type=str, default="./runs/mixmatch/t1")
    parser.add_argument("--tsa", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--weight", type=str, default="/data/weights/hayoung/mixmatch/t1/model_last.pth")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--randaug_u", help="use additional randaug for unlabeled dataset", action="store_true")
    parser.add_argument("--k", type=int, help="number of augmentation for unlabeled data", default=2)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--unsup_weight", type=float, default=100.0)
    args = parser.parse_args()

    return args


def get_loaders(args, resolution, train_transform=None, test_transform=None):
    # transforms
    train_transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # train_transform_u = transforms.Compose([
    #     transforms.PILToTensor(),
    # ])
    # dataset
    if args.dataset_name == "stl10":
        valset = datasets.STL10(args.datadir, "test", transform=test_transform, download=True)
        trainset = datasets.STL10(args.datadir, "train+unlabeled", transform=train_transform, download=True)
        # unlabelset = datasets.STL10(args.datadir, "unlabeled", transform=train_transform_u, download=True)
    else:
        raise Exception("Not supported dataset")
    # loader
    trainloader = DataLoader(trainset, args.batch_size, True, num_workers=0, pin_memory=True)
    valloader = DataLoader(valset, args.batch_size, False, num_workers=0, pin_memory=True)
    # trainloader_u = DataLoader(unlabelset, args.batch_size, False, num_workers=0, pin_memory=True)

    return trainloader, valloader


def get_model(args, device):
    if args.model_arch == "efficientnet":
        model = EfficientNet.from_name(args.model_name)
        model._fc = nn.Linear(model._fc.in_features, args.num_classes)
    elif args.model_arch == "wideresnet":
        depth = int(args.model_name.split("-")[1])
        w_factor = int(args.model_name.split("-")[2])
        model = WideResNet(depth, args.num_classes, w_factor)
    else:
        raise Exception("Not supported network architecture")

    # if args.ema:
    #     ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    #     ema = ema.to(device)
    # else:
    #     ema = None

    model.to(device)
    return model

def main(args):
    writer = SummaryWriter(args.tensorboard_path)
    if not os.path.exists(args.tensorboard_path):
        os.mkdir(args.tensorboard_path)
    
    resolution = (args.resolution, args.resolution)
    train_transform = transforms.Compose([
        transforms.Pad(12),
        transforms.RandomCrop(96),
        transforms.Resize(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    ])
    if args.randaug_u:
        train_transform.transforms.append(RandAugment(1, 2))
    trainloader, valloader = get_loaders(args, resolution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)
    ema = None  # temporarily None
    
    supcriterion = nn.BCELoss(reduction='mean')
    unsupcriterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    if args.resume:
        model, optimizer, scheduler, last_epoch, best_val_loss, best_val_acc = \
            load_full_checkpoint(model, optimizer, scheduler, args.weight)
        print("Loaded checkpoint from: {}".format(args.weight))
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
        best_val_acc = 0.
    
    for ep in range(start_epoch, args.num_epochs):
        scheduler.step()
        print("Epoch {} --------------------------------------------".format(ep + 1))
        train_loss, train_acc, model, optimizer = \
            train(ep, model, trainloader, train_transform,
                  supcriterion, unsupcriterion, optimizer, writer, args.num_epochs, ema)
        print("Train loss: {}\tTrain acc: {}".format(train_loss, train_acc))
        val_loss, val_acc = eval_model(ep, model, valloader, writer, args.num_epochs, ema)
        print("Val loss: {}\tVal acc: {}".format(val_loss, val_acc))
        print("--------------------------------------------")
        # scheduler.step(val_loss)
        print("{}".format(optimizer.state_dict))
        if ep == 0:
            best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, True)
        save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, False)
    print("Best Val Loss: {} / Acc: {}".format(best_val_loss, best_val_acc))


def train(ep, model, loader, _transforms, sup_criterion, unsup_criterion, optimizer, writer, eps, ema):
    model.train()
    train_loss = 0.
    train_acc = 0.
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    softmax = nn.Softmax(dim=1)
    sharpen_softmax = SharpenSoftmax(0.5, dim=1)
    beta = torch.distributions.beta.Beta(args.alpha, args.alpha)

    for i, (inputs_w, labels_w) in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        inputs_w, labels_w = inputs_w.cuda(), labels_w.cuda()

        # generate X', U'
        with torch.no_grad():
            # mask for acquiring unlabeled data
            mask_w = labels_w == -1
            # calculate W
            inputs_w_l = normalize(_transforms(inputs_w[~mask_w]) / 255.)  # for labeled data
            # !! for convinence, shuffle k augmented unlabeled data within a mini-batch, not for entire dataset !!
            # !! Therefore, args.batch_size doesn't work as an actual batch size.
            # !! Real batch size depends on args.batch_size, args.k, and the number of chosen labeled data
            inputs_w_uk = []
            pred_k = torch.zeros((mask_w.sum().item(), args.num_classes)).cuda()
            for _ in range(args.k):
                inputs_w_u = normalize(_transforms(inputs_w[mask_w]) / 255.)
                inputs_w_uk.append(inputs_w_u)
                pred_k = pred_k + model(inputs_w_u)
            pred_k = pred_k / args.k
            pred_k = sharpen_softmax(pred_k)
            # mixup prime and W
            inputs = torch.cat([inputs_w_l] + inputs_w_uk, dim=0)
            labels = [nn.functional.one_hot(l, args.num_classes) for l in labels_w[~mask_w]]
            labels = torch.stack(labels, dim=0) if len(labels) > 0 else torch.Tensor([]).view(0, args.num_classes).cuda()
            labels = torch.cat([labels] + args.k * [pred_k], dim=0)
            lam = beta.sample(sample_shape=torch.Size([inputs.size()[0]])).cuda()
            lam_p = torch.max(lam, 1-lam)
            w_index = torch.randperm(inputs.size()[0])
            mixed_inputs = lam_p[:, None, None, None] * inputs + (1 - lam_p[:, None, None, None]) * inputs[w_index]
            mixed_labels = lam_p[:, None] * labels + (1 - lam_p[:, None]) * labels[w_index]
            # shuffle final input
            _index = torch.randperm(mixed_inputs.size()[0])
            mixed_inputs = mixed_inputs[_index]
            mixed_labels = mixed_labels[_index]
            # keep index of X' for computing different loss function
            labeled_mask = _index < labels_w[~mask_w].size()[0]

        # forward
        outputs = model(mixed_inputs).softmax(dim=1)
        if labeled_mask.sum() > 0:
            sup_loss = sup_criterion(outputs[labeled_mask], mixed_labels[labeled_mask])
        else:
            sup_loss = 0
        unsup_loss = unsup_criterion(outputs[~labeled_mask], mixed_labels[~labeled_mask])
        total_loss = sup_loss + args.unsup_weight * unsup_loss

        # backward
        # if args.tsa:
        #     tsa_mask = get_tsa_mask(sup_outputs, eps, ep, len(unsuploader), i)
        #     sup_loss = (sup_loss * tsa_mask.max(1)[0]).sum()
        # else:
        #     sup_loss = sup_loss.mean()
        train_loss += total_loss.item()
        total_loss.backward() 
        optimizer.step()
        if ema is not None:
            ema.update()
        if labeled_mask.sum() > 0:
            running_acc = (outputs[labeled_mask].argmax(1) == mixed_labels[labeled_mask].argmax(1)).sum().item()
        else:
            running_acc = 0
        train_acc += running_acc
        writer.add_scalar("train acc", running_acc, ep * len(loader) + i)
    train_loss /= len(loader)
    train_acc /= len(loader.dataset)
    return train_loss, train_acc, model, optimizer


def eval_model(ep, model, loader, writer, eps, ema):
    model.eval()
    val_loss = 0
    val_acc = 0
    criterion = nn.CrossEntropyLoss()
    if ema is not None:
        with ema.average_parameters():
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(tqdm(loader)):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    running_acc = (outputs.argmax(1) == labels).sum().item()
                    val_acc += running_acc
                    loss = criterion(outputs, labels).sum()
                    val_loss += loss.item()
                    writer.add_scalar("val loss", loss.item(), ep * len(loader) + i)
                    writer.add_scalar("val acc", running_acc, ep * len(loader) + i)
    else:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(loader)):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                running_acc = (outputs.argmax(1) == labels).sum().item()
                val_acc += running_acc
                loss = criterion(outputs, labels).sum()
                val_loss += loss.item()
                writer.add_scalar("val loss", loss.item(), ep * len(loader) + i)
                writer.add_scalar("val acc", running_acc, ep * len(loader) + i)
    val_loss /= len(loader)
    val_acc /= len(loader.dataset)
    return val_loss, val_acc


def save_checkpoint(ep, model, optimizer, scheduler, savepath, best_val_loss, best_val_acc, isbest):
    save_dict = {
        "epoch": ep,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc
    }
    if isbest:
        torch.save(save_dict, os.path.join(savepath, "model_best.pth"))
    else:
        torch.save(save_dict, os.path.join(savepath, "model_last.pth"))


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    main(args)