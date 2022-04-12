import argparse
import os
import torch
import math
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
from utils.misc import load_full_checkpoint, SharpenSoftmax, LogWeight


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="stl10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_arch", default="wideresnet")
    parser.add_argument("--model_name", default="wideresnet-28-2")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--results_dir", type=str, default="/data/weights/hayoung/mixmatch/t4")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--datadir", type=str, default="/data/data/stl10")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--tensorboard_path", type=str, default="./runs/mixmatch/t4")
    parser.add_argument("--tsa", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--weight", type=str, default="/data/weights/hayoung/mixmatch/t3/model_last.pth")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--k", type=int, help="number of augmentation for unlabeled data", default=2)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--unsup_weight", type=float, default=50.0)
    parser.add_argument("--use_disk", action="store_true")
    parser.add_argument("--temp_disk", type=str, default="./temp")
    parser.add_argument("--cosine_tmax", type=int, default=300)
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
        ])
    # dataset
    if args.dataset_name == "stl10":
        valset = datasets.STL10(args.datadir, "test", transform=test_transform, download=True)
        trainset = datasets.STL10(args.datadir, "train", transform=train_transform, download=True)
        unlabelset = datasets.STL10(args.datadir, "unlabeled", transform=train_transform, download=True)
    else:
        raise Exception("Not supported dataset")

    # loader
    trainloader = DataLoader(trainset, 1, True, num_workers=0, pin_memory=True)
    valloader = DataLoader(valset, args.batch_size, False, num_workers=0, pin_memory=True)
    trainloader_u = DataLoader(unlabelset, args.batch_size, True, num_workers=0, pin_memory=True)

    return trainloader, valloader, trainloader_u


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
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    ])
    trainloader, valloader, trainloader_u = get_loaders(args, resolution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)
    ema = None  # temporarily None
    
    supcriterion = nn.BCELoss(reduction='mean')
    unsupcriterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_tmax)

    if args.resume:
        model, optimizer, scheduler, last_epoch, best_val_loss, best_val_acc = \
            load_full_checkpoint(model, optimizer, scheduler, args.weight)
        print("Loaded checkpoint from: {}".format(args.weight))
        optimizer.state_dict()['param_groups'][0]['lr'] = args.lr
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
        best_val_acc = 0.
    
    for ep in range(start_epoch, args.num_epochs):
        print("Epoch {} --------------------------------------------".format(ep + 1))
        train_loss, train_acc, model, optimizer = \
            train(args, ep, model, trainloader, trainloader_u, train_transform,
                  supcriterion, unsupcriterion, optimizer, writer, args.num_epochs, ema)
        print("Train loss: {}\tTrain acc: {}".format(train_loss, train_acc))
        scheduler.step()
        val_loss, val_acc = eval_model(ep, model, valloader, writer, args.num_epochs, ema)
        print("Val loss: {}\tVal acc: {}".format(val_loss, val_acc))
        print("--------------------------------------------")
        # scheduler.step(val_loss)
        print("{}".format(optimizer.state_dict()['param_groups'][0]["lr"]))
        if ep == 0:
            best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, True)
        save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, False)
    print("Best Val Loss: {} / Acc: {}".format(best_val_loss, best_val_acc))


def train(args, ep, model, loader, loader_u, _transforms, sup_criterion, unsup_criterion, optimizer, writer, eps, ema):
    model.train()
    train_loss = 0.
    train_acc = 0.
    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    softmax = nn.Softmax(dim=1)
    sharpen_softmax = SharpenSoftmax(0.5, dim=1)
    beta = torch.distributions.beta.Beta(args.alpha, args.alpha)

    # label guess
    # To store all transformed inputs of STL-10 with args.k==2, additional ~30GB of memory space is required.
    print("Processing label guess...")
    if args.use_disk:
        print("Using disk intead of memory")
        os.chdir("../MixMatch")
        iu = 0
        inputs_unlabeled = []
        guessed_labels = []
        with torch.no_grad():
            for (samples, _) in tqdm(loader_u):
                samples = samples.cuda()
                pred_k = torch.zeros((samples.size()[0], args.num_classes)).cuda()
                for _ in range(args.k):
                    samples_k = _transforms(samples) / 255.
                    pred_k = pred_k + sharpen_softmax(model(samples_k))
                    for s in samples_k.cpu():
                        inputs_unlabeled.append(iu)
                        np.save(os.path.join(args.temp_disk, str(iu) + ".npy"), s.numpy())
                        iu += 1
                pred_k = pred_k / args.k
                guessed_labels = guessed_labels + [p for p in pred_k] * args.k
    else:
        inputs_unlabeled = []
        guessed_labels = []
        with torch.no_grad():
            for (samples, _) in tqdm(loader_u):
                samples = samples.cuda()
                pred_k = torch.zeros((samples.size()[0], args.num_classes)).cuda()
                for _ in range(args.k):
                    samples_k = _transforms(samples) / 255.
                    pred_k = pred_k + sharpen_softmax(model(samples_k))
                    for s in samples_k.cpu():
                        inputs_unlabeled.append(s)
                pred_k = pred_k / args.k
                guessed_labels = guessed_labels + [p for p in pred_k] * args.k
    print("Label guess done. {}/{}".format(len(inputs_unlabeled), len(guessed_labels)))

    # Unsup weight coeff
    unsup_weight_coeff = LogWeight(10, args.num_epochs)
    # compute input batch
    index_a = list(range(len(inputs_unlabeled)))
    index_w = list(range(len(inputs_unlabeled) + len(loader)))
    np.random.shuffle(index_a)
    np.random.shuffle(index_w)
    def worldset_generator(wset):
        yield from wset
    labeled_generator = iter(loader)
    batch_ind_size = math.ceil(len(index_a) / (args.batch_size * args.k))
    for i in tqdm(range(batch_ind_size)):
        optimizer.zero_grad()
        # construct set A for Mixup
        labeled_inputs = []
        labeled_labels = []
        u_batch_ind_a = index_a[i*args.batch_size*args.k : (i+1)*args.batch_size*args.k]  # data index for unlabeled set, size: (args.batch_size * args.k) except the last chunk
        for _ in range(args.batch_size):  # get labeled set, size: args.batch_size
            try:
                _inputs, _labels = labeled_generator.next()
                labeled_inputs.append((_transforms(_inputs) / 255.).cuda())
                labeled_labels.append(nn.functional.one_hot(_labels.cuda(), args.num_classes))
            except StopIteration:
                labeled_generator = iter(loader)
                _inputs, _labels = labeled_generator.next()
                labeled_inputs.append((_transforms(_inputs) / 255.).cuda())
                labeled_labels.append(nn.functional.one_hot(_labels.cuda(), args.num_classes))
        if args.use_disk:
            unlabeled_inputs = []
            for b in u_batch_ind_a:
                u_inputs = torch.from_numpy(np.load(os.path.join(args.temp_disk, str(b) + ".npy"))).unsqueeze(0).cuda()
                unlabeled_inputs.append(u_inputs)
        else:
            unlabeled_inputs = [inputs_unlabeled[b].unsqueeze(0).cuda() for b in u_batch_ind_a]
        unlabeled_labels = [guessed_labels[b].unsqueeze(0).cuda() for b in u_batch_ind_a]
        batch_ind_a = list(range(len(labeled_inputs) + len(unlabeled_inputs)))
        np.random.shuffle(batch_ind_a)
        label_mask = np.asarray(batch_ind_a) < len(labeled_inputs)  # mask for distinguishing labeled and unlabeled sample
        inputs_a = torch.cat(labeled_inputs + unlabeled_inputs, dim=0)[batch_ind_a]
        labels_a = torch.cat(labeled_labels + unlabeled_labels, dim=0)[batch_ind_a]
        
        # construct set B for Mixup
        ind_generator = worldset_generator(index_w)
        inputs_b = []
        labels_b = []
        for _ in range(len(inputs_a)):
            # get index
            try:
                ind_b = next(ind_generator)
            except StopIteration:
                ind_generator = worldset_generator(index_w)
                ind_b = next(ind_generator)

            if ind_b >= len(inputs_unlabeled):  # get sample from labeled data
                _inputs, _labels = loader.dataset.__getitem__(ind_b - len(inputs_unlabeled))
                inputs_b.append(_transforms(_inputs.unsqueeze(0).cuda()) / 255.)
                labels_b.append(nn.functional.one_hot(torch.tensor(_labels), args.num_classes).unsqueeze(0).cuda())
            else:  # get sample from unlabeled data
                if args.use_disk:
                    u_inputs = torch.from_numpy(np.load(os.path.join(args.temp_disk, str(ind_b) + ".npy"))).unsqueeze(0).cuda()
                    inputs_b.append(u_inputs)
                else:
                    inputs_b.append(inputs_unlabeled[ind_b].unsqueeze(0).cuda())
                labels_b.append(guessed_labels[ind_b].unsqueeze(0).cuda())
        inputs_b = torch.cat(inputs_b, dim=0)
        labels_b = torch.cat(labels_b, dim=0)

        # Mixup
        lam = beta.sample(sample_shape=torch.Size([inputs_a.size()[0]])).cuda()
        lam_p = torch.max(lam, 1-lam)
        inputs = lam_p[:, None, None, None] * inputs_a + (1 - lam_p[:, None, None, None]) * inputs_b
        labels = lam_p[:, None] * labels_a + (1 - lam_p[:, None]) * labels_b
    
        # forward & backward
        outputs = model(inputs).softmax(dim=1)
        sup_loss = sup_criterion(outputs[label_mask], labels[label_mask])
        unsup_loss = unsup_criterion(outputs[~label_mask], labels[~label_mask])
        # total_loss = sup_loss + args.unsup_weight * unsup_loss
        total_loss = sup_loss + (unsup_weight_coeff(ep) * args.unsup_weight) * unsup_loss
        train_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()

        # acc for labeled input
        running_acc = (outputs[label_mask].argmax(1) == labels[label_mask].argmax(1)).sum().item()
        train_acc += running_acc
        writer.add_scalar("train acc", running_acc, ep * len(loader) * i)
    train_loss /= (i + 1)
    train_acc /= ((i + 1) * args.batch_size)
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
