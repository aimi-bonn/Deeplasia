from __future__ import print_function
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader

from lib.datasets.boneage_dataset import BoneAgeDataset
from lib.datasets.boneage_masked_dataset import BoneAgeMaskedDataset
from lib.models.EfficientdBAM import EfficientdBAM
from lib.modules import Swish

saved_model_dir = "saved_models"


# Many of these are deprecated and should be removed or re-implemented
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Bone Age Test')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=((5e-4)), metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--val_percentage', type=float, default=0.1,
                        help='Percentage of dataset to be used as validation split')
    parser.add_argument('--img-scale', type=float, default=1.,
                        help='Fraction the image will be scaled to. (Default=1.0 can lead to OOM Error)')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout to be used in center layers of UNet (Default=0.0 means no dropout)')

    parser.add_argument('--session', type=int, default=1, dest='session',
                        help='Session used to distinguish model tests.')
    parser.add_argument('--model-type', default='UNet', dest='model_type',
                        help='Model type to use. (Options: UNet, OctUNet, DilatedUNet, PixelShuffleUNet)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='If model-type is OctUNet alpha will be assigned to the middle Octave Convolutions')
    parser.add_argument('--ACT-type', default='ReLU', dest='ACT_type',
                        help='activation function to use in UNet. (Options: ReLU, PReLU, Swish)')
    parser.add_argument('--up-type', default='upsample', dest='up_type',
                        help='Upsampling type to use in UpConv part of UNet (Options: upsample, upconv)')
    parser.add_argument('--norm', default='BN', dest='norm',
                        help='Which Normalization to use: None, BN, LN, IN, GN')

    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use tensorboard for logging')

    parser.add_argument('--grad_acc_interval', type=int, default=1, dest='grad_acc_interval',
                        help='Accumulates gradients of n batches. (n >= 1, default=1).')

    parser.add_argument('--val-interval', type=int, default=500,
                        help='how many batches to wait before validation is evaluated (and optimizer is stepped).')

    parser.add_argument('--data_mean', type=float, default=-1,
                        help='how many batches to wait before validation is evaluated (and optimizer is stepped).')
    parser.add_argument('--data_std', type=float, default=-1,
                        help='how many batches to wait before validation is evaluated (and optimizer is stepped).')

    parser.add_argument('--BN_ACT_test', action='store_true', default=False,
                        help='Use BN_ACT instead of ACT_BN')

    parser.add_argument('--predict_gender', action='store_true', default=False,
                        help='Try and predict gender as well as bone age')
    parser.add_argument('--use_gender', action='store_true', default=False,
                        help='Also use gender to predict bone age')
    parser.add_argument('--use_masks', action='store_true', default=False,
                        help='Use hand masks of image to predict bone age')

    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, val_loader=None, scheduler=None, epochs=-1):
    model.train()

    # Time measurements
    tick = datetime.datetime.now()

    # Tensorboard Writer
    if args.use_tensorboard:
        writer = SummaryWriter(
            comment=f"s{args.session}_{args.model_type}_{args.up_type}_{args.ACT_type}_bs{args.batch_size}"
                    f"_scale{int(args.img_scale * 100)}_{args.norm}_dropout{args.dropout}")
    global_step = 0
    boneage_crit = nn.MSELoss().to(device)

    if epochs == -1:
        epochs = args.epochs
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.
        for batch_idx, (data, boneage_t, gender_t) in enumerate(train_loader):

            data = data.to(device, dtype=torch.float32)
            boneage_target = boneage_t.to(device, dtype=torch.float32).unsqueeze(1)
            gender_target = gender_t.to(device, dtype=torch.float32).unsqueeze(1)

            if args.use_gender:
                data = (data, gender_target)

            if args.predict_gender:
                boneage, gender = model(data)
                gender_loss = F.binary_cross_entropy_with_logits(gender, gender_target)
            else:
                boneage = model(data)

            boneage_loss = boneage_crit(boneage, boneage_target)
            loss = boneage_loss
            if args.predict_gender:
                loss += gender_loss

            loss = loss / args.grad_acc_interval
            loss.backward()

            # Gradient accumulation, part 2:
            if (batch_idx+1) % args.grad_acc_interval == 0:
                # Clipping gradients here, use if we get exploding gradients
                nn.utils.clip_grad_value_(model.parameters(), 0.1)

                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            if (batch_idx+1) % args.log_interval == 0:
                tock = datetime.datetime.now()
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(Elapsed time {:.1f}s)'.format(
                    tock.strftime("%H:%M:%S"), epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                                                      100. * batch_idx / len(train_loader), loss.item(),
                    (tock - tick).total_seconds()))
                tick = tock

                if args.use_tensorboard:
                    writer.add_scalar('Train/overall_loss', loss.item(), global_step)
                    writer.add_scalar('Train/boneage_loss', boneage_loss.item(), global_step)
                    if args.predict_gender:
                        writer.add_scalar('Train/gender_loss', gender_loss.item(), global_step)

                if args.dry_run:
                    break

            if val_loader:
                if (batch_idx+1) % args.val_interval == 0:
                    avg_val_loss, avg_val_boneage, avg_val_mae, avg_val_gender = validate(model, device, val_loader, args)

                    tick = datetime.datetime.now()

                    if scheduler:
                        scheduler.step(avg_val_mae)

                    if args.use_tensorboard:
                        writer.add_scalar('Val/overall_loss', avg_val_loss, global_step)
                        writer.add_scalar('Val/boneage_loss', avg_val_boneage, global_step)
                        if args.predict_gender:
                            writer.add_scalar('Val/gender_loss', avg_val_gender, global_step)
                        writer.add_scalar('Val/MAE', (avg_val_mae * args.data_std), global_step)

            global_step += 1

        ## Epoch is completed
        print(f"Overall average training loss: {epoch_loss / len(train_loader):.6f}")

        # Save model
        if args.save_model:
            print(
                f"Saving model in: {saved_model_dir}/s{args.session}_boneage_adam_augment_{args.model_type}_e{epoch}_{args.ACT_type}"
                f"_{args.up_type}_{args.norm}_bs{args.batch_size}_scale{int(args.img_scale * 100)}_dropout{args.dropout}.pt")
            torch.save(model.state_dict(),
                       f"{saved_model_dir}/s{args.session}_boneage_adam_augment_{args.model_type}_e{epoch}_{args.ACT_type}"
                       f"_{args.up_type}_{args.norm}_bs{args.batch_size}_scale{int(args.img_scale * 100)}_dropout{args.dropout}.pt")

        # Plot the performance on the validation set
        #validate(model, device, val_loader, args, plot=True)

    if args.use_tensorboard:
        writer.flush()
        writer.close()


def validate(model, device, val_loader, args, plot=False):
    model.eval()
    mae_loss = nn.L1Loss().to(device)
    criterion = nn.MSELoss().to(device)
    val_gender_loss = 0.
    val_boneage_loss = 0.
    val_mae_loss = 0.

    boneage_preds = []
    boneage_targets = []
    gender_preds = []
    gender_targets = []

    tick = datetime.datetime.now()
    with torch.no_grad():
        for (data, boneage_t, gender_t) in val_loader:
            data = data.to(device, dtype=torch.float32)
            boneage_target = boneage_t.to(device, dtype=torch.float32).unsqueeze(1)
            gender_target = gender_t.to(device, dtype=torch.float32).unsqueeze(1)

            boneage_targets.append(boneage_target.squeeze())

            if args.use_gender:
                data = (data, gender_target)

            if args.predict_gender:
                boneage, gender = model(data)
                val_gender_loss += F.binary_cross_entropy_with_logits(gender, gender_target)

                boneage_preds.append(boneage.squeeze())
                gender_preds.append(gender.squeeze())
                gender_targets.append(gender_target.squeeze())
            else:
                boneage = model(data)
                boneage_preds.append(boneage.squeeze())

            val_mae_loss += mae_loss(boneage, boneage_target).item()
            val_boneage_loss += criterion(boneage, boneage_target).item()

    val_loss = val_boneage_loss + val_gender_loss
    model.train()

    # plots the predicted age vs. actual age + MAD
    if plot:
        boneage_preds = torch.stack(boneage_preds).cpu().view(-1)
        boneage_targets = torch.stack(boneage_targets).cpu().view(-1)

        if args.data_mean != -1 and args.data_std != -1:
            boneage_preds = boneage_preds * args.data_std + args.data_mean
            boneage_targets = boneage_targets * args.data_std + args.data_mean

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(boneage_targets, boneage_preds, 'r.', label='predictions')
        ax.plot(boneage_targets, boneage_targets, 'b-', label='actual')
        ax.legend(loc='upper right')
        ax.set_xlabel('Actual Age (Months)')
        ax.set_ylabel('Predicted Age (Months)')
        ax.text(25, 200, f'MAE = {val_mae_loss / len(val_loader) * args.data_std:0.2f}')
        plt.show()
        #plt.waitforbuttonpress()

    print(f"Average Overall Loss ({val_loss / len(val_loader)}), BoneAgeLoss ({val_boneage_loss / len(val_loader)}), GenderLoss ({val_gender_loss / len(val_loader)}) during validation")
    print(f"Elapsed time during validation: {(datetime.datetime.now() - tick).total_seconds():.1f}s")
    print(f'\tMAE = {val_mae_loss / len(val_loader) * args.data_std:0.2f}')

    return val_loss / len(val_loader), val_boneage_loss / len(val_loader), val_mae_loss / len(val_loader), val_gender_loss / len(val_loader)


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 0,
                       'pin_memory': True})

    dataset_train = BoneAgeDataset() if not args.use_masks else BoneAgeMaskedDataset()
    val_ids = dataset_train.split_and_get_val(ratio=args.val_percentage)
    dataset_val = BoneAgeDataset(validation=val_ids) if not args.use_masks else BoneAgeMaskedDataset(validation=val_ids)
    data_mean, data_std = dataset_train.get_mean_std()
    args.data_mean = data_mean
    args.data_std = data_std

    train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, **kwargs, shuffle=False, drop_last=True)

    if args.norm == 'BN':
        norm_type = nn.BatchNorm2d
    elif args.norm == 'GN':
        norm_type = nn.GroupNorm
    elif args.norm == 'None':
        norm_type = None
    else:
        print(f"Unknown Normalization type given: {args.norm}")
        exit()
    print(f"Using Normalization type: {norm_type}")

    if args.ACT_type == "ReLU":
        ACT_type = nn.ReLU
    elif args.ACT_type == "PReLU":
        ACT_type = nn.PReLU
    elif args.ACT_type == "Swish":
        ACT_type = Swish.Swish
    else:
        print(f"Invalid ACT_type given! (Got {args.ACT_type})")
        ACT_type = nn.ReLU

    if args.model_type == 'EfficientdBAM':
        model = EfficientdBAM(in_channels=1, num_classes=1, pretrained=True).to(device)
        model_name = "efficientdBAM"
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")

    ## TEST NUMBER OF IMAGES BEFORE VALIDATION:
    args.log_interval = args.val_interval // args.batch_size // 5 - 1
    args.val_interval = (args.val_interval // args.batch_size)*2

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_sched.ReduceLROnPlateau(optimizer, factor=0.75, patience=10, verbose=True, min_lr=5e-6, threshold=1e-3)

    ## To continue training or when only interested in validation, load weights --> use correct model though
    #model.load_state_dict(
    #    torch.load(f"s15_boneage_adam_augment_EfficientdBAM_e25_PReLU_upconv_GN_bs4_scale100_dropout0.0.pt",
    #               map_location=device))

    train(args, model, device, train_loader, optimizer, val_loader=val_loader, scheduler=scheduler)
    # validate(model, device, val_loader, args, plot=True)

if __name__ == '__main__':
    main()
