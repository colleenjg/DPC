import os
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm
import copy
#from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../utils')
sys.path.append('../')
from dataset_3d import *
from model_3d import *
from resnet_2d3d import neq_load_customized
from augmentation import *
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

import yaml

from stimuli import GaborSequenceGenerator

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=1, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--surprise_epoch', default=10, type=int)
parser.add_argument('--blank', default=False, type=bool)
parser.add_argument('--roll', default='False', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--p_E', default=0.1, type=float)
parser.add_argument('--e_pos', default='E', type=str)

def main():
    global args; args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    ### dpc model ###
    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len, 

                        network=args.net, 
                        pred_step=args.pred_step)
    else: raise ValueError('wrong model!')

    #model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion; criterion = nn.CrossEntropyLoss()

    ### optimizer ###
    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: pass # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    #old_backbone_weights = {k: v.clone() for k, v in model.module.backbone.named_parameters()}
    #old_agg_weights = {k: v.clone() for k, v in model.module.agg.named_parameters()}
    #old_network_pred_weights = {k: v.clone() for k, v in model.module.network_pred.named_parameters()}

    
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    best_acc = 0
    global iteration; iteration = 0

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr: # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else: print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']),flush=True)
        else: 
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    if args.dataset == 'ucf101': # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim,args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'k400': # designed for kinetics400, short size=150, rand crop to 128x128
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    if args.dataset == 'gabors':
        train_loader = GaborSequenceGenerator(batch_size=args.batch_size, blank=args.blank, roll=args.roll, p_E=args.p_E, e_pos=args.e_pos, num_seq=args.num_seq, num_trials=20, WIDTH=128, HEIGHT=128)
        val_loader = GaborSequenceGenerator(batch_size=1, num_seq=args.num_seq, num_trials=20, WIDTH=128, HEIGHT=128)
    else:
        train_loader = get_data(transform, 'train')
        val_loader = get_data(transform, 'val')

    # setup tools
    global de_normalize; de_normalize = denorm()
    global img_path; img_path, model_path = set_path(args)
#    global writer_train
#    try: # old version
#        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
#        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
#    except: # v1.7
#        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
#        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))
    
    loss_dict = {'Training' : {},
                 'Validation' : {}}
    
    detailed_loss_dict = {}
    loss_foreach_bigdict = {}
    dot_foreach_dict = {}
    target_foreach_dict = {}
    
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > args.surprise_epoch:
            train_loader.mode = 'surp'
            print('mode: '+train_loader.mode)
    
        train_loss, train_acc, train_accuracy_list, detailed_loss, loss_foreach_dict, dot_foreach, target_foreach = train(train_loader, model, optimizer, epoch)
        #val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch)
        val_acc = 0
        loss_dict['Training'][epoch] = train_loss
        #loss_dict['Validation'][epoch] = val_loss
        loss_foreach_bigdict[epoch] = loss_foreach_dict
        dot_foreach_dict[epoch] = dot_foreach        
        target_foreach_dict[epoch] = target_foreach
        detailed_loss_dict[epoch] = detailed_loss 
        
        # Save to yaml
        #print(os.getenv('SLURM_TMPDIR') + '/loss.yaml',flush=True)
        yaml.dump(detailed_loss_dict, open(os.getenv('SLURM_TMPDIR') + '/loss_%d_%d.yaml'%(args.surprise_epoch,args.seed), 'w'))
        yaml.dump(train_loader.prev_seq, open(os.getenv('SLURM_TMPDIR') + '/seq_%d_%d.yaml'%(args.surprise_epoch,args.seed), 'w'))
        yaml.dump(loss_foreach_bigdict, open(os.getenv('SLURM_TMPDIR') + '/loss_foreach_%d_%d.yaml'%(args.surprise_epoch,args.seed), 'w'))
        yaml.dump(dot_foreach_dict, open(os.getenv('SLURM_TMPDIR') + '/dot_foreach_%d_%d.yaml'%(args.surprise_epoch,args.seed), 'w'))
        yaml.dump(target_foreach_dict, open(os.getenv('SLURM_TMPDIR') + '/target_foreach_%d_%d.yaml'%(args.surprise_epoch,args.seed), 'w'))

        #print('train_loss '+str(train_loss),flush=True)
        #print('val_loss: '+str(val_loss),flush=True)
        #print(train_loader.prev_seq[-1],flush=True)


        
        # save curve
#        writer_train.add_scalar('global/loss', train_loss, epoch)
#        writer_train.add_scalar('global/accuracy', train_acc, epoch)
#        writer_val.add_scalar('global/loss', val_loss, epoch)
#        writer_val.add_scalar('global/accuracy', val_acc, epoch)
#        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
#        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
#        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
#        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
#        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
#        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
        is_best = val_acc > best_acc; best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration}, 
                         is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=False)

        #if epoch == 4:
            #new_weights_backbone = {k: v.clone() for k, v in model.module.backbone.named_parameters()}
            #new_weights_agg = {k: v.clone() for k, v in model.module.agg.named_parameters()}
            #new_weights_network_pred = {k: v.clone() for k, v in model.module.network_pred.named_parameters()}
            
            #weight_changes_backbone = {k: new_weights_backbone[k] - old_backbone_weights[k] for k in old_backbone_weights}
            #weight_changes_agg = {k: new_weights_agg[k] - old_agg_weights[k] for k in old_agg_weights}
            #weight_changes_network_pred = {k: new_weights_network_pred[k] - old_network_pred_weights[k] for k in old_network_pred_weights}

            #sum_changes_backbone = {k: torch.sum(new_weights_backbone[k] - old_backbone_weights[k]) for k in old_backbone_weights}
            #sum_changes_agg = {k: torch.sum(new_weights_agg[k] - old_agg_weights[k]) for k in old_agg_weights}
            #sum_changes_network_pred = {k: torch.sum(new_weights_network_pred[k] - old_network_pred_weights[k]) for k in old_network_pred_weights}
            
            #mean_changes_backbone = {k: torch.mean(torch.abs(new_weights_backbone[k] - old_backbone_weights[k])) for k in old_backbone_weights}
            #mean_changes_agg = {k: torch.mean(torch.abs(new_weights_agg[k] - old_agg_weights[k])) for k in old_agg_weights}
            #mean_changes_network_pred = {k: torch.mean(torch.abs(new_weights_network_pred[k] - old_network_pred_weights[k])) for k in old_network_pred_weights}
            #print(sum_changes_backbone+'\n',flush=True)
            #print(sum_changes_agg+'\n',flush=True)
            #print(sum_changes_network_pred+'\n',flush=True)

            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            
            #backbone_means = [mean_changes_backbone[key] for key in mean_changes_backbone]

 
            #agg_means = [mean_changes_agg[key] for key in mean_changes_agg]
            #print(backbone_means)
            #print(agg_means)            
            #mean_backbone = torch.mean(torch.stack(backbone_means), dim=0)
            #mean_agg = torch.mean(torch.stack(agg_means), dim=0)
            #print(mean_backbone)
            #print(mean_agg)            
            

           
    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    detailed_loss = []
    loss_foreach_dict = {}
    dot_foreach = {}
    target_foreach = {}

    for idx, input_seq in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        print(input_seq.shape) #[10, 4, 3, 5, 128, 128]
        print(B) #10
        print('model called next')
        [score_, mask_] = model(input_seq)
        print('score')
        print(score_.shape)

        print('mask')
        print(mask_.shape)
        # visualize
#        if (iteration == 0) or (iteration == args.print_freq):
#            if B > 2: input_seq = input_seq[0:2,:]
#            writer_train.add_image('input_seq',
#                                   de_normalize(vutils.make_grid(
#                                       input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), 
#                                       nrow=args.num_seq*args.seq_len)),
#                                   iteration)
        del input_seq
        
        if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)
        
        # score is a 6d tensor: [B, P, SQ, B, N, SQ]
        score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
        target_flattened = target_.view(B*NP*SQ, B2*NS*SQ)
        target_flattened = target_flattened.argmax(dim=1)

        loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

        accuracy_list[0].update(top1.item(),  B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        criterion_measure = nn.CrossEntropyLoss(reduction='none')       
        loss_foreach_dict[idx] = criterion_measure(score_flattened, target_flattened).view(B,SQ).mean(axis=1)
        
        dot_foreach[idx] = score_
        target_foreach[idx] = target_
        
        del score_

        
        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                   epoch, idx, len(data_loader), top1, top3, top5, time.time()-tic, loss=losses), flush=True)
            if args.dataset == 'gabors':
                print(data_loader.prev_seq[-1],flush=True)
                print(data_loader.prev_seq[-args.batch_size:])
                print(loss_foreach_dict[idx],flush=True)
                detailed_loss.append(losses.val)
                #yaml.dump(losses.val, open(os.getenv('SLURM_TMPDIR') + '/loss.yaml', 'w'))
                #yaml.dump(data_loader.prev_seq, open(os.getenv('SLURM_TMPDIR') + '/seq.yaml', 'w'))
        

#            writer_train.add_scalar('local/loss', losses.val, iteration)
#            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list], detailed_loss, loss_foreach_dict, dot_foreach, target_foreach


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_flattened.argmax(dim=1)

            
            loss = criterion(score_flattened, target_flattened)
            top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(),  B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
           epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=5,
                              big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds)
    elif args.dataset == 'gabors':
        pass
    else:
        raise ValueError('dataset not supported')

    if args.dataset == 'gabors':
        data_loader = GaborSequenceGenerator(batch_size=args.batch_size, num_trials=20, num_seq=args.num_seq, blank=args.blank, roll=args.roll, p_E=args.p_E, e_pos=args.e_pos, WIDTH=128, HEIGHT=128)
    else:
        sampler = data.RandomSampler(dataset)

        if mode == 'train':
            data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
        elif mode == 'val':
            data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
        print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = os.getenv('SLURM_TMPDIR')+'/log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt%s' % args.pretrain.replace('/','-').replace('.pth.tar', '') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

if __name__ == '__main__':
    main()
