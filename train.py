import argparse
import math
import os
import random
import shutil
import time
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from resnet import *
from utils.utils_data import *
from utils.utils_loss import *
from utils.cub200 import load_cub200

from utils.voc import load_voc

from resnet import *
from deco import DECoLoss

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'sun397', 'voc'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment/cifar10', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--data_dir', default='../codes/data/', type=str,
                    help='experiment directory for loading pre-generated data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 supported)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=800, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800',
                    help='where to decay lrt')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--eta', default=0.9, type=float,
                    help='final weight of reliable sample loss')
parser.add_argument('--t', default=2, type=float,
                    help='tau for logits-adjustment')
parser.add_argument('--alpha_range', default='0.2,0.6', type=str,
                    help='ratio of clean labels (alpha)')
parser.add_argument('--e', default=50, type=int,
                    help='warm-up training')

parser.add_argument('--partial_rate', default=0.3, type=float,
                    help='ambiguity level')
parser.add_argument('--hierarchical', default=False, type=bool,
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--imb_type', default='exp', choices=['exp', 'step'],
                    help='imbalance data type')
parser.add_argument('--imb_ratio', default=100, type=float,
                    help='imbalance ratio for long-tailed dataset generation')
parser.add_argument('--save_ckpt', action='store_true',
                    help='whether save the model')

parser.add_argument('--resume', default='', type=str, help='models path for load')
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension of mlp head')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--temp', default=0.1, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--warmup_epoch_head', default=80, type=int, help = 'head warmup epochs')
parser.add_argument('--warmup_epoch_tail', default=100, type=int, help = 'tail warmup epochs')
parser.add_argument('--records', action='store_true', help='use RECORDS')
parser.add_argument('--prot_start', default=80, type=int, help = 'Start Prototype Updating')


class Trainer():
    def __init__(self, args):
        self.args = args
        model_path = '{ds}_p{pr}_alpha{alpha}_tau{t}_ep{ep}_e{e}_imb_{it}{imr}_sd_{seed}'.format(
            ds=args.dataset,
            pr=args.partial_rate,
            ep=args.epochs,
            alpha=args.alpha_range,
            it=args.imb_type,
            imr=args.imb_ratio,
            seed=args.seed,
            t=args.t,
            e=args.e,
        )

        args.data_dir_prod = os.path.join(args.data_dir, 'pre-processed-data')
        args.exp_dir = os.path.join(args.exp_dir, model_path)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        if not os.path.exists(args.data_dir_prod):
            os.makedirs(args.data_dir_prod)

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            cudnn.deterministic = True

        if args.dataset == 'cifar10':
            args.num_class = 10
            many_shot_num = 4
            low_shot_num = 3
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt \
                = load_cifar(args=args)

        elif args.dataset == 'cifar100':
            args.num_class = 100
            many_shot_num = 33
            low_shot_num = 33
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt \
                = load_cifar(args=args)
        elif args.dataset == 'sun397':
            input_size = 224
            args.num_class = 397
            many_shot_num = 132
            low_shot_num = 132
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt = load_sun397(
                data_dir=args.data_dir,
                input_size=input_size,
                partial_rate=args.partial_rate,
                batch_size=args.batch_size)
        elif args.dataset == 'voc':
            train_loader, train_givenY, test_loader, train_label_cnt = load_voc(
                batch_size=args.batch_size, con=True)
            args.num_class = 20
            many_shot_num = 6
            low_shot_num = 7
        else:
            raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_givenY = train_givenY.cuda()
        self.acc_shot = AccurracyShot(train_label_cnt, args.num_class, many_shot_num, low_shot_num)

        self.train_label_cnt = train_label_cnt
        self.many_shot_thr = train_label_cnt.sort()[0][args.num_class - many_shot_num - 1]
        self.low_shot_thr = train_label_cnt.sort()[0][low_shot_num]


    def train(self):
        print("=> creating model 'resnet18'")
        if args.dataset in ['sun397', 'voc']:
            print('Loading Pretrained Model')
            model = DHNet_Atten(args.num_class, pretrained=True)
        else:
            model1 = DHNet_Atten(args.num_class)
            model2 = DHNet_Atten(args.num_class)
            model3 = ensemble(args.num_class)

        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()

        optimizer1 = torch.optim.SGD(model1.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        optimizer2 = torch.optim.SGD(model2.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        optimizer3 = torch.optim.SGD(model3.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)

        num_instance = self.train_givenY.shape[0]
        sel_stats = {
            'dist': torch.zeros(num_instance).cuda(),
            'is_rel': torch.ones(num_instance).bool().cuda(),
            'label':torch.zeros(num_instance).cuda(),
        }

        loss_fn = PLL_loss(self.train_givenY, mu=0.6)
        loss_con_rb = DECoLoss(contrast_dim=args.feat_dim, temperature=args.temp, num_classes=args.num_class).cuda()
        loss_con_ba = SupConLoss(temperature=0.1)

        self.loss_fn = loss_fn
        self.loss_con_rb = loss_con_rb
        self.loss_con_ba = loss_con_ba
        args.feat_mean = None

        args.start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume)
                else:
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                self.loss_fn.confidence = checkpoint['confidence'].cuda()
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))


        best_acc_ens = 0
        best_acc_head = 0
        best_acc_tail = 0

        for epoch in range(args.start_epoch, args.epochs):
            is_best_ens = False
            is_best_head = False
            is_best_tail = False

            adjust_learning_rate(args, optimizer1, epoch)
            adjust_learning_rate(args, optimizer2, epoch)
            s_time = time.time()
            alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * linear_rampup(epoch, args.e)
            if epoch >= args.prot_start:
                sel_stats1 = self.sample_selection(model1, model2, loss_fn, epoch, alpha, emp_dist_head=emp_dist, sel_stats=sel_stats)
                idx_chosen1 = torch.nonzero(sel_stats1['is_rel'])
                idx_chosen1 = idx_chosen1.squeeze()
                print(torch.bincount(sel_stats1['label'][idx_chosen1].long()))
                model1.train()
                model2.train()
                emp_dist, eq_sum_tail, eq_sum_head, acc_tail, acc_head, data, label = self.train_loop(model1, model2, model3, loss_fn, optimizer1, optimizer2, optimizer3, epoch, sel_stats1)
            else:
                emp_dist, eq_sum_tail, eq_sum_head, acc_tail, acc_head, data, label = self.train_loop(model1, model2, model3, loss_fn, optimizer1, optimizer2, optimizer3, epoch, None)
            e_time = time.time()
            print('Epoch {} training time: {:.2f} seconds'.format(epoch, e_time - s_time))
            
            emp_dist1 = list(map(lambda x: round(x, 2), emp_dist.tolist()))

            acc_test_tail, acc_many_tail, acc_med_tail, acc_few_tail = self.test(model1, model2, model3, self.test_loader, type=1)
            acc_test_head, acc_many_head, acc_med_head, acc_few_head = self.test(model1, model2, model3, self.test_loader, type=2)
            acc_test_ens, acc_many_ens, acc_med_ens, acc_few_ens = self.test(model1, model2, model3, self.test_loader, type=3)

            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write(
                    'Epoch {}/{}: Acc_tail {:.2f}, Best Acc_tail {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr1 {:.5f}) (lr2 {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_tail, best_acc_tail, acc_many_tail, acc_med_tail, acc_few_tail,
                        optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr']))

                f.write(
                    'Epoch {}/{}: Acc_head {:.2f}, Best Acc_head {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr1 {:.5f}) (lr2 {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_head, best_acc_head, acc_many_head, acc_med_head, acc_few_head,
                        optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr']))
                f.write(
                    'Epoch {}/{}: Acc_ens {:.2f}, Best Acc_ens {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr1 {:.5f}) (lr2 {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_ens, best_acc_ens, acc_many_ens, acc_med_ens, acc_few_ens,
                        optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr']))
                print('Epoch {}/{}: emp_dist {}'.format(epoch, args.epochs, emp_dist1))
                print('Epoch {}/{}: select_head {}'.format(epoch, args.epochs, eq_sum_head))
                print('Epoch {}/{}: select_tail {}'.format(epoch, args.epochs, eq_sum_tail))
                print('Epoch {}/{}: acc_pseu_tail {}'.format(epoch, args.epochs, acc_tail))
                print('Epoch {}/{}: acc_pseu_head {}'.format(epoch, args.epochs, acc_head))
                data = torch.cat(data, dim=0)
                label = torch.cat(label, dim=0)
                acc_many, acc_med, acc_few, _ = self.get_shot_acc(data, label)
                print('==> Logit_adj acc: [%.2f%%, %.2f%%, %.2f%%]\n' % (
                    acc_many, acc_med, acc_few))

            if acc_test_ens > best_acc_ens:
                best_acc_ens = acc_test_ens
                is_best_ens = True

            if acc_test_head > best_acc_head:
                best_acc_head = acc_test_head
                is_best_head = True
            
            if acc_test_tail > best_acc_tail:
                best_acc_tail = acc_test_tail
                is_best_tail = True

            if args.save_ckpt:
                self.save_checkpoint({
                    'confidence': loss_fn.confidence.detach(),
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'model1_state_dict': model1.state_dict(),
                    'model2_state_dict': model2.state_dict(),
                    'model3_state_dict': model3.state_dict(),
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'optimizer3': optimizer3.state_dict(),
                }, is_best=is_best_ens, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
                    best_file_name='{}/checkpoint_best_ens.pth.tar'.format(args.exp_dir))
    
    def get_shot_acc(self, preds, labels, acc_per_cls=False):
        sum=0
        self.test_class_count = []
        for l in range(args.num_class):
            self.test_class_count.append(len(labels[labels == l]))

        class_correct = []
        for l in range(args.num_class):
            class_correct.append((preds[labels == l] == labels[labels == l]).sum())

        many_shot = []
        median_shot = []
        low_shot = []
        for i in range(args.num_class):
            if self.train_label_cnt[i] > self.many_shot_thr:
                many_shot.append((class_correct[i] / float(self.test_class_count[i])))
                sum += class_correct[i]
            elif self.train_label_cnt[i] < self.low_shot_thr:
                low_shot.append((class_correct[i] / float(self.test_class_count[i])))
                sum += class_correct[i]
            else:
                median_shot.append((class_correct[i] / float(self.test_class_count[i])))
                sum += class_correct[i]

        if len(many_shot) == 0:
            many_shot.append(0)
        if len(median_shot) == 0:
            median_shot.append(0)
        if len(low_shot) == 0:
            low_shot.append(0)

        if acc_per_cls:
            class_accs = [c / cnt for c, cnt in zip(class_correct, self.test_class_count)] 
            return np.mean(many_shot) * 100, np.mean(median_shot) * 100, np.mean(low_shot) * 100, class_accs
        else:
            return np.mean(many_shot) * 100, np.mean(median_shot) * 100, np.mean(low_shot) * 100, sum

    def get_high_confidence(self, loss_vec,  pseudo_label_idx, nums_vec):
        idx_chosen = []
        chosen_flags = torch.zeros(len(loss_vec)).cuda()
        for j, nums in enumerate(nums_vec):
            indices = np.where(pseudo_label_idx.cpu().numpy() == j)[0]
            if len(indices) == 0:
                continue
            loss_vec_j = loss_vec[indices]
            sorted_idx_j = loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(nums)), len(indices)), 1)
            idx_chosen.append(indices[sorted_idx_j[:partition_j]])
        idx_chosen = np.concatenate(idx_chosen)
        chosen_flags[idx_chosen] = 1

        idx_chosen = torch.where(chosen_flags == 1)[0]
        return idx_chosen
    def sample_selection(self, model1, model2, loss_fn, epoch, alpha, emp_dist_head, sel_stats):
        train_loader = self.train_loader
        emp_dist_tail = torch.Tensor([1 / args.num_class for _ in range(args.num_class)]).cuda()
        model1.eval()
        model2.eval()
        bs = 0
        loss_vec = []
        pred_vec = []
        for i, (images_w, images_w1, images_s, labels, true_labels, index) in enumerate(train_loader):
            X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
            bs += X_w.shape[0]
            X_w1 = images_w1.cuda()
            Y_true = true_labels.long().detach().cuda()

            logits_w_head, feat_w_head, z1_head, p1_head = model2(X_w)
            logits_w_tail, feat_w_tail, z1_tail, p1_tail = model1(X_w)

            logit_adj = F.softmax(logits_w_head - args.t * torch.log(emp_dist_head), dim=1)
            sel_stats['label'][index] = copy.deepcopy(torch.argmax(logit_adj, dim=1).float().clone().detach())

            _, ce_loss_vec = loss_fn(logits_w_tail, None, targets=logit_adj)
            sel_stats['dist'][index] = copy.deepcopy(ce_loss_vec.clone().detach())
        
        loss_vec = sel_stats['dist']
        pred_vec = sel_stats['label']

        r_vec = emp_dist_tail * bs * alpha
        idx_chosen = self.get_high_confidence(loss_vec, pred_vec, r_vec.tolist())
        n = loss_vec.shape[0]
        is_rel = torch.zeros(n).bool().cuda()
        is_rel[idx_chosen] = True
        sel_stats['is_rel'] = is_rel
        return sel_stats

    def get_loss(self, X_w, logits_w, logits_s, ce_label, Y, index, model, loss_fn, emp_dist, alpha, eta, epoch,
                 is_tail, feat_w, feat_w1, feat_s, common_idx):
        bs = X_w.shape[0]
        
        prediction = F.softmax(logits_w.detach(), dim=1)
        prediction_adj = prediction * Y
        prediction_adj = prediction_adj / prediction_adj.sum(dim=1, keepdim=True)
        
        _, ce_loss_vec = loss_fn(logits_w, None, targets=ce_label)

        loss_pseu, _ = loss_fn(logits_w, index)
        pseudo_label_idx = ce_label.max(dim=1)[1]
        r_vec = emp_dist * bs * alpha

        if is_tail:
            if epoch < args.prot_start:
                idx_chosen = self.get_high_confidence(ce_loss_vec, pseudo_label_idx, r_vec.tolist())
            else:
                idx_chosen = common_idx
        else:
            idx_chosen = self.get_high_confidence(ce_loss_vec, pseudo_label_idx, r_vec.tolist())

        if is_tail:
            contrast_logit = self.loss_con_rb(feat_w[idx_chosen], ce_label[idx_chosen].max(dim=1)[1], args=args)
            loss_scl_tail = F.cross_entropy(contrast_logit + torch.log(emp_dist), ce_label[idx_chosen].max(dim=1)[1])
        else:
            feat_w, feat_w1 =  F.normalize(feat_w, dim=1), F.normalize(feat_w1, dim=1)
            features = torch.cat([feat_w.unsqueeze(1), feat_w1.unsqueeze(1)], dim=1)
            loss_scl_head = self.loss_con_ba(features[idx_chosen], ce_label[idx_chosen].max(dim=1)[1])

        if epoch < 1 or len(idx_chosen) == 0:
            loss = loss_pseu
        else:
            loss_ce, _ = loss_fn(logits_s[idx_chosen], None, targets=ce_label[idx_chosen])

            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            X_w_c = X_w[idx_chosen]
            ce_label_c = ce_label[idx_chosen]
            idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[idx]
            ce_label_c_rand = ce_label_c[idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
            ce_label_c_mix = l * ce_label_c + (1 - l) * ce_label_c_rand
            if is_tail:
                logits_mix, _, _, _ = model(X_w_c_mix)
            else:
                logits_mix, _, _, _ = model(X_w_c_mix)
            loss_mix, _ = loss_fn(logits_mix, None, targets=ce_label_c_mix)
            if is_tail and epoch >= args.warmup_epoch_tail: 
                loss = (loss_mix + loss_ce) * eta + loss_pseu + loss_scl_tail
            elif not is_tail and epoch >= args.warmup_epoch_head:
                loss = (loss_mix + loss_ce) * eta + loss_pseu + loss_scl_head
            else:
                loss = (loss_mix + loss_ce) * eta + loss_pseu

        return loss, prediction_adj, idx_chosen

    def train_loop(self, model1, model2, model3, loss_fn, optimizer1, optimizer2, optimizer3, epoch, sel_stats):
        args = self.args
        train_loader = self.train_loader

        batch_time = AverageMeter('Time', ':1.2f')
        data_time = AverageMeter('DataTime', ':1.2f')
        acc_head = AverageMeter('Acc@head', ':2.2f')
        acc_con = AverageMeter('Acc@con', ':2.2f')
        acc_tail = AverageMeter('Acc@tail', ':2.2f')
        acc_en = AverageMeter('Acc@en', ':2.2f')
        acc_logt_adj =  AverageMeter('Acc@la', ':2.2f')
        loss_head_log = AverageMeter('Loss@head', ':2.2f')
        loss_tail_log = AverageMeter('Loss@tail', ':2.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, acc_head, acc_con, acc_tail, acc_en, acc_logt_adj, loss_head_log, loss_tail_log],
            prefix="Epoch: [{}]".format(epoch))

        model1.train()
        model2.train()
        model3.train()

        eta = args.eta * linear_rampup(epoch, args.e)
        alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * linear_rampup(epoch, args.e)

        end = time.time()
        emp_dist_tail = torch.Tensor([1 / args.num_class for _ in range(args.num_class)]).cuda()
        emp_dist_head = loss_fn.confidence.sum(0) / loss_fn.confidence.sum()
        eq_sum_head=0
        eq_sum_tail=0
        sum_head = 0
        sum_tail = 0
        sum_acc = 0
        eq_sum_acc_head = 0
        eq_sum_acc_tail = 0
        pred_list = []
        true_list = []
        
        class_correct = torch.zeros(args.num_class).cuda()
        class_total = torch.zeros(args.num_class).cuda()
        
        for i, (images_w, images_w1, images_s, labels, true_labels, index) in enumerate(train_loader):
            data_time.update(time.time() - end)

            X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
            X_w1 = images_w1.cuda()
            Y_true = true_labels.long().detach().cuda()
            logits_w_head, feat_w_head, z1_head, p1_head = model2(X_w)
            logits_s_head, feat_s_head, z2_head, p2_head = model2(X_s)

            _, feat_w_head1, _, _ = model2(X_w1)
            
            logits_w_tail, feat_w_tail, z1_tail, p1_tail = model1(X_w)
            logits_s_tail, feat_s_tail, z2_tail, p2_tail = model1(X_s)
            
            _, feat_w_tail1, _, _ = model1(X_w1)

            pseudo_label = loss_fn.confidence[index]
            pseudo_pred = pseudo_label.max(dim=1)[1]
            for class_idx in range(args.num_class):
                class_mask = Y_true == class_idx
                if class_mask.sum() > 0:
                    class_correct[class_idx] += (pseudo_pred[class_mask] == Y_true[class_mask]).sum()
                    class_total[class_idx] += class_mask.sum()
                    
            if epoch >= args.prot_start:
                all_idx = torch.nonzero(sel_stats['is_rel']).squeeze()
                common_idx = np.intersect1d(index.cpu().numpy(), all_idx.cpu().numpy())
                common_idx_chosen = [np.where(index.cpu().numpy() == label)[0][0] for label in common_idx]
                
                if len(common_idx_chosen) < 1:
                    print(common_idx_chosen)
                    continue

            loss_head, prediction_head, idx_chosen_head = self.get_loss(X_w, logits_w_head, logits_s_head, pseudo_label, Y, index, model2,
                                                     loss_fn, emp_dist_head, alpha, eta, epoch, False, feat_w_head, feat_w_head1, feat_s_head, None)
            logit_adj = F.softmax(logits_w_head - args.t * torch.log(emp_dist_head), dim=1)
            if epoch >= args.prot_start:
                loss_tail, prediction_tail, idx_chosen_tail = self.get_loss(X_w, logits_w_tail, logits_s_tail, logit_adj, Y, index, model1,
                                                     loss_fn, emp_dist_tail, alpha, eta, epoch, True, feat_w_tail, feat_w_tail1, feat_s_head, common_idx_chosen)
            else:
                loss_tail, prediction_tail, idx_chosen_tail = self.get_loss(X_w, logits_w_tail, logits_s_tail, logit_adj, Y, index, model1,
                                                     loss_fn, emp_dist_tail, alpha, eta, epoch, True, feat_w_tail, feat_w_tail1, feat_s_head, None)
            eq_count_head = torch.eq(logits_w_head[idx_chosen_head].max(dim=1)[1], Y_true[idx_chosen_head]).sum().item()
            eq_sum_head += eq_count_head
            sum_head += len(idx_chosen_head)
            eq_count_tail = torch.eq(logits_w_tail[idx_chosen_tail].max(dim=1)[1], Y_true[idx_chosen_tail]).sum().item()
            eq_sum_tail += eq_count_tail
            sum_tail += len(idx_chosen_tail)
            eq_count_acc_head = torch.eq(pseudo_label[idx_chosen_head].max(dim=1)[1], Y_true[idx_chosen_head]).sum().item()
            eq_sum_acc_head += eq_count_acc_head
            eq_count_acc_tail = torch.eq(logit_adj[idx_chosen_tail].max(dim=1)[1], Y_true[idx_chosen_tail]).sum().item()
            eq_sum_acc_tail += eq_count_acc_tail

            pred_list.append(logit_adj[idx_chosen_tail].max(dim=1)[1].cpu())
            true_list.append(Y_true[idx_chosen_tail].cpu())

            sum_acc += X_w.shape[0]

            fusion_pred = model3(logits_w_head.detach(), logits_w_tail.detach(), emp_dist_head)
            fusion_loss = torch.sum(-pseudo_label * torch.log(fusion_pred+1e-8))/fusion_pred.shape[0]
            ratio = 0.5 * linear_rampup(epoch, args.epochs)
            
            loss_con_head = self.SimSiamLoss(p1_head, z2_head) + self.SimSiamLoss(p2_head, z1_head)
            loss_con_tail = self.SimSiamLoss(p1_tail, z2_tail) + self.SimSiamLoss(p2_tail, z1_tail)

            loss = loss_head + loss_tail + ratio * fusion_loss + 10*loss_con_head + 10*loss_con_tail

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            loss_head_log.update(loss_head.item())
            loss_tail_log.update(loss_tail.item())
            acc = accuracy(logits_w_head, Y_true)[0]
            acc_head.update(acc[0])

            acc = accuracy(logits_w_tail, Y_true)[0]
            acc_tail.update(acc[0])

            acc = accuracy(fusion_pred.detach(), Y_true)[0]
            acc_en.update(acc[0])

            acc = accuracy(pseudo_label, Y_true)[0]
            acc_con.update(acc[0])
            acc = accuracy(logit_adj[idx_chosen_tail], Y_true[idx_chosen_tail])[0]
            acc_logt_adj.update(acc[0])

            loss_fn.confidence_move_update(prediction_tail, index)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        
        if i > 0 and class_total.sum() > 0:
            print('==> Pseudo-label Per-class Accuracy:')
            for class_idx in range(args.num_class):
                if class_total[class_idx] > 0:
                    class_acc = class_correct[class_idx] / class_total[class_idx] * 100
                    print(f'Class {class_idx}: {class_acc:.2f}% ({class_total[class_idx].item()} samples)')
        
        return emp_dist_head, eq_sum_tail/sum_tail, eq_sum_head/sum_head, eq_sum_acc_tail/sum_tail, eq_sum_acc_head/sum_head, pred_list, true_list
    
    def SimSiamLoss(self, p, z, version='simplified'):
        z = z.detach()

        if version == 'original':
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def test(self, model1, model2, model3, test_loader, type=1):
        with torch.no_grad():
            if type == 1:
                print('==> Evaluation tail...')
            elif type == 2:
                print('==> Evaluation head...')
            else:
                print('==> Evaluation ensemble...')
            model1.eval()
            model2.eval()
            model3.eval()   

            pred_list = []
            true_list = []
            for _, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                if type == 1:
                    outputs, _, _, _ = model1(images)
                    pred = F.softmax(outputs, dim=1)
                elif type == 2:
                    outputs, _, _, _ = model2(images)
                    pred = F.softmax(outputs, dim=1)
                else:
                    logit_tail, _, _, _ = model1(images)
                    logit_head, _, _, _ = model2(images)
                    pred = model3(logit_head, logit_tail, self.loss_fn.get_distribution())

                pred_list.append(pred.cpu())
                true_list.append(labels)

            pred_list = torch.cat(pred_list, dim=0)
            true_list = torch.cat(true_list, dim=0)

            acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
            acc_many, acc_med, acc_few = self.acc_shot.get_shot_acc(pred_list.max(dim=1)[1], true_list)
            print('==> Test Accuracy is %.2f%% (%.2f%%), [%.2f%%, %.2f%%, %.2f%%]' % (
                acc1, acc5, acc_many, acc_med, acc_few))

        return float(acc1), float(acc_many), float(acc_med), float(acc_few)


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_file_name)


if __name__ == '__main__':
    args = parser.parse_args()

    [args.alpha_start, args.alpha_end] = [float(item) for item in args.alpha_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.imb_factor = 1. / args.imb_ratio
    print(args)
    
    trainer = Trainer(args)
    trainer.train()
