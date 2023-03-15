from trainer import register_trainer
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from .gradient import EmbeddingLevelGradientTrainer
from .base import BaseTrainer
import logging
from tqdm import tqdm
from utils.instance_utils import GAIR, entropy_reweight


@register_trainer('iwpgd')
class InstanceWisePGDTrainer(EmbeddingLevelGradientTrainer):
    def __init__(self,
                 args,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        EmbeddingLevelGradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
        self.Lambda = args.Lambda

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument('--adv_steps', default=5, type=int,
                            help='Number of gradient ascent steps for the adversary')
        group.add_argument('--adv_learning_rate', default=0.03, type=float,
                            help='Step size of gradient ascent')
        group.add_argument('--adv_init_mag', default=0.05, type=float,
                            help='Magnitude of initial (adversarial?) perturbation')
        group.add_argument('--adv_max_norm', default=0.0, type=float,
                            help='adv_max_norm = 0 means unlimited')
        group.add_argument('--adv_norm_type', default='l2', type=str,
                            help='norm type of the adversary')
        group.add_argument('--adv_change_rate', default=0.2, type=float,
                            help='change rate of a sentence')
        group.add_argument('--reweight_type', default='geometry', choices=['geometry', 'entropy'])
        group.add_argument('--Lambda',type=str, default='-1.0', help='parameter for GAIR')
        group.add_argument('--Lambda_max',type=float, default=float('inf'), help='max Lambda')
        group.add_argument('--Lambda_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
        group.add_argument('--weight_assignment_function', default='Tanh', choices=['Discrete','Sigmoid','Tanh'])
        group.add_argument('--begin_epoch', default=1, type=int)
        group.add_argument('--tau', default=0, type=float)

    def train_epoch(self, args, epoch: int) -> None:
        # if args.local_rank == 0:
        print("Epoch {}:".format(epoch))
        logging.info("Epoch {}:".format(epoch))
        self.epoch_now = epoch
        self.model.train()
        self.Lambda = self.adjust_Lambda(args, epoch+1)
        epoch_iterator = tqdm(self.data_loader)
        oom_number = 0
        for batch in epoch_iterator:
            try:
                loss = self.train_batch(args, batch)
                epoch_iterator.set_description('loss: {:.4f}'.format(loss))
            except RuntimeError as e:
                if args.local_rank == 0:
                    if "out of memory" in str(e):
                        logging.warning('oom in batch forward / backward pass, attempting to recover from OOM')
                        print('oom in batch forward / backward pass, attempting to recover from OOM')
                        self.model.zero_grad()
                        self.optimizer.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        oom_number += 1
                    else:
                        raise e
        if args.local_rank == 0:
            logging.warning('oom number : {}, oom rate : {:.2f}%'.format(oom_number, oom_number / len(self.data_loader) * 100))
        return

    def get_adversarial_examples_Kappa(self, args, batch: Tuple) -> Tuple:
        word_embedding_layer = self.model.get_input_embeddings()

        # init input_ids and mask, sentence length
        batch_in_token_ids = batch[0]
        attention_mask = batch[1]
        golds = batch[3]
        Kappa = torch.zeros(len(batch_in_token_ids))

        embedding_init = word_embedding_layer(batch_in_token_ids)

        delta = EmbeddingLevelGradientTrainer.delta_initial(args, embedding_init, attention_mask)

        for astep in range(args.adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = (delta + embedding_init, batch[1], batch[2])
            logits = self.forward(args, batch)[0]

            predict = logits.max(1, keepdim=True)[1]
            for p in range(len(batch_in_token_ids)):
                if predict[p] == golds[p]:
                    Kappa[p] += 1
            
            # (1) backward
            losses = self.loss_function(logits, golds.view(-1))
            loss = torch.mean(losses)
            loss.backward()

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            delta = self.delta_update(args, embedding_init, delta, delta_grad)

            self.model.zero_grad()
            self.optimizer.zero_grad()
            embedding_init = word_embedding_layer(batch_in_token_ids)

        delta.requires_grad = False
        return (embedding_init + delta, batch[1], batch[2]), Kappa

    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device) for t in batch)
        golds = batch[3]

        # for PGD-K, clean batch is not used when training
        adv_batch, Kappa = self.get_adversarial_examples_Kappa(args, batch)

        self.model.zero_grad()
        self.optimizer.zero_grad()

        # (0) forward
        logits = self.forward(args, adv_batch)[0]
        # (1) backward
        losses = self.loss_function(logits, golds.view(-1))

        if args.reweight_type == 'geometry':
            reweights = GAIR(args.adv_steps, Kappa, self.Lambda, args.weight_assignment_function).cuda()
        elif args.reweight_type == 'entropy':
            tau = args.tau if args.tau != 0 else 1
            reweights = entropy_reweight(logits, tau)
        loss = losses.mul(reweights).mean()
        loss.backward()
        # (2) update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def adjust_Lambda(self, args, epoch):
        Lam = float(args.Lambda)
        Lambda = args.Lambda_max
        begin_epoch = args.begin_epoch
        if args.Lambda_schedule == 'linear':
            if epoch >= begin_epoch:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= begin_epoch:
                Lambda = Lam
            elif epoch >= 2*begin_epoch:
                Lambda = Lam-2.0
        elif args.Lambda_schedule == 'fixed':
            if epoch >= begin_epoch:
                Lambda = Lam
        return Lambda