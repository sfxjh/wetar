from trainer import register_trainer
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from utils.hook import EmbeddingHook
from .base import BaseTrainer
import itertools
import higher
from utils.my_utils import convert_batch_to_bert_input_dict
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
import logging
from tqdm import tqdm


@register_trainer('l2rew')
class LearnToReweightTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None,
                 train_len=0):
        BaseTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
        if args.sgd_lr > 0:
            self.inner_opt = SGD(model.parameters(), lr=args.sgd_lr)
        else:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
            ]
            self.inner_opt = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        batch_size = data_loader.batch_size
        init_weights = torch.ones((train_len, 1), device=model.device) / batch_size
        self.global_weights = torch.zeros((train_len, args.epochs - 1), device=model.device)
        self.global_weights = torch.cat((init_weights, self.global_weights), dim=-1)
        self.weighted_average = torch.tensor([2**i for i in range(0, args.epochs)], dtype=torch.long, device=model.device)
        self.elu=torch.nn.ELU(alpha=args.alpha)
        


    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument('--sgd_lr', default=0, type=float)
        group.add_argument('--augment_ratio', default=0, type=float)
        group.add_argument('--augment_method', default='textfooler', type=str)
        group.add_argument('--max_norm', default=0, type=float)
        group.add_argument('--adv_init_method', default='grad', choices=['rand', 'grad'])
        group.add_argument('--adv_init_mag', default=0.05, type=float,
                            help='Magnitude of initial (adversarial?) perturbation')
        group.add_argument('--eta_start', default='zeros', choices=['ones', 'zeros'])
        group.add_argument('--adv_learning_rate', default=0.05, type=float,
                            help='Step size of gradient ascent')
        group.add_argument('--epsilon', default=0, type=float)
        group.add_argument('--epochs_update_valid', default=1, type=int)
        group.add_argument('--epsilon_dir', default=-1, type=float)
        group.add_argument('--global_weight', default='False', type=str)
        group.add_argument('--base_aug', default='False', type=str)
        # group.add_argument('--adv-steps', default=5, type=int,
        #                     help='Number of gradient ascent steps for the adversary')

        # group.add_argument('--adv-max-norm', default=0.0, type=float,
        #                     help='adv_max_norm = 0 means unlimited')
        # group.add_argument('--adv-norm-type', default='l2', type=str,
        #                     help='norm type of the adversary')
        # group.add_argument('--adv-change-rate', default=0.2, type=float,
        #                     help='change rate of a sentence')
        group.add_argument('--valid_batch_size',default=32,type=int,help='batch size of validation set')
        group.add_argument('--valid_iter',default=1,type=int,help='one train batch corresponds to how much valid batch')
        group.add_argument('--alpha',default=0.0,type=float,help='alpha for elu activation')

    def train_epoch(self, args, epoch: int, valid_rob_dataloader) -> None:
        print("Epoch {}:".format(epoch))
        logging.info("Epoch {}:".format(epoch))
        self.epoch_now = epoch
        self.model.train()
        self.valid_rob_dataloader = itertools.cycle(valid_rob_dataloader)
        epoch_iterator = tqdm(self.data_loader)
        oom_number = 0
        for batch in epoch_iterator:
            try:
                loss = self.train_batch(args, batch)
                epoch_iterator.set_description('loss: {:.4f}'.format(loss))
            except RuntimeError as e:
                raise e
        logging.warning('oom number : {}, oom rate : {:.2f}%'.format(oom_number, oom_number / len(self.data_loader) * 100))
        return

    def train_batch(self, args, batch: Tuple) -> float:
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.train(args, batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.write_tensorboard(loss)
        if args.local_rank == 0:
            self.global_step += 1
        return loss

    def base_train_batch(self, args, batch: Tuple) -> float:
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.base_train(args, batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.write_tensorboard(loss)
        if args.local_rank == 0:
            self.global_step += 1
        return loss

    def base_train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device)for t in batch)
        logits = self.forward(args, batch)[0]
        golds = batch[3]
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def delta_initial(self, args, embedding: torch.Tensor, attention_mask: torch.Tensor):
        if args.adv_init_mag > 0:
            input_mask = attention_mask.to(embedding)
            input_lengths = torch.sum(input_mask, 1)
            delta = torch.zeros_like(embedding).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = input_lengths * embedding.size(-1)
            magnitude = args.adv_init_mag / torch.sqrt(dims)
            delta = (delta * magnitude.view(-1, 1, 1).detach())
        else:
            delta = torch.zeros_like(embedding)
        return delta

    def delta_update(self, args, embedding: torch.Tensor, delta: torch.Tensor, delta_grad: torch.Tensor):
        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
        denorm = torch.clamp(denorm, min=1e-8)
        delta = (delta + args.adv_learning_rate * delta_grad / denorm).detach()
        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
        exceed_mask = (delta_norm > args.max_norm).to(embedding)
        reweights = (args.max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
        delta = (delta * reweights).detach()
        return delta
    
    def delta_norm(self, args, delta: torch.Tensor, eta_grads: torch.Tensor):
        eta_l1_norm = torch.norm(eta_grads.view(eta_grads.size(0), -1), p=1, dim=1).view(delta.size(0), -1, 1)
        eta_l1_norm = torch.clamp(eta_l1_norm, min=1e-8)
        eta_w = eta_grads / eta_l1_norm
        delta = delta * eta_w
        denorm = torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1, 1)
        denorm = torch.clamp(denorm, min=1e-8)
        delta = (args.max_norm * delta / denorm).detach() # max_norm 太粗糙了这个，没有考虑中间的
        return delta

    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device)for t in batch)
        golds = batch[3]
        input_nums = batch[4]
        # input_nums=0

        if args.max_norm > 0:
            batch_in_token_ids = batch[0]
            attention_mask = batch[1]
            word_embedding_layer = self.model.get_input_embeddings()
            embedding_init = word_embedding_layer(batch_in_token_ids)
            delta = self.delta_initial(args, embedding_init, attention_mask)
            if args.adv_init_method == 'grad':
                delta.requires_grad_()
                batch = (delta + embedding_init, batch[1], batch[2])
                logits = self.forward(args, batch)[0]
                losses = self.loss_function(logits, golds.view(-1))
                loss = torch.mean(losses)
                loss.backward()
                delta_grad = delta.grad.clone().detach()
                delta = self.delta_update(args, embedding_init, delta, delta_grad)
                self.model.zero_grad()
                self.optimizer.zero_grad()

        if args.max_norm > 0:
            if args.eta_start == 'ones':
                eta = torch.ones((delta.size(0), delta.size(1)), device=delta.device, requires_grad=True).view(
                    delta.size(0), -1, 1)
            elif args.eta_start == 'zeros':
                eta = torch.zeros((delta.size(0), delta.size(1)), device=delta.device, requires_grad=True).view(
                    delta.size(0), -1, 1)
            delta_rew = delta * eta
            embedding_init = word_embedding_layer(batch[0])
            batch = (delta_rew + embedding_init, batch[1], batch[2])
        eta=None

        # with higher.innerloop_ctx(self.model, self.inner_opt) as (meta_model, meta_opt):
        #     # 1. Update meta model on the training data
        #     if args.max_norm > 0:
        #         if args.eta_start == 'ones':
        #             eta = torch.ones((delta.size(0), delta.size(1)), device=delta.device, requires_grad=True).view(delta.size(0), -1, 1)
        #         elif args.eta_start == 'zeros':
        #             eta = torch.zeros((delta.size(0), delta.size(1)), device=delta.device, requires_grad=True).view(delta.size(0), -1, 1)
        #         delta_rew = delta * eta
        #         embedding_init = word_embedding_layer(batch_in_token_ids)
        #         batch = (delta_rew + embedding_init, batch[1], batch[2])
        #     inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
        #     meta_logits = meta_model(**inputs)[0]
        #     meta_losses = self.loss_function(meta_logits, golds.view(-1))
        #     epsilon = torch.zeros_like(meta_losses, requires_grad=True)
        #     epsilon = torch.clamp(epsilon, min=args.epsilon)
        #     meta_loss = torch.mean(epsilon * meta_losses)
        #     meta_opt.step(meta_loss)
        #
        #     # 2. Calculate the grads of eps on the validation set
        #     valid_batch = next(self.valid_rob_dataloader)
        #     valid_batch = tuple(t.to(self.model.device) for t in valid_batch)
        #     valid_golds = valid_batch[3]
        #     valid_inputs = convert_batch_to_bert_input_dict(valid_batch, args.model_type)
        #     valid_logits = meta_model(**valid_inputs)[0]
        #     valid_loss = torch.mean(self.loss_function(valid_logits, valid_golds.view(-1)))
        #     if args.max_norm > 0:
        #         epsilon_grads, eta_grads = torch.autograd.grad(valid_loss, (epsilon, eta))
        #         epsilon_grads = epsilon_grads.detach()
        #         eta_grads = eta_grads.detach()
        #     else:
        #         epsilon_grads = torch.autograd.grad(valid_loss, epsilon)[0].detach()
        #
        # w_tilde = torch.clamp(-epsilon_grads, min=0)
        # l1_norm = torch.sum(w_tilde)
        # if l1_norm != 0:
        #     w = w_tilde / l1_norm
        # else:
        #     w = w_tilde
        #

        w_tilde=None
        eta_grads=None
        for _ in range(args.valid_iter):
            w_new,eta_new=self.valid_iter(args,batch,eta)
            if(w_tilde is None):
                w_tilde=torch.zeros_like(w_new)
            w_tilde+=w_new
            if(args.max_norm>0):
                if(eta_grads is None):
                    eta_grads=torch.zeros_like(eta_new)
                eta_grads+=eta_new

        if args.alpha==0.0:
            w_tilde = torch.clamp(w_tilde, min=0)
        else:
            w_tilde=self.elu(-w_tilde) + args.alpha
        assert torch.min(w_tilde)>=0

        l1_norm = torch.sum(w_tilde)
        if l1_norm != 0:
            w = w_tilde / l1_norm
        else:
            w = w_tilde


        if args.max_norm > 0:
            eta_grads = eta_grads / torch.norm(eta_grads, p=1).view(-1, 1)
            eta_reweights = delta.size(1) * args.epsilon_dir
            delta = self.delta_norm(args, delta, eta +  eta_reweights * eta_grads)
            embedding_init = word_embedding_layer(batch_in_token_ids)
            batch = (delta + embedding_init, batch[1], batch[2])

        for i in range(len(input_nums)):
            self.global_weights[input_nums[i]][self.epoch_now] = w[i]
        if args.global_weight == 'True':
            for i in range(len(input_nums)):
        #         self.global_weights[input_nums[i]][self.epoch_now] = w[i]
                w[i] = torch.sum(self.global_weights[input_nums[i]]*self.weighted_average)
            w = w / torch.sum(w)

        logits = self.forward(args, batch)[0]
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(w * losses) * args.batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def valid_iter(self, args, batch: Tuple,eta) -> Tuple:
        eta_grads=None
        with higher.innerloop_ctx(self.model, self.inner_opt) as (meta_model, meta_opt):
            # 1. Update meta model on the training data
            inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
            meta_logits = meta_model(**inputs)[0]
            meta_losses = self.loss_function(meta_logits, batch[3].view(-1))
            epsilon = torch.zeros_like(meta_losses, requires_grad=True)
            epsilon = torch.clamp(epsilon, min=args.epsilon)
            meta_loss = torch.mean(epsilon * meta_losses)
            meta_opt.step(meta_loss)

            # 2. Calculate the grads of eps on the validation set
            valid_batch = next(self.valid_rob_dataloader)
            valid_batch = tuple(t.to(self.model.device) for t in valid_batch)
            valid_golds = valid_batch[3]
            valid_inputs = convert_batch_to_bert_input_dict(valid_batch, args.model_type)
            valid_logits = meta_model(**valid_inputs)[0]
            valid_loss = torch.mean(self.loss_function(valid_logits, valid_golds.view(-1)))
            if args.max_norm > 0:
                epsilon_grads, eta_grads = torch.autograd.grad(valid_loss, (epsilon, eta))
                epsilon_grads = epsilon_grads.detach()
                eta_grads = eta_grads.detach()
                # eta_grads = eta_grads / torch.norm(eta_grads, p=1).view(-1, 1)
            else:
                epsilon_grads = torch.autograd.grad(valid_loss, epsilon)[0].detach()

            # if args.alpha==0.0:
            #     w_tilde = torch.clamp(-epsilon_grads, min=0)
            # else:
            #     w_tilde=self.elu(-epsilon_grads)+args.alpha
            # assert torch.min(w_tilde)>0
            w_tilde = -epsilon_grads

            return w_tilde,eta_grads

    def save_global_weights(self, save_dir, file_name):
        save_file_name = '{}.pth'.format(file_name)
        from utils.public import check_and_create_path
        import os
        check_and_create_path(save_dir)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.global_weights, save_path)
        logging.info('Saving global_weights to {}'.format(save_path))
