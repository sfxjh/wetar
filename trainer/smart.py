from trainer import register_trainer
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from .gradient import EmbeddingLevelGradientTrainer
from .base import BaseTrainer
from utils.smart_perturbation import SmartPerturbation


@register_trainer('smart')
class SMARTTrainer(EmbeddingLevelGradientTrainer):
    def __init__(self,args,
                 data_loader: DataLoader,
                 model: nn.Module,#BertForSequenceClassification
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        EmbeddingLevelGradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
        self.adv_teacher=SmartPerturbation(epsilon=args.adv_epsilon,step_size=args.adv_step_size,noise_var=args.adv_noise_var,norm_p=args.adv_p_norm,k=args.adv_k,
        norm_level=args.adv_norm_level)
        """
        1.L∞范数的计算问题,epsilon是否等于adv_max_norm?
        2.更新noise时也应该使用symmetric loss?
        3.momentum acceleration
        """
    
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument("--adv_opt", default=0, type=int)
        group.add_argument("--adv_norm_level", default=0, type=int)
        group.add_argument("--adv_p_norm", default="inf", type=str)
        group.add_argument("--adv_alpha", default=1, type=float)
        group.add_argument("--adv_k", default=1, type=int)
        group.add_argument("--adv_step_size", default=1e-3, type=float)
        group.add_argument("--adv_noise_var", default=1e-5, type=float)
        group.add_argument("--adv_epsilon", default=1e-5, type=float)


    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device) for t in batch)
        golds = batch[3]

        # for PGD-K, clean batch is not used when training
        # adv_batch = self.get_adversarial_examples(args, batch)

        self.model.zero_grad()
        self.optimizer.zero_grad()

        # (0) forward
        logits = self.forward(args, batch)[0]
        # (1) ce loss
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        # (2) adv_loss
        adv_loss,emb_val,eff_perturb=self.adv_teacher.forward(self.model,logits,batch[0],attention_mask=batch[1],token_type_ids=batch[2])#batch[0]是input_ids?
        # (3) backward
        loss=loss+args.adv_alpha*adv_loss
        loss.backward()
        # (4) update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()