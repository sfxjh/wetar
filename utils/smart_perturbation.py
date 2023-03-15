from copy import deepcopy
import torch
import logging
import random
from torch.nn import Parameter
from functools import wraps
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

logger = logging.getLogger(__name__)


def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()

class SmartPerturbation:
    def __init__(
        self,
        epsilon=1e-6,#1e-5 or 1e-6?
        multi_gpu_on=False,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p="inf",
        k=1,
        fp16=False,
        loss_map=[],
        norm_level=0,
    ):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.norm_level = norm_level > 0

    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == "l2":
            if sentence_level:
                direction = grad / (
                    torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon
                )
            else:
                direction = grad / (
                    torch.norm(grad, dim=-1, keepdim=True) + self.epsilon
                )
        elif self.norm_p == "l1":
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (
                    grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon
                )
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (
                    grad.abs().max(-1, keepdim=True)[0] + self.epsilon
                )
        return direction, eff_direction

    def forward(
        self,
        model,#BertForSequenceClassification
        logits,
        input_ids,
        token_type_ids,
        attention_mask,
        premise_mask=None,
        hyp_mask=None,
        task_id=0,
        pairwise=1,
    ):
        # adv training
        # vat_args = [
        #     input_ids,
        #     token_type_ids,
        #     attention_mask,
        #     premise_mask,
        #     hyp_mask,
        #     task_id,
        #     1,
        # ]

        # init delta
        # embed = model(*vat_args)
        embed = self.embed_encode(model,input_ids,token_type_ids)
        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)
        for step in range(0, self.K):
            # vat_args = [
            #     input_ids,
            #     token_type_ids,
            #     attention_mask,
            #     premise_mask,
            #     hyp_mask,
            #     task_id,
            #     2,
            #     embed + noise,
            # ]
            # adv_logits = model(*vat_args)
            adv_logits=self.encode(model,None,token_type_ids,attention_mask,embed+noise)
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            (delta_grad,) = torch.autograd.grad(
                adv_loss, noise, only_inputs=True, retain_graph=False
            )
            norm = delta_grad.norm() #delta_grad会改变吗？
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            noise, eff_noise = self._norm_grad(
                delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level
            )
            noise = noise.detach()
            noise.requires_grad_()
        # vat_args = [
        #     input_ids,
        #     token_type_ids,
        #     attention_mask,
        #     premise_mask,
        #     hyp_mask,
        #     task_id,
        #     2,
        #     embed + noise,
        # ]
        # adv_logits = model(*vat_args)
        adv_logits=self.encode(model,None,token_type_ids,attention_mask,embed+noise)
        # adv_lc = self.loss_map[task_id]
        adv_lc=SymKlCriterion()
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        return adv_loss, embed.detach().abs().mean(), eff_noise.detach().abs().mean()

    def embed_encode(self,model, input_ids, token_type_ids=None):
        #model:BertForSequenceClassification
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = model.bert.embeddings(input_ids, token_type_ids)
        return embedding_output

    def encode(
        self,
        model,
        input_ids,
        token_type_ids,
        attention_mask,
        inputs_embeds=None,
    ):
        outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )#todo:model的返回类型
        # last_hidden_state = outputs.last_hidden_state
        # all_hidden_states = outputs.hidden_states  # num_layers + 1 (embeddings)
        # return last_hidden_state, all_hidden_states
        return outputs.logits

class Criterion(_Loss):
    def __init__(self, alpha=1.0, name="criterion"):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        return

class SymKlCriterion(Criterion):
    def __init__(self, alpha=1.0, name="KL Div Criterion"):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(
        self, input, target, weight=None, ignore_index=-1, reduction="batchmean"
    ):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = F.kl_div(
            F.log_softmax(input, dim=-1, dtype=torch.float32),
            F.softmax(target.detach(), dim=-1, dtype=torch.float32),
            reduction=reduction,
        ) + F.kl_div(
            F.log_softmax(target, dim=-1, dtype=torch.float32),
            F.softmax(input.detach(), dim=-1, dtype=torch.float32),
            reduction=reduction,
        )
        loss = loss * self.alpha
        return loss