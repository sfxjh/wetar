import torch
import numpy as np

def norm_samples_expander(input_embeds, batch, sample_norm, sample_num, is_ball='False'):
    input_size = np.array(input_embeds.size())
    # if sample_num > 1:
    #     input_size[0] = input_size[0] * sample_num
    input_size = tuple(input_size)
    dim = input_size[-1]
    if is_ball == 'False':
        samples = torch.normal(0, sample_norm / np.sqrt(dim), size=input_size).to(input_embeds.device)
    else:
        samples = ball_like_sample(input_embeds, sample_norm)
    res = [samples]
    for i in range(1, len(batch)):
        temp_shape = [1 for _ in range(len(batch[i].shape))]
        temp_shape[0] = temp_shape[0] * sample_num
        res.append(batch[i].repeat(temp_shape))
    samples = samples * (res[1].unsqueeze(-1).expand_as(samples))
    res[0] = samples.detach()
    return (res[0], res[1], res[2], res[3])


def input_expander(input_embeds, sample_num):
    temp_shape = [1 for _ in range(len(input_embeds.shape))]
    temp_shape[0] = temp_shape[0] * sample_num
    sample_input_embeds = input_embeds.repeat(temp_shape)
    return sample_input_embeds


def norm_samples(input_embeds, sample_norm, is_ball='False'):
    input_size = np.array(input_embeds.size())
    dim = input_size[-1]
    input_size = tuple(input_size)
    if is_ball == 'False':
        samples = torch.normal(0, sample_norm / np.sqrt(dim), size=input_size).to(input_embeds.device)
    else:
        samples = ball_like_sample(input_embeds, sample_norm)
    return samples.detach()


def ball_like_sample(input_embeds, sample_norm):
    samples = torch.randn_like(input_embeds)
    samples_norm = torch.norm(samples.view(samples.size(0), -1), dim=-1).view(-1, 1, 1)
    samples = samples / samples_norm
    samples = samples * sample_norm
    return samples

# def uniform
# def adt_loss(model,
#              x_natural,
#              y,
#              optimizer,
#              learning_rate=1.0,
#              epsilon=8.0/255.0,
#              perturb_steps=10,
#              num_samples=10,
#              lbd=0.0):

#     model.eval()
#     batch_size = len(x_natural)
#     # generate adversarial example
#     mean = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)
#     var = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)
#     optimizer_adv = optim.Adam([mean, var], lr=learning_rate, betas=(0.0, 0.0))

#     for _ in range(perturb_steps):
#         for s in range(num_samples):
#             adv_std = F.softplus(var)
#             rand_noise = torch.randn_like(x_natural)
#             adv = torch.tanh(mean + rand_noise * adv_std)
#             # omit the constants in -logp
#             negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
#             entropy = negative_logp.mean() # entropy
#             x_adv = torch.clamp(x_natural + epsilon * adv, 0.0, 1.0)

#             # minimize the negative loss
#             with torch.enable_grad():
#                 loss = -F.cross_entropy(model(x_adv), y) - lbd * entropy
#             loss.backward(retain_graph=True if s != num_samples - 1 else False)

#         optimizer_adv.step()
    
#     x_adv = torch.clamp(x_natural + epsilon * torch.tanh(mean + F.softplus(var) * torch.randn_like(x_natural)), 0.0, 1.0)
#     model.train()
#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     # zero gradient
#     optimizer.zero_grad()
#     # calculate robust loss
#     logits = model(x_adv)
#     loss = F.cross_entropy(logits, y)
#     return loss