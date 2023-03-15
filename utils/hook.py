from torch import nn as nn

class EmbeddingHook:
    forward_value = None
    backward_gradient = None

    @classmethod
    def fw_hook_layers(cls, module, inputs, outputs):
        cls.forward_value = outputs

    @classmethod
    def bw_hook_layers(cls, module, grad_in, grad_out):
        cls.backward_gradient = grad_out[0]

    @classmethod
    def register_embedding_hook(cls, embedding: nn.Embedding):
        fw_hook = embedding.register_forward_hook(cls.fw_hook_layers)
        bw_hook = embedding.register_backward_hook(cls.bw_hook_layers)
        return [fw_hook, bw_hook]

    @classmethod
    def reading_embedding_hook(cls):
        return cls.forward_value, cls.backward_gradient


class AttentionHook:
    forward_value = None
    backward_gradient = None

    @classmethod
    def fw_hook_layers(cls, module, inputs, outputs):
        cls.forward_value = outputs

    @classmethod
    def bw_hook_layers(cls, module, grad_in, grad_out):
        cls.backward_gradient = grad_out[0]

    @classmethod
    def register_attention_hook(cls, value):
        fw_hook = value.register_forward_hook(cls.fw_hook_layers)
        bw_hook = value.register_backward_hook(cls.bw_hook_layers)
        return [fw_hook, bw_hook]

    @classmethod
    def reading_value_hook(cls):
        return cls.forward_value, cls.backward_gradient


class ModuleHook:
    layers = 12
    forward_value = [[] for _ in range(layers)]
    count_layer = 0

    @classmethod
    def fw_hook_layers(cls, module, inputs, outputs):
        cls.forward_value[cls.count_layer].append(outputs[0])
        cls.count_layer = (cls.count_layer + 1) % 12

    @classmethod
    def hook_clean_up(cls):
        cls.forward_value = [[] for _ in range(cls.layers)]
        cls.count_layer = 0

def bert_model_hooker(model: nn.Module, layers=12):
    from utils.hook import ModuleHook
    nums_str = [str(i) for i in range(layers)]
    module_name = "bert.encoder.layer"
    layers_str = [module_name + '.' + nums_str[i] for i in range(layers)]
    hooks = {}
    for name, module in model.named_modules():
        if name in layers_str:
            # print(name)
            hooks[name] = module.register_forward_hook(ModuleHook.fw_hook_layers)
    return hooks