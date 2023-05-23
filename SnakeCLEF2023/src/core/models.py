import sys
import torch.nn as nn
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoConfig, AutoModelForImageClassification

class Transformer(nn.Module):
    def __init__(self, pretrained_checkpoint, pretrained=False):
        
        super().__init__()
        self.pretrained_checkpoint = pretrained_checkpoint
        if pretrained:
            self.net = AutoModelForImageClassification.from_pretrained(pretrained_checkpoint)
        else:
            _config = AutoConfig.from_pretrained(pretrained_checkpoint)
            self.net = AutoModelForImageClassification.from_config(_config)
        config = self.net.config
        self.default_cfg = {
            'num_classes': self.net.num_labels,
            'input_size': (config.num_channels, config.image_size, config.image_size),
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
            'classifier': 'classifier'}

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def __setattr__(self, name, value):
        
        if name == 'classifier':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    @property
    def classifier(self):
        if hasattr(self.net, 'classifier'):
            out = self.net.classifier
        elif hasattr(self.net, 'cls_classifier') and hasattr(self.net, 'distillation_classifier'):
            out = self.net.cls_classifier
        else:
            raise ValueError(
                f'Alien classifier in the Vision Transformer: {self.pretrained_checkpoint}')
        return out

    @classifier.setter
    def classifier(self, val):
        if hasattr(self.net, 'classifier'):
            self.net.classifier = val
        elif hasattr(self.net, 'cls_classifier') and hasattr(self.net, 'distillation_classifier'):
            self.net.cls_classifier = val
            self.net.distillation_classifier = val
        else:
            raise ValueError(
                f'Alien classifier in the Vision Transformer: {self.pretrained_checkpoint}')

    def forward(self, x):
        return self.net(x).logits


class _FunctionWrapper:
    def __init__(self, func, func_name):
        self.func = func
        self.func.__name__ = func_name
        self.__name__ = func_name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.func.__name__

    def __repr__(self):
        return str(self)


def model_factory(arch_name, pretrained_checkpoint, use_hf=False):
    def create_model(out_dim=None, pretrained=False, *args, **kwargs):
        if use_hf:
            model = Transformer(pretrained_checkpoint, pretrained=pretrained)
        else:
            model = timm.create_model(pretrained_checkpoint, pretrained=pretrained, *args, **kwargs)

        input_size = model.default_cfg.get('test_input_size', model.default_cfg['input_size'])
        model.pretrained_config = {
            'pretrained_checkpoint': pretrained_checkpoint,
            'input_size': input_size[-1],
            'image_mean': model.default_cfg['mean'],
            'image_std': model.default_cfg['std']}

        # set custom classification head
        if out_dim is not None:
            classifier_attr = model.default_cfg['classifier']
            # some models like distilled transformers have multiple heads
            if isinstance(classifier_attr, str):
                classifier_attr = (classifier_attr,)
            for attr in classifier_attr:
                in_features = getattr(model, attr).in_features
                setattr(model, attr, nn.Linear(in_features, out_dim))

        return model
    return _FunctionWrapper(create_model, arch_name)


_timm_model_specs = {

    'efficientnet_b0': 'tf_efficientnet_b0',
    'efficientnet_b1': 'tf_efficientnet_b1',
    'vit_small_384': 'vit_small_patch16_384',
    'deit_base_distilled_384': 'deit_base_distilled_patch16_384',
    'deit_base_384': 'deit_base_patch16_384',

}


_hf_model_specs = {

    'vit_base_224': 'google/vit-base-patch16-224',
    'vit_base_384': 'google/vit-base-patch16-384',
    'vit_base_patch32_224': 'google/vit-base-patch32-224',
    'vit_base_patch32_384': 'google/vit-base-patch32-384',
    'vit_large_224': 'google/vit-large-patch16-224',
    'vit_large_384': 'google/vit-large-patch16-384',
    'deit_base_distilled_384': 'facebook/deit-base-distilled-patch16-384',

}


# create dictionary with models
MODELS = {
    **{k: model_factory(k, v) for k, v in _timm_model_specs.items()},
    **{k: model_factory(k, v, use_hf=True) for k, v in _hf_model_specs.items()}}


# add models as attribute of this module
for k, v in MODELS.items():
    setattr(sys.modules[__name__], k, v)


def get_model_wrapper(model_arch):
    if callable(model_arch):
        model_function = model_arch
    elif model_arch in MODELS:
        model_function = MODELS[model_arch]
    else:
        raise ValueError(f'Alien model architecture "{model_arch}".')
    return model_function


def get_model(model_arch, out_dim=None, pretrained=True, *args, **kwargs):
    model_function = get_model_wrapper(model_arch)
    model = model_function(out_dim, pretrained, *args, **kwargs)
    return model


