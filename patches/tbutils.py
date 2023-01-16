import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules import shared


def tensorboard_setup(log_directory):
    os.makedirs(os.path.join(log_directory, "tensorboard"), exist_ok=True)
    return SummaryWriter(
        log_dir=os.path.join(log_directory, "tensorboard"),
        flush_secs=shared.opts.training_tensorboard_flush_every)

def tensorboard_log_hyperparameter(tensorboard_writer:SummaryWriter, **kwargs):
    for keys in kwargs:
        if type(kwargs[keys]) not in [bool, str, float, int,None]:
            kwargs[keys] = str(kwargs[keys])
    tensorboard_writer.add_hparams({
        'lr' : kwargs.get('lr', 0.01),
        'GA steps' : kwargs.get('GA_steps', 1),
        'bsize' : kwargs.get('batch_size', 1),
        'layer structure' : kwargs.get('layer_structure', '1,2,1'),
        'activation' : kwargs.get('activation', 'Linear'),
        'weight_init' : kwargs.get('weight_init', 'Normal'),
        'dropout_structure' : kwargs.get('dropout_structure', '0,0,0'),
        'steps' : kwargs.get('max_steps', 10000),
        'latent sampling': kwargs.get('latent_sampling_method', 'once'),
        'template file': kwargs.get('template', 'nothing'),
        'CosineAnnealing' : kwargs.get('CosineAnnealing', False),
        'beta_repeat epoch': kwargs.get('beta_repeat_epoch', 0),
        'epoch_mult':kwargs.get('epoch_mult', 1),
        'warmup_step' : kwargs.get('warmup', 5),
        'min_lr' : kwargs.get('min_lr', 6e-7),
        'decay' : kwargs.get('gamma_rate', 1),
        'adamW' : kwargs.get('adamW_opts', False),
        'adamW_decay' : kwargs.get('adamW_decay', 0.01),
        'adamW_beta1' : kwargs.get('adamW_beta_1', 0.9),
        'adamW_beta2': kwargs.get('adamW_beta_2', 0.99),
        'adamW_eps': kwargs.get('adamW_eps', 1e-8),
        'gradient_clip_opt':kwargs.get('gradient_clip', 'None'),
        'gradient_clip_value' : kwargs.get('gradient_clip_value', 1e-1),
        'gradient_clip_norm' : kwargs.get('gradient_clip_norm_type', 2)
        },
        {'hparam/loss' : kwargs.get('loss', 0.0)}
    )
def tensorboard_add(tensorboard_writer:SummaryWriter, loss, global_step, step, learn_rate, epoch_num, base_name=""):
    prefix = base_name + "/" if base_name else ""
    tensorboard_add_scaler(tensorboard_writer, prefix+"Loss/train", loss, global_step)
    tensorboard_add_scaler(tensorboard_writer, prefix+f"Loss/train/epoch-{epoch_num}", loss, step)
    tensorboard_add_scaler(tensorboard_writer, prefix+"Learn rate/train", learn_rate, global_step)
    tensorboard_add_scaler(tensorboard_writer, prefix+f"Learn rate/train/epoch-{epoch_num}", learn_rate, step)


def tensorboard_add_scaler(tensorboard_writer:SummaryWriter, tag, value, step):
    tensorboard_writer.add_scalar(tag=tag,
                                  scalar_value=value, global_step=step)


def tensorboard_add_image(tensorboard_writer:SummaryWriter, tag, pil_image, step, base_name=""):
    # Convert a pil image to a torch tensor
    prefix = base_name + "/" if base_name else ""
    img_tensor = torch.as_tensor(np.array(pil_image, copy=True))
    img_tensor = img_tensor.view(pil_image.size[1], pil_image.size[0],
                                 len(pil_image.getbands()))
    img_tensor = img_tensor.permute((2, 0, 1))

    tensorboard_writer.add_image(prefix+tag, img_tensor, global_step=step)
