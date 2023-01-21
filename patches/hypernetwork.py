import glob
import inspect
import os
import sys
import traceback

import torch
from torch.nn.init import normal_, xavier_uniform_, zeros_, xavier_normal_, kaiming_uniform_, kaiming_normal_

import scripts.xy_grid
from modules.shared import opts
try:
    from modules.hashes import sha256
except (ImportError, ModuleNotFoundError):
    print("modules.hashes is not found, will use backup module from extension!")
    from .hashes_backup import sha256

import modules.hypernetworks.hypernetwork
from modules import devices, shared, sd_models, processing, generation_parameters_copypaste
from .hnutil import parse_dropout_structure, find_self
from .shared import version_flag

def init_weight(layer, weight_init="Normal", normal_std=0.01, activation_func="relu"):
    w, b = layer.weight.data, layer.bias.data
    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
        normal_(w, mean=0.0, std=normal_std)
        normal_(b, mean=0.0, std=0)
    elif weight_init == 'XavierUniform':
        xavier_uniform_(w)
        zeros_(b)
    elif weight_init == 'XavierNormal':
        xavier_normal_(w)
        zeros_(b)
    elif weight_init == 'KaimingUniform':
        kaiming_uniform_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
        zeros_(b)
    elif weight_init == 'KaimingNormal':
        kaiming_normal_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
        zeros_(b)
    else:
        raise KeyError(f"Key {weight_init} is not defined as initialization!")


class ResBlock(torch.nn.Module):
    """Residual Block"""
    def __init__(self, n_inputs, n_outputs, activation_func, weight_init, add_layer_norm, dropout_p, normal_std, device=None, state_dict=None, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs
        self.upsample_layer = None
        self.upsample = kwargs.get("upsample_model", None)
        if self.upsample == "Linear":
            self.upsample_layer = torch.nn.Linear(n_inputs, n_outputs, bias=False)
        linears = [torch.nn.Linear(n_inputs, n_outputs)]
        init_weight(linears[0], weight_init, normal_std, activation_func)
        if add_layer_norm:
            linears.append(torch.nn.LayerNorm(n_outputs))
            init_weight(linears[1], weight_init, normal_std, activation_func)
        if dropout_p > 0:
            linears.append(torch.nn.Dropout(p=dropout_p))
        if activation_func == "linear" or activation_func is None:
            pass
        elif activation_func in HypernetworkModule.activation_dict:
            linears.append(HypernetworkModule.activation_dict[activation_func]())
        else:
            raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')
        self.linear = torch.nn.Sequential(*linears)
        if state_dict is not None:
            self.load_state_dict(state_dict)
        if device is not None:
            self.to(device)

    def trainables(self, train=False):
        layer_structure = []
        for layer in self.linear:
            if train:
                layer.train()
            else:
                layer.eval()
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                layer_structure += [layer.weight, layer.bias]
        return layer_structure

    def forward(self, x, **kwargs):
        if self.upsample_layer is None:
            interpolated = torch.nn.functional.interpolate(x, size=self.n_outputs, mode="nearest-exact")
        else:
            interpolated = self.upsample_layer(x)
        return interpolated + self.linear(x)



class HypernetworkModule(torch.nn.Module):
    multiplier = 1.0
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal',
                 add_layer_norm=False, activate_output=False, dropout_structure=None, device=None, generation_seed=None, normal_std=0.01, **kwargs):
        super().__init__()
        self.skip_connection = skip_connection = kwargs.get('skip_connection', False)
        upsample_linear = kwargs.get('upsample_linear', None)
        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"
        assert skip_connection or dropout_structure is None or dropout_structure[0] == dropout_structure[-1] == 0, "Dropout Sequence should start and end with probability 0!"
        assert dropout_structure is None or len(dropout_structure) == len(layer_structure), "Dropout Sequence should match length with layer structure!"

        linears = []
        if skip_connection:
            if generation_seed is not None:
                torch.manual_seed(generation_seed)
        for i in range(len(layer_structure) - 1):
            if skip_connection:
                n_inputs, n_outputs = int(dim * layer_structure[i]), int(dim * layer_structure[i+1])
                dropout_p = dropout_structure[i+1]
                if activation_func is None:
                    activation_func = "linear"
                linears.append(ResBlock(n_inputs, n_outputs, activation_func, weight_init, add_layer_norm, dropout_p, normal_std, device, upsample_model=upsample_linear))
                continue

            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func except last layer
            if activation_func == "linear" or activation_func is None or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Everything should be now parsed into dropout structure, and applied here.
            # Since we only have dropouts after layers, dropout structure should start with 0 and end with 0.
            if dropout_structure is not None and dropout_structure[i+1] > 0:
                assert 0 < dropout_structure[i+1] < 1, "Dropout probability should be 0 or float between 0 and 1!"
                linears.append(torch.nn.Dropout(p=dropout_structure[i+1]))
            # Code explanation : [1, 2, 1] -> dropout is missing when last_layer_dropout is false. [1, 2, 2, 1] -> [0, 0.3, 0, 0], when its True, [0, 0.3, 0.3, 0].

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        elif not skip_connection:
            if generation_seed is not None:
                torch.manual_seed(generation_seed)
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    w, b = layer.weight.data, layer.bias.data
                    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
                        normal_(w, mean=0.0, std=normal_std)
                        normal_(b, mean=0.0, std=0)
                    elif weight_init == 'XavierUniform':
                        xavier_uniform_(w)
                        zeros_(b)
                    elif weight_init == 'XavierNormal':
                        xavier_normal_(w)
                        zeros_(b)
                    elif weight_init == 'KaimingUniform':
                        kaiming_uniform_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    elif weight_init == 'KaimingNormal':
                        kaiming_normal_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    else:
                        raise KeyError(f"Key {weight_init} is not defined as initialization!")
        if device is None:
            self.to(devices.device)
        else:
            self.to(device)


    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x, multiplier=None):
        if self.skip_connection:
            if self.training:
                return self.linear(x)
            else:
                resnet_result = self.linear(x)
                residual = resnet_result - x
                if multiplier is None or not isinstance(multiplier, (int, float)):
                    multiplier = self.multiplier if not version_flag else HypernetworkModule.multiplier
                return x + multiplier * residual  # interpolate
        if multiplier is None or not isinstance(multiplier, (int, float)):
            return x + self.linear(x) * ((self.multiplier if not version_flag else HypernetworkModule.multiplier) if not self.training else 1)
        return x + self.linear(x) * multiplier

    def trainables(self, train=False):
        layer_structure = []
        self.train(train)
        for layer in self.linear:
            if train:
                layer.train()
            else:
                layer.eval()
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                layer_structure += [layer.weight, layer.bias]
            elif type(layer) == ResBlock:
                layer_structure += layer.trainables(train)
        return layer_structure

    def set_train(self,mode=True):
        self.train(mode)
        for layer in self.linear:
            if mode:
                layer.train(mode)
            else:
                layer.eval()


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, activate_output=False, **kwargs):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure
        self.activation_func = activation_func
        self.weight_init = weight_init
        self.add_layer_norm = add_layer_norm
        self.use_dropout = use_dropout
        self.activate_output = activate_output
        self.last_layer_dropout = kwargs['last_layer_dropout'] if 'last_layer_dropout' in kwargs else True
        self.optimizer_name = None
        self.optimizer_state_dict = None
        self.dropout_structure = kwargs['dropout_structure'] if 'dropout_structure' in kwargs and use_dropout else None
        self.optional_info = kwargs.get('optional_info', None)
        self.skip_connection = kwargs.get('skip_connection', False)
        self.upsample_linear = kwargs.get('upsample_linear', None)
        self.training = False
        generation_seed = kwargs.get('generation_seed', None)
        normal_std = kwargs.get('normal_std', 0.01)
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure, generation_seed=generation_seed, normal_std=normal_std, skip_connection=self.skip_connection,
                                   upsample_linear=self.upsample_linear),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure, generation_seed=generation_seed, normal_std=normal_std, skip_connection=self.skip_connection,
                                   upsample_linear=self.upsample_linear),
            )
        self.eval()

    def weights(self, train=False):
        self.training = train
        res = []
        for k, layers in self.layers.items():
            for layer in layers:
                res += layer.trainables(train)
        return res

    def eval(self):
        self.training = False
        for k, layers in self.layers.items():
            for layer in layers:
                layer.eval()
                layer.set_train(False)

    def train(self, mode=True):
        self.training = mode
        for k, layers in self.layers.items():
            for layer in layers:
                layer.set_train(mode)

    def detach_grad(self):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.requires_grad_(False)

    def shorthash(self):
        sha256v = sha256(self.filename, f'hypernet/{self.name}')
        return sha256v[0:10]

    def extra_name(self):
        if version_flag:
            return ""
        found = find_self(self)
        if found is not None:
            return f" <hypernet:{found}:1.0>"
        return f" <hypernet:{self.name}:1.0>"

    def save(self, filename):
        state_dict = {}
        optimizer_saved_dict = {}

        for k, v in self.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['layer_structure'] = self.layer_structure
        state_dict['activation_func'] = self.activation_func
        state_dict['is_layer_norm'] = self.add_layer_norm
        state_dict['weight_initialization'] = self.weight_init
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name
        state_dict['activate_output'] = self.activate_output
        state_dict['use_dropout'] = self.use_dropout
        state_dict['dropout_structure'] = self.dropout_structure
        state_dict['last_layer_dropout'] = (self.dropout_structure[-2] != 0) if self.dropout_structure is not None else self.last_layer_dropout
        state_dict['optional_info'] = self.optional_info if self.optional_info else None
        state_dict['skip_connection'] = self.skip_connection
        state_dict['upsample_linear'] = self.upsample_linear

        if self.optimizer_name is not None:
            optimizer_saved_dict['optimizer_name'] = self.optimizer_name

        torch.save(state_dict, filename)
        if shared.opts.save_optimizer_state and self.optimizer_state_dict:
            optimizer_saved_dict['hash'] = self.shorthash()
            optimizer_saved_dict['optimizer_state_dict'] = self.optimizer_state_dict
            torch.save(optimizer_saved_dict, filename + '.optim')

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        print(self.layer_structure)
        optional_info = state_dict.get('optional_info', None)
        if optional_info is not None:
            self.optional_info = optional_info
        self.activation_func = state_dict.get('activation_func', None)
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        self.dropout_structure = state_dict.get('dropout_structure', None)
        self.use_dropout = True if self.dropout_structure is not None and any(self.dropout_structure) else state_dict.get('use_dropout', False)
        self.activate_output = state_dict.get('activate_output', True)
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)  # Silent fix for HNs before 4918eb6
        self.skip_connection = state_dict.get('skip_connection', False)
        self.upsample_linear = state_dict.get('upsample_linear', False)
        # Dropout structure should have same length as layer structure, Every digits should be in [0,1), and last digit must be 0.
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)
        if hasattr(shared.opts, 'print_hypernet_extra') and shared.opts.print_hypernet_extra:
            if optional_info is not None:
                print(f"INFO:\n {optional_info}\n")
            print(f"Activation function is {self.activation_func}")
            print(f"Weight initialization is {self.weight_init}")
            print(f"Layer norm is set to {self.add_layer_norm}")
            print(f"Dropout usage is set to {self.use_dropout}")
            print(f"Activate last layer is set to {self.activate_output}")
            print(f"Dropout structure is set to {self.dropout_structure}")
        optimizer_saved_dict = torch.load(self.filename + '.optim', map_location = 'cpu') if os.path.exists(self.filename + '.optim') else {}
        self.optimizer_name = "AdamW"

        if optimizer_saved_dict.get('hash', None) == self.shorthash() or optimizer_saved_dict.get('hash', None) == sd_models.model_hash(filename):
            self.optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
        else:
            self.optimizer_state_dict = None
        if self.optimizer_state_dict:
            self.optimizer_name = optimizer_saved_dict.get('optimizer_name', 'AdamW')
            print("Loaded existing optimizer from checkpoint")
            print(f"Optimizer name is {self.optimizer_name}")
        else:
            print("No saved optimizer exists in checkpoint")

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, sd[0], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure, skip_connection=self.skip_connection, upsample_linear=self.upsample_linear),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure, skip_connection=self.skip_connection, upsample_linear=self.upsample_linear),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)
        self.eval()

    def to(self, device):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.to(device)

        return self

    def set_multiplier(self, multiplier):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.multiplier = multiplier

        return self

    def __call__(self, context, *args, **kwargs):
        return self.forward(context, *args, **kwargs)

    def forward(self, context, context_v=None, layer=None):
        context_layers = self.layers.get(context.shape[2], None)
        if context_v is None:
            context_v = context
        if context_layers is None:
            return context, context_v
        if layer is not None and hasattr(layer, 'hyper_k') and hasattr(layer, 'hyper_v'):
            layer.hyper_k = context_layers[0]
            layer.hyper_v = context_layers[1]
        transform_k, transform_v = context_layers[0](context), context_layers[1](context_v)
        return transform_k, transform_v


def list_hypernetworks(path):
    res = {}
    for filename in sorted(glob.iglob(os.path.join(path, '**/*.pt'), recursive=True)):
        name = os.path.splitext(os.path.basename(filename))[0]
        idx = 0
        while name in res:
            idx += 1
            name = name + f"({idx})"
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name] = filename
    for filename in glob.iglob(os.path.join(path, '**/*.hns'), recursive=True):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name != "None":
            res[name] = filename
    return res

def find_closest_first(keyset, target):
    for keys in keyset:
        if target == keys.rsplit('(', 1)[0]:
            return keys
    return None



def load_hypernetwork(filename):
    hypernetwork = None
    path = shared.hypernetworks.get(filename, None)
    if path is None:
        filename = find_closest_first(shared.hypernetworks.keys(), filename)
        path = shared.hypernetworks.get(filename, None)
    print(path)
    # Prevent any file named "None.pt" from being loaded.
    if path is not None and filename != "None":
        print(f"Loading hypernetwork {filename}")
        if path.endswith(".pt"):
            try:
                hypernetwork = Hypernetwork()
                hypernetwork.load(path)
                if hasattr(shared, 'loaded_hypernetwork'):
                    shared.loaded_hypernetwork = hypernetwork
                else:
                    return hypernetwork

            except Exception:
                print(f"Error loading hypernetwork {path}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        elif path.endswith(".hns"):
            # Load Hypernetwork processing
            try:
                from .hypernetworks import load as load_hns
                if hasattr(shared, 'loaded_hypernetwork'):
                    shared.loaded_hypernetwork = load_hns(path)
                else:
                    hypernetwork = load_hns(path)
                    print(f"Loaded Hypernetwork Structure {path}")
                    return hypernetwork
            except Exception:
                print(f"Error loading hypernetwork processing file {path}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        else:
            print(f"Tried to load unknown file extension: {filename}")
    else:
        if hasattr(shared, 'loaded_hypernetwork'):
            if shared.loaded_hypernetwork is not None:
                print(f"Unloading hypernetwork")
                shared.loaded_hypernetwork = None
    return hypernetwork


def apply_hypernetwork(hypernetwork, context, layer=None):
    if hypernetwork is None:
        return context, context
    if isinstance(hypernetwork, Hypernetwork):
        hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)
        if hypernetwork_layers is None:
            return context, context
        if layer is not None:
            layer.hyper_k = hypernetwork_layers[0]
            layer.hyper_v = hypernetwork_layers[1]

        context_k = hypernetwork_layers[0](context)
        context_v = hypernetwork_layers[1](context)
        return context_k, context_v
    context_k, context_v = hypernetwork(context, layer=layer)
    return context_k, context_v


def apply_single_hypernetwork(hypernetwork, context_k, context_v, layer=None):
    if hypernetwork is None:
        return context_k, context_v
    if isinstance(hypernetwork, Hypernetwork):
        hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context_k.shape[2], None)
        if hypernetwork_layers is None:
            return context_k, context_v
        if layer is not None:
            layer.hyper_k = hypernetwork_layers[0]
            layer.hyper_v = hypernetwork_layers[1]

        context_k = hypernetwork_layers[0](context_k)
        context_v = hypernetwork_layers[1](context_v)
        return context_k, context_v
    context_k, context_v = hypernetwork(context_k, context_v, layer=layer)
    return context_k, context_v


def apply_strength(value=None):
    HypernetworkModule.multiplier = value if value is not None else shared.opts.sd_hypernetwork_strength


def apply_hypernetwork_strength(p, x, xs):
    apply_strength(x)


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0):
    index = position_in_batch + iteration * p.batch_size

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)

    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Seed": all_seeds[index],
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        "Model": (None if not opts.add_model_name_to_info or not shared.sd_model.sd_checkpoint_info.model_name else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
        "Batch size": (None if p.batch_size < 2 else p.batch_size),
        "Batch pos": (None if p.batch_size < 2 else position_in_batch),
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Conditional mask weight": getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        "Eta": (None if p.sampler is None or p.sampler.eta == p.sampler.default_eta else p.sampler.eta),
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta,
    }

    generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: " + p.all_negative_prompts[index] if p.all_negative_prompts[index] else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip()


modules.hypernetworks.hypernetwork.list_hypernetworks = list_hypernetworks
modules.hypernetworks.hypernetwork.load_hypernetwork = load_hypernetwork
if hasattr(modules.hypernetworks.hypernetwork, 'apply_hypernetwork'):
    modules.hypernetworks.hypernetwork.apply_hypernetwork = apply_hypernetwork
else:
    modules.hypernetworks.hypernetwork.apply_single_hypernetwork = apply_single_hypernetwork
if hasattr(modules.hypernetworks.hypernetwork, 'apply_strength'):
    modules.hypernetworks.hypernetwork.apply_strength = apply_strength
modules.hypernetworks.hypernetwork.Hypernetwork = Hypernetwork
modules.hypernetworks.hypernetwork.HypernetworkModule = HypernetworkModule
if hasattr(scripts.xy_grid, 'apply_hypernetwork_strength'):
    scripts.xy_grid.apply_hypernetwork_strength = apply_hypernetwork_strength

# Fix calculating hash for multiple hns
processing.create_infotext = create_infotext