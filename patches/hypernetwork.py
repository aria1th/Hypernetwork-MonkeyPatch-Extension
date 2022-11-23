import datetime
import glob
import html
import inspect
import os
import sys
import traceback
from collections import defaultdict, deque
from statistics import stdev, mean

import torch
import tqdm
from torch.nn.init import normal_, xavier_uniform_, zeros_, xavier_normal_, kaiming_uniform_, kaiming_normal_
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

import modules.hypernetworks.hypernetwork
from modules import devices, shared, sd_models, processing, sd_samplers
from .hnutil import parse_dropout_structure, optim_to
from modules.hypernetworks.hypernetwork import report_statistics, save_hypernetwork, stack_conds, optimizer_dict
from modules.textual_inversion import textual_inversion
from .dataset import PersonalizedBase
from modules.textual_inversion.learn_schedule import LearnRateScheduler


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
                 add_layer_norm=False, activate_output=False, dropout_structure=None, device=None):
        super().__init__()

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"
        assert dropout_structure is None or dropout_structure[0] == dropout_structure[-1] == 0, "Dropout Sequence should start and end with probability 0!"
        assert dropout_structure is None or len(dropout_structure) == len(layer_structure), "Dropout Sequence should match length with layer structure!"

        linears = []
        for i in range(len(layer_structure) - 1):

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
            if dropout_structure is not None and (p := dropout_structure[i+1]) > 0:
                assert 0 < p < 1, "Dropout probability should be 0 or float between 0 and 1!"
                linears.append(torch.nn.Dropout(p=p))
            # Code explanation : [1, 2, 1] -> dropout is missing when last_layer_dropout is false. [1, 2, 2, 1] -> [0, 0.3, 0, 0], when its True, [0, 0.3, 0.3, 0].

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            #torch.manual_seed(42) # fix seed.
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    w, b = layer.weight.data, layer.bias.data
                    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
                        normal_(w, mean=0.0, std=0.01)
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
        if multiplier is None or not isinstance(multiplier, (int, float)):
            return x + self.linear(x) * self.multiplier
        return x + self.linear(x) * multiplier

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
        self.dropout_structure = kwargs['dropout_structure'] if 'dropout_structure' in kwargs else None
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
            )
        self.eval()

    def weights(self, train=False):
        res = []
        for k, layers in self.layers.items():
            for layer in layers:
                res += layer.trainables(train)
        return res

    def eval(self):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.eval()


    def train(self):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.train()


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
        state_dict['dropout_structure'] = self.dropout_structure

        if self.optimizer_name is not None:
            optimizer_saved_dict['optimizer_name'] = self.optimizer_name

        torch.save(state_dict, filename)
        if shared.opts.save_optimizer_state and self.optimizer_state_dict:
            optimizer_saved_dict['hash'] = sd_models.model_hash(filename)
            optimizer_saved_dict['optimizer_state_dict'] = self.optimizer_state_dict
            torch.save(optimizer_saved_dict, filename + '.optim')

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        print(self.layer_structure)
        self.activation_func = state_dict.get('activation_func', None)
        print(f"Activation function is {self.activation_func}")
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        print(f"Weight initialization is {self.weight_init}")
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        print(f"Layer norm is set to {self.add_layer_norm}")
        self.use_dropout = state_dict.get('use_dropout', False)
        print(f"Dropout usage is set to {self.use_dropout}" )
        self.activate_output = state_dict.get('activate_output', True)
        print(f"Activate last layer is set to {self.activate_output}")
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)  # Silent fix for HNs before 4918eb6
        # Dropout structure should have same length as layer structure, Every digits should be in [0,1), and last digit must be 0.
        self.dropout_structure = state_dict.get('dropout_structure', None)
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)
        print(f"Dropout structure is set to {self.dropout_structure}")

        optimizer_saved_dict = torch.load(self.filename + '.optim', map_location = 'cpu') if os.path.exists(self.filename + '.optim') else {}
        self.optimizer_name = "AdamW"

        if sd_models.model_hash(filename) == optimizer_saved_dict.get('hash', None):
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
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)

    def to(self, device):
        for values in self.layers.values():
            values[0].to(device)
            values[1].to(device)

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
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name+ f"({sd_models.model_hash(filename)})"] = filename
    for filename in glob.iglob(os.path.join(path, '**/*.hns'), recursive=True):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name != "None":
            res[name] = filename
    return res


def load_hypernetwork(filename):
    path = shared.hypernetworks.get(filename, None)
    print(path)
    # Prevent any file named "None.pt" from being loaded.
    if path is not None and filename != "None":
        print(f"Loading hypernetwork {filename}")
        if path.endswith(".pt"):
            try:
                shared.loaded_hypernetwork = Hypernetwork()
                shared.loaded_hypernetwork.load(path)

            except Exception:
                print(f"Error loading hypernetwork {path}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        elif path.endswith(".hns"):
            # Load Hypernetwork processing
            try:
                from .hypernetworks import load as load_hns
                shared.loaded_hypernetwork = load_hns(path)
                print(f"Loaded Hypernetwork Structure {path}")
            except Exception:
                print(f"Error loading hypernetwork processing file {path}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        else:
            print(f"Tried to load unknown file extension: {filename}")
    else:
        if shared.loaded_hypernetwork is not None:
            print(f"Unloading hypernetwork")

        shared.loaded_hypernetwork = None


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


def train_hypernetwork(hypernetwork_name, learn_rate, batch_size, data_root, log_directory, training_width,
                       training_height, steps, create_image_every, save_hypernetwork_every, template_file,
                       preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps,
                       preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height,
                       use_beta_scheduler=False, beta_repeat_epoch=4000, min_lr=1e-7, gamma_rate=1):
    # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
    from modules import images
    try:
        beta_repeat_epoch = int(beta_repeat_epoch)
        assert beta_repeat_epoch > 0, f"Cannot use too small cycle {beta_repeat_epoch}!"
        min_lr = float(min_lr)
        assert min_lr < 1, f"Cannot use minimum lr with {min_lr}!"
        gamma_rate = float(gamma_rate)
        assert 0 <= gamma_rate <= 1, f"Cannot use gamma rate with {gamma_rate}!"
    except ValueError:
        raise RuntimeError("Cannot use advanced LR scheduler settings!")
    save_hypernetwork_every = save_hypernetwork_every or 0
    create_image_every = create_image_every or 0
    textual_inversion.validate_train_inputs(hypernetwork_name, learn_rate, batch_size, data_root, template_file, steps,
                                            save_hypernetwork_every, create_image_every, log_directory,
                                            name="hypernetwork")

    load_hypernetwork(hypernetwork_name)
    assert shared.loaded_hypernetwork is not None, f"Cannot load {hypernetwork_name}!"
    if not isinstance(shared.loaded_hypernetwork, Hypernetwork):
        raise RuntimeError("Cannot perform training for Hypernetwork structure pipeline!")
    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps
    losses_list = []
    hypernetwork_name = hypernetwork_name.rsplit('(', 1)[0]
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
    unload = shared.opts.unload_models_when_training

    if save_hypernetwork_every > 0:
        hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
        os.makedirs(hypernetwork_dir, exist_ok=True)
    else:
        hypernetwork_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    hypernetwork = shared.loaded_hypernetwork
    checkpoint = sd_models.select_checkpoint()

    ititial_step = hypernetwork.step or 0
    if ititial_step >= steps:
        shared.state.textinfo = f"Model has already been trained beyond specified max steps"
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)
    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = PersonalizedBase(data_root=data_root, width=training_width,
                                                                height=training_height,
                                                                repeats=shared.opts.training_image_repeats_per_epoch,
                                                                placeholder_token=hypernetwork_name,
                                                                model=shared.sd_model, device=devices.device,
                                                                template_file=template_file, include_cond=True,
                                                                batch_size=batch_size)

    if unload:
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    size = len(ds.indexes)
    loss_dict = defaultdict(lambda: deque(maxlen=1024))
    losses = torch.zeros((size,))
    previous_mean_losses = [0]
    previous_mean_loss = 0
    print("Mean loss of {} elements".format(size))

    weights = hypernetwork.weights(True)

    # Here we use optimizer from saved HN, or we can specify as UI option.
    if hypernetwork.optimizer_name in optimizer_dict:
        optimizer = optimizer_dict[hypernetwork.optimizer_name](params=weights, lr=scheduler.learn_rate)
        optimizer_name = hypernetwork.optimizer_name
    else:
        print(f"Optimizer type {hypernetwork.optimizer_name} is not defined!")
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate)
        optimizer_name = 'AdamW'

    if hypernetwork.optimizer_state_dict:  # This line must be changed if Optimizer type can be different from saved optimizer.
        try:
            optimizer.load_state_dict(hypernetwork.optimizer_state_dict)
        except RuntimeError as e:
            print("Cannot resume from saved optimizer!")
            print(e)

    scheduler_beta = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=beta_repeat_epoch, T_mult=1, eta_min=min_lr)

    scheduler_gamma = ExponentialLR(optimizer=optimizer, gamma=gamma_rate)
    steps_without_grad = 0

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"

    pbar = tqdm.tqdm(enumerate(ds), total=steps - ititial_step)
    for i, entries in pbar:
        hypernetwork.step = i + ititial_step
        if use_beta_scheduler:
            scheduler_beta.step(hypernetwork.step)
            scheduler_gamma.step(hypernetwork.step)
        if len(loss_dict) > 0:
            previous_mean_losses = [i[-1] for i in loss_dict.values()]
            previous_mean_loss = mean(previous_mean_losses)
        if not use_beta_scheduler:
            scheduler.apply(optimizer, hypernetwork.step)
        if i + ititial_step > steps:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = stack_conds([entry.cond for entry in entries]).to(devices.device)
            # c = torch.vstack([entry.cond for entry in entries]).to(devices.device)
            x = torch.stack([entry.latent for entry in entries]).to(devices.device)
            loss_infos = shared.sd_model(x, c)[1]
            loss = loss_infos[
                'val/loss_simple']  # + loss_infos['val/loss_vlb'] * 0.4 #its 'prior class preserving' loss
            del x
            del c

            losses[hypernetwork.step % losses.shape[0]] = loss.item()
            losses_list.append(loss.item())
            for entry in entries:
                loss_dict[entry.filename].append(loss.item())
            optimizer.zero_grad()
            weights[0].grad = None
            loss.backward()

            if weights[0].grad is None:
                steps_without_grad += 1
            else:
                steps_without_grad = 0
            assert steps_without_grad < 10, 'no gradient found for the trained weight after backward() for 10 steps in a row; this is a bug; training cannot continue'
            optimizer.step()

        steps_done = hypernetwork.step + 1

        if torch.isnan(losses[hypernetwork.step % losses.shape[0]]):
            raise RuntimeError("Loss diverged.")

        if len(previous_mean_losses) > 1:
            std = stdev(previous_mean_losses)
        else:
            std = 0
        dataset_loss_info = f"dataset loss:{mean(previous_mean_losses):.3f}" + u"\u00B1" + f"({std / (len(previous_mean_losses) ** 0.5):.3f})"
        pbar.set_description(dataset_loss_info)

        if hypernetwork_dir is not None and steps_done % save_hypernetwork_every == 0:
            # Before saving, change name to match current checkpoint.
            hypernetwork_name_every = f'{hypernetwork_name}-{steps_done}'
            last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork_name_every}.pt')
            hypernetwork.optimizer_name = optimizer_name
            if shared.opts.save_optimizer_state:
                hypernetwork.optimizer_state_dict = optimizer.state_dict()
            save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, last_saved_file)
            hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.

        textual_inversion.write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, len(ds), {
            "loss": f"{previous_mean_loss:.7f}",
            "learn_rate": scheduler_beta.get_last_lr()
        })

        if images_dir is not None and steps_done % create_image_every == 0:
            forced_filename = f'{hypernetwork_name}-{steps_done}'
            last_saved_image = os.path.join(images_dir, forced_filename)
            optimizer.zero_grad()
            optim_to(optimizer, devices.cpu)
            shared.sd_model.cond_stage_model.to(devices.device)
            shared.sd_model.first_stage_model.to(devices.device)

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                do_not_save_grid=True,
                do_not_save_samples=True,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.negative_prompt = preview_negative_prompt
                p.steps = preview_steps
                p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
                p.cfg_scale = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                p.steps = 20

            preview_text = p.prompt

            processed = processing.process_images(p)
            image = processed.images[0] if len(processed.images) > 0 else None

            if unload:
                shared.sd_model.cond_stage_model.to(devices.cpu)
                shared.sd_model.first_stage_model.to(devices.cpu)

            if image is not None:
                shared.state.current_image = image
                last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt,
                                                                     shared.opts.samples_format, processed.infotexts[0],
                                                                     p=p, forced_filename=forced_filename,
                                                                     save_to_dirs=False)
                last_saved_image += f", prompt: {preview_text}"
            optim_to(optimizer, devices.device)

        shared.state.job_no = hypernetwork.step

        shared.state.textinfo = f"""
<p>
Loss: {previous_mean_loss:.7f}<br/>
Step: {hypernetwork.step}<br/>
Last prompt: {html.escape(entries[0].cond_text)}<br/>
Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    report_statistics(loss_dict)

    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')
    hypernetwork.optimizer_name = optimizer_name
    if shared.opts.save_optimizer_state:
        hypernetwork.optimizer_state_dict = optimizer.state_dict()
    save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, filename)
    del optimizer
    hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.
    hypernetwork.eval()
    return hypernetwork, filename


modules.hypernetworks.hypernetwork.list_hypernetworks = list_hypernetworks
modules.hypernetworks.hypernetwork.load_hypernetwork = load_hypernetwork
modules.hypernetworks.hypernetwork.apply_hypernetwork = apply_hypernetwork
