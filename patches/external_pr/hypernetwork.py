import datetime
import gc
import html
import json
import os
import sys
import time
import traceback
from collections import defaultdict, deque

import torch
import tqdm

from modules import shared, sd_models, devices, processing, sd_samplers
from modules.hypernetworks.hypernetwork import optimizer_dict, stack_conds, save_hypernetwork, report_statistics
from modules.textual_inversion import textual_inversion
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from ..tbutils import tensorboard_setup, tensorboard_add, tensorboard_add_image, tensorboard_log_hyperparameter
from .textual_inversion import validate_train_inputs, write_loss
from ..hypernetwork import Hypernetwork, load_hypernetwork
from . import sd_hijack_checkpoint
from ..hnutil import optim_to
from ..ui import create_hypernetwork_load
from ..scheduler import CosineAnnealingWarmUpRestarts
from .dataset import PersonalizedBase, PersonalizedDataLoader
from ..ddpm_hijack import set_scheduler


def get_lr_from_optimizer(optimizer: torch.optim.Optimizer):
    return optimizer.param_groups[0].get('d', 1) * optimizer.param_groups[0].get('lr', 1)


def set_accessible(obj):
    setattr(shared, 'accessible_hypernetwork', obj)
    if hasattr(shared, 'loaded_hypernetworks'):
        shared.loaded_hypernetworks.clear()
        shared.loaded_hypernetworks = [obj,]


def remove_accessible():
    delattr(shared, 'accessible_hypernetwork')
    if hasattr(shared, 'loaded_hypernetworks'):
        shared.loaded_hypernetworks.clear()

def get_training_option(filename):
    print(filename)
    if os.path.exists(os.path.join(shared.cmd_opts.hypernetwork_dir, filename)) and os.path.isfile(
            os.path.join(shared.cmd_opts.hypernetwork_dir, filename)):
        filename = os.path.join(shared.cmd_opts.hypernetwork_dir, filename)
    elif os.path.exists(filename) and os.path.isfile(filename):
        filename = filename
    elif os.path.exists(os.path.join(shared.cmd_opts.hypernetwork_dir, filename + '.json')) and os.path.isfile(
            os.path.join(shared.cmd_opts.hypernetwork_dir, filename + '.json')):
        filename = os.path.join(shared.cmd_opts.hypernetwork_dir, filename + '.json')
    else:
        return False
    print(f"Loading setting from {filename}!")
    with open(filename, 'r') as file:
        obj = json.load(file)
    return obj


def prepare_training_hypernetwork(hypernetwork_name, learn_rate=0.1, use_adamw_parameter=False, use_dadaptation=False, **adamW_kwarg_dict):
    """ returns hypernetwork object binded with optimizer"""
    hypernetwork = load_hypernetwork(hypernetwork_name)
    hypernetwork.to(devices.device)
    assert hypernetwork is not None, f"Cannot load {hypernetwork_name}!"
    if not isinstance(hypernetwork, Hypernetwork):
        raise RuntimeError("Cannot perform training for Hypernetwork structure pipeline!")
    set_accessible(hypernetwork)
    weights = hypernetwork.weights(True)
    hypernetwork_name = hypernetwork_name.rsplit('(', 1)[0]
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')
    # Here we use optimizer from saved HN, or we can specify as UI option.
    if hypernetwork.optimizer_name == 'DAdaptAdamW':
        use_dadaptation = True
    optimizer = None
    optimizer_name = 'AdamW'
    # Here we use optimizer from saved HN, or we can specify as UI option.
    if hypernetwork.optimizer_name in optimizer_dict:
        if use_adamw_parameter:
            if hypernetwork.optimizer_name != 'AdamW' and hypernetwork.optimizer_name != 'DAdaptAdamW':
                raise NotImplementedError(f"Cannot use adamW paramters for optimizer {hypernetwork.optimizer_name}!")
            if use_dadaptation:
                from .dadapt_test.install import get_dadapt_adam
                optim_class = get_dadapt_adam(hypernetwork.optimizer_name)
                if optim_class != torch.optim.AdamW:
                    print('Optimizer class is ' + str(optim_class))
                    optimizer = optim_class(params=weights, lr=learn_rate, decouple=True, **adamW_kwarg_dict)
                    hypernetwork.optimizer_name = 'DAdaptAdamW'
                else:
                    optimizer = torch.optim.AdamW(params=weights, lr=learn_rate, **adamW_kwarg_dict)
            else:
                optimizer = torch.optim.AdamW(params=weights, lr=learn_rate, **adamW_kwarg_dict)
        else:
            optimizer = optimizer_dict[hypernetwork.optimizer_name](params=weights, lr=learn_rate)
        optimizer_name = hypernetwork.optimizer_name
    else:
        print(f"Optimizer type {hypernetwork.optimizer_name} is not defined!")
        if use_dadaptation:
            from .dadapt_test.install import get_dadapt_adam
            optim_class = get_dadapt_adam(hypernetwork.optimizer_name)
            if optim_class != torch.optim.AdamW:
                optimizer = optim_class(params=weights, lr=learn_rate, decouple=True, **adamW_kwarg_dict)
                optimizer_name = 'DAdaptAdamW'
                hypernetwork.optimizer_name = 'DAdaptAdamW'
    if optimizer is None:
        optimizer = torch.optim.AdamW(params=weights, lr=learn_rate, **adamW_kwarg_dict)
        optimizer_name = 'AdamW'
    if hypernetwork.optimizer_state_dict:  # This line must be changed if Optimizer type can be different from saved optimizer.
        try:
            optimizer.load_state_dict(hypernetwork.optimizer_state_dict)
            optim_to(optimizer, devices.device)
            print('Loaded optimizer successfully!')
        except RuntimeError as e:
            print("Cannot resume from saved optimizer!")
            print(e)

    return hypernetwork, optimizer, weights, optimizer_name

def train_hypernetwork(id_task, hypernetwork_name, learn_rate, batch_size, gradient_step, data_root, log_directory,
                       training_width, training_height, steps, shuffle_tags, tag_drop_out, latent_sampling_method,
                       create_image_every, save_hypernetwork_every, template_file, preview_from_txt2img, preview_prompt,
                       preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed,
                       preview_width, preview_height,
                       use_beta_scheduler=False, beta_repeat_epoch=4000, epoch_mult=1, warmup=10, min_lr=1e-7,
                       gamma_rate=1, save_when_converge=False, create_when_converge=False,
                       move_optimizer=True,
                       use_adamw_parameter=False, adamw_weight_decay=0.01, adamw_beta_1=0.9, adamw_beta_2=0.99,
                       adamw_eps=1e-8,
                       use_grad_opts=False, gradient_clip_opt='None', optional_gradient_clip_value=1e01,
                       optional_gradient_norm_type=2, latent_sampling_std=-1,
                       noise_training_scheduler_enabled=False, noise_training_scheduler_repeat=False, noise_training_scheduler_cycle=128,
                       load_training_options='', loss_opt='loss_simple', use_dadaptation=False
                       ):
    # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
    from modules import images
    if load_training_options != '':
        dump: dict = get_training_option(load_training_options)
        if dump and dump is not None:
            print(f"Loading from {load_training_options}")
            learn_rate = dump['learn_rate']
            batch_size = dump['batch_size']
            gradient_step = dump['gradient_step']
            training_width = dump['training_width']
            training_height = dump['training_height']
            steps = dump['steps']
            shuffle_tags = dump['shuffle_tags']
            tag_drop_out = dump['tag_drop_out']
            save_when_converge = dump['save_when_converge']
            create_when_converge = dump['create_when_converge']
            latent_sampling_method = dump['latent_sampling_method']
            template_file = dump['template_file']
            use_beta_scheduler = dump['use_beta_scheduler']
            beta_repeat_epoch = dump['beta_repeat_epoch']
            epoch_mult = dump['epoch_mult']
            warmup = dump['warmup']
            min_lr = dump['min_lr']
            gamma_rate = dump['gamma_rate']
            use_adamw_parameter = dump['use_beta_adamW_checkbox']
            adamw_weight_decay = dump['adamw_weight_decay']
            adamw_beta_1 = dump['adamw_beta_1']
            adamw_beta_2 = dump['adamw_beta_2']
            adamw_eps = dump['adamw_eps']
            use_grad_opts = dump['show_gradient_clip_checkbox']
            gradient_clip_opt = dump['gradient_clip_opt']
            optional_gradient_clip_value = dump['optional_gradient_clip_value']
            optional_gradient_norm_type = dump['optional_gradient_norm_type']
            latent_sampling_std = dump.get('latent_sampling_std', -1)
            noise_training_scheduler_enabled = dump.get('noise_training_scheduler_enabled', False)
            noise_training_scheduler_repeat = dump.get('noise_training_scheduler_repeat', False)
            noise_training_scheduler_cycle = dump.get('noise_training_scheduler_cycle', 128)
            loss_opt = dump.get('loss_opt', 'loss_simple')
            use_dadaptation = dump.get('use_dadaptation', False)
    try:
        if use_adamw_parameter:
            adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps = [float(x) for x in
                                                                         [adamw_weight_decay, adamw_beta_1,
                                                                          adamw_beta_2, adamw_eps]]
            assert 0 <= adamw_weight_decay, "Weight decay paramter should be larger or equal than zero!"
            assert (all(0 <= x <= 1 for x in [adamw_beta_1, adamw_beta_2,
                                              adamw_eps])), "Cannot use negative or >1 number for adamW parameters!"
            adamW_kwarg_dict = {
                'weight_decay': adamw_weight_decay,
                'betas': (adamw_beta_1, adamw_beta_2),
                'eps': adamw_eps
            }
            print('Using custom AdamW parameters')
        else:
            adamW_kwarg_dict = {
                'weight_decay': 0.01,
                'betas': (0.9, 0.99),
                'eps': 1e-8
            }
        if use_beta_scheduler:
            print("Using Beta Scheduler")
            beta_repeat_epoch = int(beta_repeat_epoch)
            assert beta_repeat_epoch > 0, f"Cannot use too small cycle {beta_repeat_epoch}!"
            min_lr = float(min_lr)
            assert min_lr < 1, f"Cannot use minimum lr with {min_lr}!"
            gamma_rate = float(gamma_rate)
            print(f"Using learn rate decay(per cycle) of {gamma_rate}")
            assert 0 <= gamma_rate <= 1, f"Cannot use gamma rate with {gamma_rate}!"
            epoch_mult = float(epoch_mult)
            assert 1 <= epoch_mult, "Cannot use epoch multiplier smaller than 1!"
            warmup = int(warmup)
            assert warmup >= 1, "Warmup epoch should be larger than 0!"
            print(f"Save when converges : {save_when_converge}")
            print(f"Generate image when converges : {create_when_converge}")
        else:
            beta_repeat_epoch = 4000
            epoch_mult = 1
            warmup = 10
            min_lr = 1e-7
            gamma_rate = 1
            save_when_converge = False
            create_when_converge = False
    except ValueError:
        raise RuntimeError("Cannot use advanced LR scheduler settings!")
    if noise_training_scheduler_enabled:
        set_scheduler(noise_training_scheduler_cycle, noise_training_scheduler_repeat, True)
        print(f"Noise training scheduler is now ready for {noise_training_scheduler_cycle}, {noise_training_scheduler_repeat}!")
    else:
        set_scheduler(-1, False, False)
    if use_grad_opts and gradient_clip_opt != "None":
        try:
            optional_gradient_clip_value = float(optional_gradient_clip_value)
        except ValueError:
            raise RuntimeError(f"Cannot convert invalid gradient clipping value {optional_gradient_clip_value})")
        if gradient_clip_opt == "Norm":
            try:
                grad_norm = int(optional_gradient_norm_type)
            except ValueError:
                raise RuntimeError(f"Cannot convert invalid gradient norm type {optional_gradient_norm_type})")
            assert grad_norm >= 0, f"P-norm cannot be calculated from negative number {grad_norm}"
            print(
                f"Using gradient clipping by Norm, norm type {optional_gradient_norm_type}, norm limit {optional_gradient_clip_value}")

            def gradient_clipping(arg1):
                torch.nn.utils.clip_grad_norm_(arg1, optional_gradient_clip_value, optional_gradient_norm_type)
                return
        else:
            print(f"Using gradient clipping by Value, limit {optional_gradient_clip_value}")

            def gradient_clipping(arg1):
                torch.nn.utils.clip_grad_value_(arg1, optional_gradient_clip_value)
                return
    else:
        def gradient_clipping(arg1):
            return
    save_hypernetwork_every = save_hypernetwork_every or 0
    create_image_every = create_image_every or 0
    if not os.path.isfile(template_file):
        template_file = textual_inversion.textual_inversion_templates.get(template_file, None)
        if template_file is not None:
            template_file = template_file.path
        else:
            raise AssertionError(f"Cannot find {template_file}!")
    validate_train_inputs(hypernetwork_name, learn_rate, batch_size, gradient_step, data_root, template_file, steps, save_hypernetwork_every, create_image_every, log_directory, name="hypernetwork")
    shared.state.job = "train-hypernetwork"
    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps
    tmp_scheduler = LearnRateScheduler(learn_rate, steps, 0)
    hypernetwork, optimizer, weights, optimizer_name = prepare_training_hypernetwork(hypernetwork_name, tmp_scheduler.learn_rate, use_adamw_parameter, use_dadaptation, **adamW_kwarg_dict)

    hypernetwork_name = hypernetwork_name.rsplit('(', 1)[0]
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
    unload = shared.opts.unload_models_when_training

    if save_hypernetwork_every > 0 or save_when_converge:
        hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
        os.makedirs(hypernetwork_dir, exist_ok=True)
    else:
        hypernetwork_dir = None

    if create_image_every > 0 or create_when_converge:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    checkpoint = sd_models.select_checkpoint()

    initial_step = hypernetwork.step or 0
    if initial_step >= steps:
        shared.state.textinfo = f"Model has already been trained beyond specified max steps"
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    if shared.opts.training_enable_tensorboard:
        print("Tensorboard logging enabled")
        tensorboard_writer = tensorboard_setup(log_directory)
    else:
        tensorboard_writer = None
    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    detach_grad = shared.opts.disable_ema  # test code that removes EMA
    if detach_grad:
        print("Disabling training for staged models!")
        shared.sd_model.cond_stage_model.requires_grad_(False)
        shared.sd_model.first_stage_model.requires_grad_(False)
        torch.cuda.empty_cache()
    pin_memory = shared.opts.pin_memory

    ds = PersonalizedBase(data_root=data_root, width=training_width,
                          height=training_height,
                          repeats=shared.opts.training_image_repeats_per_epoch,
                          placeholder_token=hypernetwork_name, model=shared.sd_model,
                          cond_model=shared.sd_model.cond_stage_model,
                          device=devices.device, template_file=template_file,
                          include_cond=True, batch_size=batch_size,
                          gradient_step=gradient_step, shuffle_tags=shuffle_tags,
                          tag_drop_out=tag_drop_out,
                          latent_sampling_method=latent_sampling_method,
                          latent_sampling_std=latent_sampling_std)

    latent_sampling_method = ds.latent_sampling_method

    dl = PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method,
                                batch_size=ds.batch_size, pin_memory=pin_memory)
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    if unload:
        shared.parallel_processing_allowed = False
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    if use_beta_scheduler:
        scheduler_beta = CosineAnnealingWarmUpRestarts(optimizer=optimizer, first_cycle_steps=beta_repeat_epoch,
                                                       cycle_mult=epoch_mult, max_lr=scheduler.learn_rate,
                                                       warmup_steps=warmup, min_lr=min_lr, gamma=gamma_rate)
        scheduler_beta.last_epoch = hypernetwork.step - 1
    else:
        scheduler_beta = None
        for pg in optimizer.param_groups:
            pg['lr'] = scheduler.learn_rate
    scaler = torch.cuda.amp.GradScaler()

    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0  # internal
    # size = len(ds.indexes)
    loss_dict = defaultdict(lambda: deque(maxlen=1024))
    # losses = torch.zeros((size,))
    # previous_mean_losses = [0]
    # previous_mean_loss = 0
    # print("Mean loss of {} elements".format(size))

    steps_without_grad = 0

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    if hasattr(sd_hijack_checkpoint, 'add'):
        sd_hijack_checkpoint.add()
    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        for i in range((steps - initial_step) * gradient_step):
            if scheduler.finished or hypernetwork.step > steps:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                if use_beta_scheduler:
                    scheduler_beta.step(hypernetwork.step)
                else:
                    scheduler.apply(optimizer, hypernetwork.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                with torch.autocast("cuda"):
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    if tag_drop_out != 0 or shuffle_tags:
                        shared.sd_model.cond_stage_model.to(devices.device)
                        c = shared.sd_model.cond_stage_model(batch.cond_text).to(devices.device,
                                                                                 non_blocking=pin_memory)
                        shared.sd_model.cond_stage_model.to(devices.cpu)
                    else:
                        c = stack_conds(batch.cond).to(devices.device, non_blocking=pin_memory)
                    _, losses = shared.sd_model(x, c)
                    loss = losses['val/' + loss_opt]
                    for filenames in batch.filename:
                        loss_dict[filenames].append(loss.detach().item())
                    loss /= gradient_step
                    del x
                    del c

                    _loss_step += loss.item()
                    scaler.scale(loss).backward()
                    batch.latent_sample.to(devices.cpu)
                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue
                gradient_clipping(weights)
                # print(f"grad:{weights[0].grad.detach().cpu().abs().mean().item():.7f}")
                # scaler.unscale_(optimizer)
                # print(f"grad:{weights[0].grad.detach().cpu().abs().mean().item():.15f}")
                # torch.nn.utils.clip_grad_norm_(weights, max_norm=1.0)
                # print(f"grad:{weights[0].grad.detach().cpu().abs().mean().item():.15f}")
                try:
                    scaler.step(optimizer)
                except AssertionError:
                    optimizer.param_groups[0]['capturable'] = True
                    scaler.step(optimizer)
                scaler.update()
                hypernetwork.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = hypernetwork.step + 1

                epoch_num = hypernetwork.step // steps_per_epoch
                epoch_step = hypernetwork.step % steps_per_epoch

                description = f"Training hypernetwork [Epoch {epoch_num}: {epoch_step + 1}/{steps_per_epoch}]loss: {loss_step:.7f}"
                pbar.set_description(description)
                if hypernetwork_dir is not None and (
                        (use_beta_scheduler and scheduler_beta.is_EOC(hypernetwork.step) and save_when_converge) or (
                        save_hypernetwork_every > 0 and steps_done % save_hypernetwork_every == 0)):
                    # Before saving, change name to match current checkpoint.
                    hypernetwork_name_every = f'{hypernetwork_name}-{steps_done}'
                    last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork_name_every}.pt')
                    hypernetwork.optimizer_name = optimizer_name
                    if shared.opts.save_optimizer_state:
                        hypernetwork.optimizer_state_dict = optimizer.state_dict()
                    save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, last_saved_file)
                    hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.

                write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, steps_per_epoch,
                           {
                               "loss": f"{loss_step:.7f}",
                               "learn_rate": get_lr_from_optimizer(optimizer)
                           })
                if shared.opts.training_enable_tensorboard:
                    epoch_num = hypernetwork.step // len(ds)
                    epoch_step = hypernetwork.step - (epoch_num * len(ds)) + 1
                    mean_loss = sum(sum(x) for x in loss_dict.values()) / sum(len(x) for x in loss_dict.values())
                    tensorboard_add(tensorboard_writer, loss=mean_loss, global_step=hypernetwork.step, step=epoch_step,
                                    learn_rate=scheduler.learn_rate if not use_beta_scheduler else
                                    get_lr_from_optimizer(optimizer), epoch_num=epoch_num)
                if images_dir is not None and (
                        use_beta_scheduler and scheduler_beta.is_EOC(hypernetwork.step) and create_when_converge) or (
                        create_image_every > 0 and steps_done % create_image_every == 0):
                    set_scheduler(-1, False, False)
                    forced_filename = f'{hypernetwork_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)
                    rng_state = torch.get_rng_state()
                    cuda_rng_state = None
                    if torch.cuda.is_available():
                        cuda_rng_state = torch.cuda.get_rng_state_all()
                    hypernetwork.eval()
                    if move_optimizer:
                        optim_to(optimizer, devices.cpu)
                        gc.collect()
                    shared.sd_model.cond_stage_model.to(devices.device)
                    shared.sd_model.first_stage_model.to(devices.device)

                    p = processing.StableDiffusionProcessingTxt2Img(
                        sd_model=shared.sd_model,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                    )
                    if hasattr(p, 'disable_extra_networks'):
                        p.disable_extra_networks = True
                        is_patched = True
                    else:
                        is_patched = False
                    if preview_from_txt2img:
                        p.prompt = preview_prompt + (hypernetwork.extra_name() if not is_patched else "")
                        print(p.prompt)
                        p.negative_prompt = preview_negative_prompt
                        p.steps = preview_steps
                        p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        p.prompt = batch.cond_text[0] + (hypernetwork.extra_name() if not is_patched else "")
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt

                    processed = processing.process_images(p)
                    image = processed.images[0] if len(processed.images) > 0 else None
                    if shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:
                        tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image,
                                              hypernetwork.step)

                    if unload:
                        shared.sd_model.cond_stage_model.to(devices.cpu)
                        shared.sd_model.first_stage_model.to(devices.cpu)
                    torch.set_rng_state(rng_state)
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    hypernetwork.train()
                    if move_optimizer:
                        optim_to(optimizer, devices.device)
                    if noise_training_scheduler_enabled:
                        set_scheduler(noise_training_scheduler_cycle, noise_training_scheduler_repeat, True)
                    if image is not None:
                        if hasattr(shared.state, 'assign_current_image'):
                            shared.state.assign_current_image(image)
                        else:
                            shared.state.current_image = image
                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt,
                                                                             shared.opts.samples_format,
                                                                             processed.infotexts[0], p=p,
                                                                             forced_filename=forced_filename,
                                                                             save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"
                    set_accessible(hypernetwork)

                shared.state.job_no = hypernetwork.step

                shared.state.textinfo = f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        pbar.leave = False
        pbar.close()
        hypernetwork.eval()
        shared.parallel_processing_allowed = old_parallel_processing_allowed
        if hasattr(sd_hijack_checkpoint, 'remove'):
            sd_hijack_checkpoint.remove()
        set_scheduler(-1, False, False)
        remove_accessible()
        gc.collect()
        torch.cuda.empty_cache()
    report_statistics(loss_dict)
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')
    hypernetwork.optimizer_name = optimizer_name
    if shared.opts.save_optimizer_state:
        hypernetwork.optimizer_state_dict = optimizer.state_dict()
    save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, filename)
    del optimizer
    hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.
    shared.sd_model.cond_stage_model.to(devices.device)
    shared.sd_model.first_stage_model.to(devices.device)

    return hypernetwork, filename


def internal_clean_training(hypernetwork_name, data_root, log_directory,
                            create_image_every, save_hypernetwork_every,
                            preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps,
                            preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height,
                            move_optimizer=True,
                            load_hypernetworks_option='', load_training_options='', manual_dataset_seed=-1,
                            setting_tuple=None):
    # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
    from modules import images
    base_hypernetwork_name = hypernetwork_name
    manual_seed = int(manual_dataset_seed)
    if setting_tuple is not None:
        setting_suffix = f"_{setting_tuple[0]}_{setting_tuple[1]}"
    else:
        setting_suffix = time.strftime('%Y%m%d%H%M%S')
    if load_hypernetworks_option != '':
        dump_hyper: dict = get_training_option(load_hypernetworks_option)
        hypernetwork_name = hypernetwork_name + setting_suffix
        enable_sizes = dump_hyper['enable_sizes']
        overwrite_old = dump_hyper['overwrite_old']
        layer_structure = dump_hyper['layer_structure']
        activation_func = dump_hyper['activation_func']
        weight_init = dump_hyper['weight_init']
        add_layer_norm = dump_hyper['add_layer_norm']
        use_dropout = dump_hyper['use_dropout']
        dropout_structure = dump_hyper['dropout_structure']
        optional_info = dump_hyper['optional_info']
        weight_init_seed = dump_hyper['weight_init_seed']
        normal_std = dump_hyper['normal_std']
        skip_connection = dump_hyper['skip_connection']
        hypernetwork = create_hypernetwork_load(hypernetwork_name, enable_sizes, overwrite_old, layer_structure,
                                                activation_func, weight_init, add_layer_norm, use_dropout,
                                                dropout_structure, optional_info, weight_init_seed, normal_std,
                                                skip_connection)
    else:
        hypernetwork = load_hypernetwork(hypernetwork_name)
        hypernetwork_name = hypernetwork_name.rsplit('(', 1)[0] + setting_suffix
        hypernetwork.save(os.path.join(shared.cmd_opts.hypernetwork_dir, f"{hypernetwork_name}.pt"))
        shared.reload_hypernetworks()
        hypernetwork = load_hypernetwork(hypernetwork_name)
    if load_training_options != '':
        dump: dict = get_training_option(load_training_options)
        if dump and dump is not None:
            learn_rate = dump['learn_rate']
            batch_size = dump['batch_size']
            gradient_step = dump['gradient_step']
            training_width = dump['training_width']
            training_height = dump['training_height']
            steps = dump['steps']
            shuffle_tags = dump['shuffle_tags']
            tag_drop_out = dump['tag_drop_out']
            save_when_converge = dump['save_when_converge']
            create_when_converge = dump['create_when_converge']
            latent_sampling_method = dump['latent_sampling_method']
            template_file = dump['template_file']
            use_beta_scheduler = dump['use_beta_scheduler']
            beta_repeat_epoch = dump['beta_repeat_epoch']
            epoch_mult = dump['epoch_mult']
            warmup = dump['warmup']
            min_lr = dump['min_lr']
            gamma_rate = dump['gamma_rate']
            use_adamw_parameter = dump['use_beta_adamW_checkbox']
            adamw_weight_decay = dump['adamw_weight_decay']
            adamw_beta_1 = dump['adamw_beta_1']
            adamw_beta_2 = dump['adamw_beta_2']
            adamw_eps = dump['adamw_eps']
            use_grad_opts = dump['show_gradient_clip_checkbox']
            gradient_clip_opt = dump['gradient_clip_opt']
            optional_gradient_clip_value = dump['optional_gradient_clip_value']
            optional_gradient_norm_type = dump['optional_gradient_norm_type']
            latent_sampling_std = dump.get('latent_sampling_std', -1)
            noise_training_scheduler_enabled = dump.get('noise_training_scheduler_enabled', False)
            noise_training_scheduler_repeat = dump.get('noise_training_scheduler_repeat', False)
            noise_training_scheduler_cycle = dump.get('noise_training_scheduler_cycle', 128)
            loss_opt = dump.get('loss_opt', 'loss_simple')
            use_dadaptation = dump.get('use_dadaptation', False)
        else:
            raise RuntimeError(f"Cannot load from {load_training_options}!")
    else:
        raise RuntimeError(f"Cannot load from {load_training_options}!")
    try:
        if use_adamw_parameter:
            adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps = [float(x) for x in
                                                                         [adamw_weight_decay, adamw_beta_1,
                                                                          adamw_beta_2, adamw_eps]]
            assert 0 <= adamw_weight_decay, "Weight decay paramter should be larger or equal than zero!"
            assert (all(0 <= x <= 1 for x in [adamw_beta_1, adamw_beta_2,
                                              adamw_eps])), "Cannot use negative or >1 number for adamW parameters!"
            adamW_kwarg_dict = {
                'weight_decay': adamw_weight_decay,
                'betas': (adamw_beta_1, adamw_beta_2),
                'eps': adamw_eps
            }
            print('Using custom AdamW parameters')
        else:
            adamW_kwarg_dict = {
                'weight_decay': 0.01,
                'betas': (0.9, 0.99),
                'eps': 1e-8
            }
        if use_beta_scheduler:
            print("Using Beta Scheduler")
            beta_repeat_epoch = int(beta_repeat_epoch)
            assert beta_repeat_epoch > 0, f"Cannot use too small cycle {beta_repeat_epoch}!"
            min_lr = float(min_lr)
            assert min_lr < 1, f"Cannot use minimum lr with {min_lr}!"
            gamma_rate = float(gamma_rate)
            print(f"Using learn rate decay(per cycle) of {gamma_rate}")
            assert 0 <= gamma_rate <= 1, f"Cannot use gamma rate with {gamma_rate}!"
            epoch_mult = float(epoch_mult)
            assert 1 <= epoch_mult, "Cannot use epoch multiplier smaller than 1!"
            warmup = int(warmup)
            assert warmup >= 1, "Warmup epoch should be larger than 0!"
            print(f"Save when converges : {save_when_converge}")
            print(f"Generate image when converges : {create_when_converge}")
        else:
            beta_repeat_epoch = 4000
            epoch_mult = 1
            warmup = 10
            min_lr = 1e-7
            gamma_rate = 1
            save_when_converge = False
            create_when_converge = False
    except ValueError:
        raise RuntimeError("Cannot use advanced LR scheduler settings!")
    if use_grad_opts and gradient_clip_opt != "None":
        try:
            optional_gradient_clip_value = float(optional_gradient_clip_value)
        except ValueError:
            raise RuntimeError(f"Cannot convert invalid gradient clipping value {optional_gradient_clip_value})")
        if gradient_clip_opt == "Norm":
            try:
                grad_norm = int(optional_gradient_norm_type)
            except ValueError:
                raise RuntimeError(f"Cannot convert invalid gradient norm type {optional_gradient_norm_type})")
            assert grad_norm >= 0, f"P-norm cannot be calculated from negative number {grad_norm}"
            print(
                f"Using gradient clipping by Norm, norm type {optional_gradient_norm_type}, norm limit {optional_gradient_clip_value}")

            def gradient_clipping(arg1):
                torch.nn.utils.clip_grad_norm_(arg1, optional_gradient_clip_value, optional_gradient_norm_type)
                return
        else:
            print(f"Using gradient clipping by Value, limit {optional_gradient_clip_value}")

            def gradient_clipping(arg1):
                torch.nn.utils.clip_grad_value_(arg1, optional_gradient_clip_value)
                return
    else:
        def gradient_clipping(arg1):
            return
    if noise_training_scheduler_enabled:
        set_scheduler(noise_training_scheduler_cycle, noise_training_scheduler_repeat, True)
        print(f"Noise training scheduler is now ready for {noise_training_scheduler_cycle}, {noise_training_scheduler_repeat}!")
    else:
        set_scheduler(-1, False, False)
    save_hypernetwork_every = save_hypernetwork_every or 0
    create_image_every = create_image_every or 0
    if not os.path.isfile(template_file):
        template_file = textual_inversion.textual_inversion_templates.get(template_file, None)
        if template_file is not None:
            template_file = template_file.path
        else:
            raise AssertionError(f"Cannot find {template_file}!")
    validate_train_inputs(hypernetwork_name, learn_rate, batch_size, gradient_step, data_root, template_file, steps, save_hypernetwork_every, create_image_every, log_directory, name="hypernetwork")
    hypernetwork.to(devices.device)
    assert hypernetwork is not None, f"Cannot load {hypernetwork_name}!"
    if not isinstance(hypernetwork, Hypernetwork):
        raise RuntimeError("Cannot perform training for Hypernetwork structure pipeline!")
    set_accessible(hypernetwork)
    shared.state.job = "train-hypernetwork"
    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps

    hypernetwork_name = hypernetwork_name.rsplit('(', 1)[0]
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')
    base_log_directory = log_directory
    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
    unload = shared.opts.unload_models_when_training

    if save_hypernetwork_every > 0 or save_when_converge:
        hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
        os.makedirs(hypernetwork_dir, exist_ok=True)
    else:
        hypernetwork_dir = None

    if create_image_every > 0 or create_when_converge:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    checkpoint = sd_models.select_checkpoint()

    initial_step = hypernetwork.step or 0
    if initial_step >= steps:
        shared.state.textinfo = f"Model has already been trained beyond specified max steps"
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    if shared.opts.training_enable_tensorboard:
        print("Tensorboard logging enabled")
        tensorboard_writer = tensorboard_setup(os.path.join(base_log_directory, base_hypernetwork_name))

    else:
        tensorboard_writer = None
    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    detach_grad = shared.opts.disable_ema  # test code that removes EMA
    if detach_grad:
        print("Disabling training for staged models!")
        shared.sd_model.cond_stage_model.requires_grad_(False)
        shared.sd_model.first_stage_model.requires_grad_(False)
        torch.cuda.empty_cache()
    pin_memory = shared.opts.pin_memory
    ds = PersonalizedBase(data_root=data_root, width=training_width,
                          height=training_height,
                          repeats=shared.opts.training_image_repeats_per_epoch,
                          placeholder_token=hypernetwork_name, model=shared.sd_model,
                          cond_model=shared.sd_model.cond_stage_model,
                          device=devices.device, template_file=template_file,
                          include_cond=True, batch_size=batch_size,
                          gradient_step=gradient_step, shuffle_tags=shuffle_tags,
                          tag_drop_out=tag_drop_out,
                          latent_sampling_method=latent_sampling_method,
                          latent_sampling_std=latent_sampling_std,
                          manual_seed=manual_seed)

    latent_sampling_method = ds.latent_sampling_method

    dl = PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method,
                                batch_size=ds.batch_size, pin_memory=pin_memory)
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    if unload:
        shared.parallel_processing_allowed = False
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    weights = hypernetwork.weights(True)
    if hypernetwork.optimizer_name == 'DAdaptAdamW':
        use_dadaptation = True
    optimizer = None
    # Here we use optimizer from saved HN, or we can specify as UI option.
    if hypernetwork.optimizer_name in optimizer_dict:
        if use_adamw_parameter:
            if hypernetwork.optimizer_name != 'AdamW' and hypernetwork.optimizer_name != 'DAdaptAdamW':
                raise RuntimeError(f"Cannot use adamW paramters for optimizer {hypernetwork.optimizer_name}!")
            if use_dadaptation:
                from .dadapt_test.install import get_dadapt_adam
                optim_class = get_dadapt_adam(hypernetwork.optimizer_name)
                if optim_class != torch.optim.AdamW:
                    optimizer = optim_class(params=weights, lr=scheduler.learn_rate, decouple=True, **adamW_kwarg_dict)
                else:
                    optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate, **adamW_kwarg_dict)
            else:
                optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate, **adamW_kwarg_dict)
        else:
            optimizer = optimizer_dict[hypernetwork.optimizer_name](params=weights, lr=scheduler.learn_rate)
        optimizer_name = hypernetwork.optimizer_name
    else:
        print(f"Optimizer type {hypernetwork.optimizer_name} is not defined!")
        if use_dadaptation:
            from .dadapt_test.install import get_dadapt_adam
            optim_class = get_dadapt_adam(hypernetwork.optimizer_name)
            if optim_class != torch.optim.AdamW:
                optimizer = optim_class(params=weights, lr=scheduler.learn_rate, decouple=True, **adamW_kwarg_dict)
                optimizer_name = 'DAdaptAdamW'
    if optimizer is None:
        optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate, **adamW_kwarg_dict)
        optimizer_name = 'AdamW'



    if hypernetwork.optimizer_state_dict:  # This line must be changed if Optimizer type can be different from saved optimizer.
        try:
            optimizer.load_state_dict(hypernetwork.optimizer_state_dict)
        except RuntimeError as e:
            print("Cannot resume from saved optimizer!")
            print(e)
    optim_to(optimizer, devices.device)
    if use_beta_scheduler:
        scheduler_beta = CosineAnnealingWarmUpRestarts(optimizer=optimizer, first_cycle_steps=beta_repeat_epoch,
                                                       cycle_mult=epoch_mult, max_lr=scheduler.learn_rate,
                                                       warmup_steps=warmup, min_lr=min_lr, gamma=gamma_rate)
        scheduler_beta.last_epoch = hypernetwork.step - 1
    else:
        scheduler_beta = None
        for pg in optimizer.param_groups:
            pg['lr'] = scheduler.learn_rate
    scaler = torch.cuda.amp.GradScaler()

    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0  # internal
    # size = len(ds.indexes)
    loss_dict = defaultdict(lambda: deque(maxlen=1024))
    # losses = torch.zeros((size,))
    # previous_mean_losses = [0]
    # previous_mean_loss = 0
    # print("Mean loss of {} elements".format(size))

    steps_without_grad = 0

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    if hasattr(sd_hijack_checkpoint, 'add'):
        sd_hijack_checkpoint.add()
    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        for i in range((steps - initial_step) * gradient_step):
            if scheduler.finished or hypernetwork.step > steps:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                if use_beta_scheduler:
                    scheduler_beta.step(hypernetwork.step)
                else:
                    scheduler.apply(optimizer, hypernetwork.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                with torch.autocast("cuda"):
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    if tag_drop_out != 0 or shuffle_tags:
                        shared.sd_model.cond_stage_model.to(devices.device)
                        c = shared.sd_model.cond_stage_model(batch.cond_text).to(devices.device,
                                                                                 non_blocking=pin_memory)
                        shared.sd_model.cond_stage_model.to(devices.cpu)
                    else:
                        c = stack_conds(batch.cond).to(devices.device, non_blocking=pin_memory)
                    _, losses = shared.sd_model(x, c)
                    loss = losses['val/' + loss_opt]
                    for filenames in batch.filename:
                        loss_dict[filenames].append(loss.detach().item())
                    loss /= gradient_step
                    del x
                    del c

                    _loss_step += loss.item()
                    scaler.scale(loss).backward()
                    batch.latent_sample.to(devices.cpu)
                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue
                gradient_clipping(weights)
                # print(f"grad:{weights[0].grad.detach().cpu().abs().mean().item():.7f}")
                # scaler.unscale_(optimizer)
                # print(f"grad:{weights[0].grad.detach().cpu().abs().mean().item():.15f}")
                # torch.nn.utils.clip_grad_norm_(weights, max_norm=1.0)
                # print(f"grad:{weights[0].grad.detach().cpu().abs().mean().item():.15f}")
                try:
                    scaler.step(optimizer)
                except AssertionError:
                    optimizer.param_groups[0]['capturable'] = True
                    scaler.step(optimizer)
                scaler.update()
                hypernetwork.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = hypernetwork.step + 1

                epoch_num = hypernetwork.step // steps_per_epoch
                epoch_step = hypernetwork.step % steps_per_epoch

                description = f"Training hypernetwork [Epoch {epoch_num}: {epoch_step + 1}/{steps_per_epoch}]loss: {loss_step:.7f}"
                pbar.set_description(description)
                if hypernetwork_dir is not None and (
                        (use_beta_scheduler and scheduler_beta.is_EOC(hypernetwork.step) and save_when_converge) or (
                        save_hypernetwork_every > 0 and steps_done % save_hypernetwork_every == 0)):
                    # Before saving, change name to match current checkpoint.
                    hypernetwork_name_every = f'{hypernetwork_name}-{steps_done}'
                    last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork_name_every}.pt')
                    hypernetwork.optimizer_name = optimizer_name
                    if shared.opts.save_optimizer_state:
                        hypernetwork.optimizer_state_dict = optimizer.state_dict()
                    save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, last_saved_file)
                    hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.

                write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, steps_per_epoch,
                           {
                               "loss": f"{loss_step:.7f}",
                               "learn_rate": get_lr_from_optimizer(optimizer)
                           })
                if shared.opts.training_enable_tensorboard:
                    epoch_num = hypernetwork.step // len(ds)
                    epoch_step = hypernetwork.step - (epoch_num * len(ds)) + 1
                    mean_loss = sum(sum(x) for x in loss_dict.values()) / sum(len(x) for x in loss_dict.values())
                    tensorboard_add(tensorboard_writer, loss=mean_loss, global_step=hypernetwork.step, step=epoch_step,
                                    learn_rate=scheduler.learn_rate if not use_beta_scheduler else
                                    get_lr_from_optimizer(optimizer), epoch_num=epoch_num, base_name=hypernetwork_name)
                if images_dir is not None and (
                        use_beta_scheduler and scheduler_beta.is_EOC(hypernetwork.step) and create_when_converge) or (
                        create_image_every > 0 and steps_done % create_image_every == 0):
                    set_scheduler(-1, False, False)
                    forced_filename = f'{hypernetwork_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)
                    rng_state = torch.get_rng_state()
                    cuda_rng_state = None
                    if torch.cuda.is_available():
                        cuda_rng_state = torch.cuda.get_rng_state_all()
                    hypernetwork.eval()
                    if move_optimizer:
                        optim_to(optimizer, devices.cpu)
                    shared.sd_model.cond_stage_model.to(devices.device)
                    shared.sd_model.first_stage_model.to(devices.device)

                    p = processing.StableDiffusionProcessingTxt2Img(
                        sd_model=shared.sd_model,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                    )
                    if hasattr(p, 'disable_extra_networks'):
                        p.disable_extra_networks = True
                        is_patched = True
                    else:
                        is_patched = False
                    if preview_from_txt2img:
                        p.prompt = preview_prompt + (hypernetwork.extra_name() if not is_patched else "")
                        p.negative_prompt = preview_negative_prompt
                        p.steps = preview_steps
                        p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        p.prompt = batch.cond_text[0] + (hypernetwork.extra_name() if not is_patched else "")
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt

                    processed = processing.process_images(p)
                    image = processed.images[0] if len(processed.images) > 0 else None
                    if shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:
                        tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image,
                                              hypernetwork.step, base_name=hypernetwork_name)

                    if unload:
                        shared.sd_model.cond_stage_model.to(devices.cpu)
                        shared.sd_model.first_stage_model.to(devices.cpu)
                    torch.set_rng_state(rng_state)
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    hypernetwork.train()
                    if move_optimizer:
                        optim_to(optimizer, devices.device)
                    if noise_training_scheduler_enabled:
                        set_scheduler(noise_training_scheduler_cycle, noise_training_scheduler_repeat, True)
                    if image is not None:
                        if hasattr(shared.state, 'assign_current_image'):
                            shared.state.assign_current_image(image)
                        else:
                            shared.state.current_image = image
                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt,
                                                                             shared.opts.samples_format,
                                                                             processed.infotexts[0], p=p,
                                                                             forced_filename=forced_filename,
                                                                             save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"
                    set_accessible(hypernetwork)

                shared.state.job_no = hypernetwork.step

                shared.state.textinfo = f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        pbar.leave = False
        pbar.close()
        hypernetwork.eval()
        set_scheduler(-1, False, False)
        shared.parallel_processing_allowed = old_parallel_processing_allowed
        remove_accessible()
        if hasattr(sd_hijack_checkpoint, 'remove'):
            sd_hijack_checkpoint.remove()
        if shared.opts.training_enable_tensorboard:
            mean_loss = sum(sum(x) for x in loss_dict.values()) / sum(len(x) for x in loss_dict.values()) if sum(len(x) for x in loss_dict.values()) > 0 else 0
            tensorboard_log_hyperparameter(tensorboard_writer, lr=learn_rate,
                                           GA_steps=gradient_step,
                                           batch_size=batch_size,
                                           layer_structure=hypernetwork.layer_structure,
                                           activation=hypernetwork.activation_func,
                                           weight_init=hypernetwork.weight_init,
                                           dropout_structure=hypernetwork.dropout_structure,
                                           max_steps=steps,
                                           latent_sampling_method=latent_sampling_method,
                                           template=template_file,
                                           CosineAnnealing=use_beta_scheduler,
                                           beta_repeat_epoch=beta_repeat_epoch,
                                           epoch_mult=epoch_mult,
                                           warmup=warmup,
                                           min_lr=min_lr,
                                           gamma_rate=gamma_rate,
                                           adamW_opts=use_adamw_parameter,
                                           adamW_decay=adamw_weight_decay,
                                           adamW_beta_1=adamw_beta_1,
                                           adamW_beta_2=adamw_beta_2,
                                           adamW_eps=adamw_eps,
                                           gradient_clip=gradient_clip_opt,
                                           gradient_clip_value=optional_gradient_clip_value,
                                           gradient_clip_norm_type=optional_gradient_norm_type,
                                           loss=mean_loss,
                                           base_hypernetwork_name=hypernetwork_name
                                           )
    report_statistics(loss_dict)
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')
    hypernetwork.optimizer_name = optimizer_name
    if shared.opts.save_optimizer_state:
        hypernetwork.optimizer_state_dict = optimizer.state_dict()
    save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, filename)
    del optimizer
    hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.
    shared.sd_model.cond_stage_model.to(devices.device)
    shared.sd_model.first_stage_model.to(devices.device)
    gc.collect()
    torch.cuda.empty_cache()
    return hypernetwork, filename


def train_hypernetwork_tuning(id_task, hypernetwork_name, data_root, log_directory,
                              create_image_every, save_hypernetwork_every, preview_from_txt2img, preview_prompt,
                              preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale,
                              preview_seed,
                              preview_width, preview_height,
                              move_optimizer=True,
                              optional_new_hypernetwork_name='', load_hypernetworks_options='',
                              load_training_options='', manual_dataset_seed=-1):
    load_hypernetworks_options = load_hypernetworks_options.split(',')
    load_training_options = load_training_options.split(',')
    # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
    for _i, load_hypernetworks_option in enumerate(load_hypernetworks_options):
        load_hypernetworks_option = load_hypernetworks_option.strip(' ')
        if load_hypernetworks_option != '' and get_training_option(load_hypernetworks_option) is False:
            print(f"Cannot load from {load_hypernetworks_option}!")
            continue
        for _j, load_training_option in enumerate(load_training_options):
            load_training_option = load_training_option.strip(' ')
            if get_training_option(load_training_option) is False:
                print(f"Cannot load from {load_training_option}!")
                continue
            internal_clean_training(
                hypernetwork_name if load_hypernetworks_option == '' else optional_new_hypernetwork_name, data_root,
                log_directory,
                create_image_every, save_hypernetwork_every,
                preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index,
                preview_cfg_scale, preview_seed, preview_width, preview_height,
                move_optimizer,
                load_hypernetworks_option, load_training_option, manual_dataset_seed, setting_tuple=(_i, _j))
            if shared.state.interrupted:
                return None, None
