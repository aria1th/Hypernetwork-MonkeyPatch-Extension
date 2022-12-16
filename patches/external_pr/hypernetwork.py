import csv
import datetime
import gc
import glob
import html
import os
import sys
import traceback
import inspect
from collections import defaultdict, deque

import torch
import tqdm


from modules import shared, sd_models, devices, processing, sd_samplers
from modules.hypernetworks.hypernetwork import optimizer_dict, stack_conds, save_hypernetwork, report_statistics
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from .textual_inversion import validate_train_inputs, write_loss
from ..hypernetwork import Hypernetwork, load_hypernetwork
from . import sd_hijack_checkpoint
from ..hnutil import optim_to
from ..scheduler import CosineAnnealingWarmUpRestarts
from .dataset import PersonalizedBase,PersonalizedDataLoader


def train_hypernetwork(hypernetwork_name, learn_rate, batch_size, gradient_step, data_root, log_directory,
                       training_width, training_height, steps, shuffle_tags, tag_drop_out, latent_sampling_method,
                       create_image_every, save_hypernetwork_every, template_file, preview_from_txt2img, preview_prompt,
                       preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed,
                       preview_width, preview_height,
                       use_beta_scheduler=False, beta_repeat_epoch=4000, epoch_mult=1,warmup =10, min_lr=1e-7, gamma_rate=1, save_when_converge=False, create_when_converge=False,
                       move_optimizer=True,
                       use_adamw_parameter=False, adamw_weight_decay=0.01, adamw_beta_1=0.9, adamw_beta_2=0.99,adamw_eps=1e-8):
    # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
    from modules import images

    try:
        if use_adamw_parameter:
            adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps = [float(x) for x in [adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps]]
            assert (all(0 <= x <= 1 for x in [adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps])), "Cannot use negative or >1 number for adamW parameters!"
            adamW_kwarg_dict = {
                'weight_decay' : adamw_weight_decay,
                'betas' : (adamw_beta_1, adamw_beta_2),
                'eps' : adamw_eps
            }
            print('Using custom AdamW parameters')
        else:
            adamW_kwarg_dict = {
                'weight_decay' : 0.01,
                'betas' : (0.9, 0.99),
                'eps' : 1e-8
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
            epoch_mult=1
            warmup=10
            min_lr=1e-7
            gamma_rate=1
            save_when_converge = False
            create_when_converge = False
    except ValueError:
        raise RuntimeError("Cannot use advanced LR scheduler settings!")
    save_hypernetwork_every = save_hypernetwork_every or 0
    create_image_every = create_image_every or 0
    validate_train_inputs(hypernetwork_name, learn_rate, batch_size, gradient_step, data_root,
                                            template_file, steps, save_hypernetwork_every, create_image_every,
                                            log_directory, name="hypernetwork")

    load_hypernetwork(hypernetwork_name)
    assert shared.loaded_hypernetwork is not None, f"Cannot load {hypernetwork_name}!"
    if not isinstance(shared.loaded_hypernetwork, Hypernetwork):
        raise RuntimeError("Cannot perform training for Hypernetwork structure pipeline!")

    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps

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

    hypernetwork = shared.loaded_hypernetwork
    checkpoint = sd_models.select_checkpoint()

    initial_step = hypernetwork.step or 0
    if initial_step >= steps:
        shared.state.textinfo = f"Model has already been trained beyond specified max steps"
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)

    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."

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
                                                            latent_sampling_method=latent_sampling_method)

    latent_sampling_method = ds.latent_sampling_method

    dl = PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method,
                                                                  batch_size=ds.batch_size, pin_memory=pin_memory)

    if unload:
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    weights = hypernetwork.weights(True)

    # Here we use optimizer from saved HN, or we can specify as UI option.
    if hypernetwork.optimizer_name in optimizer_dict:
        if use_adamw_parameter:
            if hypernetwork.optimizer_name != 'AdamW':
                raise RuntimeError(f"Cannot use adamW paramters for optimizer {hypernetwork.optimizer_name}!")
            optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate, **adamW_kwarg_dict)
        else:
            optimizer = optimizer_dict[hypernetwork.optimizer_name](params=weights, lr=scheduler.learn_rate)
        optimizer_name = hypernetwork.optimizer_name
    else:
        print(f"Optimizer type {hypernetwork.optimizer_name} is not defined!")
        optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate, **adamW_kwarg_dict)
        optimizer_name = 'AdamW'

    if hypernetwork.optimizer_state_dict:  # This line must be changed if Optimizer type can be different from saved optimizer.
        try:
            optimizer.load_state_dict(hypernetwork.optimizer_state_dict)
            optim_to(optimizer, devices.device)
        except RuntimeError as e:
            print("Cannot resume from saved optimizer!")
            print(e)
    if use_beta_scheduler:
        scheduler_beta = CosineAnnealingWarmUpRestarts(optimizer=optimizer, first_cycle_steps=beta_repeat_epoch, cycle_mult=epoch_mult, max_lr=scheduler.learn_rate, warmup_steps=warmup, min_lr=min_lr, gamma=gamma_rate)
        scheduler_beta.last_epoch =hypernetwork.step-1
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
                    loss = shared.sd_model(x, c)[0]
                    for filenames in batch.filename:
                        loss_dict[filenames].append(loss.item())
                    loss /= gradient_step
                    del x
                    del c

                    _loss_step += loss.item()
                    scaler.scale(loss).backward()
                    batch.latent_sample.to(devices.cpu)
                    del loss
                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue
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

                pbar.set_description(f"[Epoch {epoch_num}: {epoch_step + 1}/{steps_per_epoch}]loss: {loss_step:.7f}")
                if hypernetwork_dir is not None and ((use_beta_scheduler and scheduler_beta.is_EOC(hypernetwork.step) and save_when_converge) or (save_hypernetwork_every > 0 and steps_done % save_hypernetwork_every == 0)):
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
                                                 "learn_rate": optimizer.param_groups[0]['lr']
                                             })

                if images_dir is not None and (use_beta_scheduler and scheduler_beta.is_EOC(hypernetwork.step) and create_when_converge) or (create_image_every > 0 and steps_done % create_image_every == 0):
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
                        p.prompt = batch.cond_text[0]
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt

                    processed = processing.process_images(p)
                    image = processed.images[0] if len(processed.images) > 0 else None

                    if unload:
                        shared.sd_model.cond_stage_model.to(devices.cpu)
                        shared.sd_model.first_stage_model.to(devices.cpu)
                    torch.set_rng_state(rng_state)
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    hypernetwork.train()
                    if move_optimizer:
                        optim_to(optimizer, devices.device)
                    if image is not None:
                        shared.state.current_image = image
                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt,
                                                                             shared.opts.samples_format,
                                                                             processed.infotexts[0], p=p,
                                                                             forced_filename=forced_filename,
                                                                             save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"

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