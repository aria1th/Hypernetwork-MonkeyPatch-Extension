import csv
import datetime
import gc
import html
import os
import sys
import traceback

import torch
import tqdm
from PIL import PngImagePlugin

from .dataset import PersonalizedBase, PersonalizedDataLoader
from ..scheduler import CosineAnnealingWarmUpRestarts
from ..hnutil import optim_to

from modules import shared, devices, sd_models, images, processing, sd_samplers, sd_hijack
from modules.textual_inversion.image_embedding import caption_image_overlay, insert_image_data_embed, embedding_to_b64
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from modules.textual_inversion.textual_inversion import save_embedding

from torch.utils.tensorboard import SummaryWriter
from modules.textual_inversion.textual_inversion import tensorboard_add, tensorboard_setup, tensorboard_add_scaler, tensorboard_add_image
#apply OsError avoid here
delayed_values = {}

def write_loss(log_directory, filename, step, epoch_len, values):
    if shared.opts.training_write_csv_every == 0:
        return

    if step % shared.opts.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True
    try:
        with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
            csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

            if write_csv_header:
                csv_writer.writeheader()
            if log_directory + filename in delayed_values:
                delayed = delayed_values[log_directory + filename]
                for step, epoch, epoch_step, values in delayed:
                    csv_writer.writerow({
                        "step": step,
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        **values,
                    })
                delayed.clear()
            epoch, epoch_step = divmod(step - 1, epoch_len)
            csv_writer.writerow({
                "step": step,
                "epoch": epoch,
                "epoch_step": epoch_step,
                **values,
            })
    except OSError:
        epoch, epoch_step = divmod(step-1, epoch_len)
        if log_directory + filename in delayed_values:
            delayed_values[log_directory + filename].append((step , epoch, epoch_step, values))
        else:
            delayed_values[log_directory + filename] = [(step, epoch, epoch_step, values)]


def validate_train_inputs(model_name, learn_rate, batch_size, gradient_step, data_root, template_file, steps,
                          save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert isinstance(gradient_step, int), "Gradient accumulation step must be integer"
    assert gradient_step > 0, "Gradient accumulation step must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_file, "Prompt template file is empty"
    assert os.path.isfile(template_file), "Prompt template file doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0, "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0, "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0, "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"


def train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width,
                    training_height, steps, shuffle_tags, tag_drop_out, latent_sampling_method, create_image_every,
                    save_embedding_every, template_file, save_image_with_stored_embedding, preview_from_txt2img,
                    preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale,
                    preview_seed, preview_width, preview_height,
                    use_beta_scheduler=False, beta_repeat_epoch=4000, epoch_mult=1,warmup =10, min_lr=1e-7, gamma_rate=1, save_when_converge=False, create_when_converge=False,
                    move_optimizer=True,
                    use_adamw_parameter=False, adamw_weight_decay=0.01, adamw_beta_1=0.9, adamw_beta_2=0.99,adamw_eps=1e-8,
                    use_grad_opts=False, gradient_clip_opt='None', optional_gradient_clip_value=1e01, optional_gradient_norm_type=2
                    ):
    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, steps,
                          save_embedding_every, create_image_every, log_directory, name="embedding")
    try:
        if use_adamw_parameter:
            adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps = [float(x) for x in [adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps]]
            assert 0 <= adamw_weight_decay, "Weight decay paramter should be larger or equal than zero!"
            assert (all(0 <= x <= 1 for x in [adamw_beta_1, adamw_beta_2, adamw_eps])), "Cannot use negative or >1 number for adamW parameters!"
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
            def gradient_clipping(arg1):
                torch.nn.utils.clip_grad_norm_(arg1, optional_gradient_clip_value, optional_gradient_norm_type)
                return
        else:
            def gradient_clipping(arg1):
                torch.nn.utils.clip_grad_value_(arg1, optional_gradient_clip_value)
                return
    else:
        def gradient_clipping(arg1):
            return
    # Function gradient clipping is inplace(_) operation.
    shared.state.job = "train-embedding"
    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    unload = shared.opts.unload_models_when_training

    if save_embedding_every > 0 or save_when_converge:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0 or create_when_converge:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if (create_image_every > 0 or create_when_converge) and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()

    initial_step = embedding.step or 0
    if initial_step >= steps:
        shared.state.textinfo = f"Model has already been trained beyond specified max steps"
        return embedding, filename
    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)

    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    tensorboard_writer = None
    if shared.opts.training_enable_tensorboard:
        print("Tensorboard logging enabled")
        tensorboard_writer = tensorboard_setup(log_directory)

    pin_memory = shared.opts.pin_memory
    detach_grad = shared.opts.disable_ema # test code that removes EMA
    if detach_grad:
        print("Disabling training for staged models!")
        shared.sd_model.cond_stage_model.requires_grad_(False)
        shared.sd_model.first_stage_model.requires_grad_(False)
        torch.cuda.empty_cache()
    ds = PersonalizedBase(data_root=data_root, width=training_width,
                                                            height=training_height,
                                                            repeats=shared.opts.training_image_repeats_per_epoch,
                                                            placeholder_token=embedding_name, model=shared.sd_model,
                                                            cond_model=shared.sd_model.cond_stage_model,
                                                            device=devices.device, template_file=template_file,
                                                            batch_size=batch_size, gradient_step=gradient_step,
                                                            shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out,
                                                            latent_sampling_method=latent_sampling_method)

    latent_sampling_method = ds.latent_sampling_method

    dl = PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method,
                                                                  batch_size=ds.batch_size, pin_memory=pin_memory)
    if unload:
        shared.parallel_processing_allowed = False
        shared.sd_model.first_stage_model.to(devices.cpu)

    embedding.vec.requires_grad_(True)
    optimizer_name = 'AdamW' # hardcoded optimizer name now
    if use_adamw_parameter:
        optimizer = torch.optim.AdamW(params=[embedding.vec], lr=scheduler.learn_rate, **adamW_kwarg_dict)
    else:
        optimizer = torch.optim.AdamW(params=[embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)

    if os.path.exists(filename + '.optim'):  # This line must be changed if Optimizer type can be different from saved optimizer.
        try:
            optimizer_saved_dict = torch.load(filename + '.optim', map_location='cpu')
            if embedding.checksum() == optimizer_saved_dict.get('hash', None):
                optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
                if optimizer_state_dict is not None:
                    optimizer.load_state_dict(optimizer_state_dict)
                    print("Loaded existing optimizer from checkpoint")
        except RuntimeError as e:
            print("Cannot resume from saved optimizer!")
            print(e)
    else:
        print("No saved optimizer exists in checkpoint")
    if move_optimizer:
        optim_to(optimizer, devices.device)
    if use_beta_scheduler:
        scheduler_beta = CosineAnnealingWarmUpRestarts(optimizer=optimizer, first_cycle_steps=beta_repeat_epoch, cycle_mult=epoch_mult, max_lr=scheduler.learn_rate, warmup_steps=warmup, min_lr=min_lr, gamma=gamma_rate)
        scheduler_beta.last_epoch = embedding.step-1
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

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    is_training_inpainting_model = shared.sd_model.model.conditioning_key in {'hybrid', 'concat'}
    img_c = None

    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        for i in range((steps - initial_step) * gradient_step):
            if scheduler.finished:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                if use_beta_scheduler:
                    scheduler_beta.step(embedding.step)
                else:
                    scheduler.apply(optimizer, embedding.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                with devices.autocast():
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    shared.sd_model.cond_stage_model.to(devices.device)
                    c = shared.sd_model.cond_stage_model(batch.cond_text)
                    if is_training_inpainting_model:
                        if img_c is None:
                            img_c = processing.txt2img_image_conditioning(shared.sd_model, c, training_width, training_height)

                        cond = {"c_concat": [img_c], "c_crossattn": [c]}
                    else:
                        cond = c
                    loss = shared.sd_model(x, cond)[0] / gradient_step
                    del x
                    _loss_step += loss.item()
                scaler.scale(loss).backward()
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is None:
                            print("Found no grad!")
                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue
                gradient_clipping(embedding.vec)
                try:
                    scaler.step(optimizer)
                except AssertionError:
                    raise RuntimeError("This error happens because None of the template used embedding's text!")
                scaler.update()
                embedding.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = embedding.step + 1

                epoch_num = embedding.step // steps_per_epoch
                epoch_step = embedding.step % steps_per_epoch

                pbar.set_description(f"[Epoch {epoch_num}: {epoch_step + 1}/{steps_per_epoch}]loss: {loss_step:.7f}")
                if embedding_dir is not None and ((use_beta_scheduler and scheduler_beta.is_EOC(embedding.step) and save_when_converge) or (save_embedding_every > 0 and steps_done % save_embedding_every == 0)):
                    # Before saving, change name to match current checkpoint.
                    embedding_name_every = f'{embedding_name}-{steps_done}'
                    last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
                    # if shared.opts.save_optimizer_state:
                    # embedding.optimizer_state_dict = optimizer.state_dict()
                    save_embedding(embedding, optimizer, checkpoint, embedding_name_every, last_saved_file,
                                   remove_cached_checksum=True)
                    embedding_yet_to_be_embedded = True

                write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, steps_per_epoch, {
                    "loss": f"{loss_step:.7f}",
                    "learn_rate": scheduler.learn_rate
                })

                if images_dir is not None and ((use_beta_scheduler and scheduler_beta.is_EOC(embedding.step) and create_when_converge) or (create_image_every > 0 and steps_done % create_image_every == 0)):
                    forced_filename = f'{embedding_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)
                    rng_state = torch.get_rng_state()
                    cuda_rng_state = None
                    if torch.cuda.is_available():
                        cuda_rng_state = torch.cuda.get_rng_state_all()
                    if move_optimizer:
                        optim_to(optimizer, devices.cpu)
                        gc.collect()
                    shared.sd_model.first_stage_model.to(devices.device)

                    p = processing.StableDiffusionProcessingTxt2Img(
                        sd_model=shared.sd_model,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                        do_not_reload_embeddings=True,
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

                    if move_optimizer:
                        optim_to(optimizer, devices.device)
                    if image is not None:
                        shared.state.assign_current_image(image)
                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt,
                                                                             shared.opts.samples_format,
                                                                             processed.infotexts[0], p=p,
                                                                             forced_filename=forced_filename,
                                                                             save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"
                        if shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:
                            tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image,
                                                  embedding.step)

                    if save_image_with_stored_embedding and os.path.exists(
                            last_saved_file) and embedding_yet_to_be_embedded:

                        last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                        info = PngImagePlugin.PngInfo()
                        data = torch.load(last_saved_file)
                        info.add_text("sd-ti-embedding", embedding_to_b64(data))

                        title = "<{}>".format(data.get('name', '???'))

                        try:
                            vectorSize = list(data['string_to_param'].values())[0].shape[0]
                        except Exception as e:
                            vectorSize = '?'

                        checkpoint = sd_models.select_checkpoint()
                        footer_left = checkpoint.model_name
                        footer_mid = '[{}]'.format(checkpoint.shorthash if hasattr(checkpoint, 'shorthash') else checkpoint.hash)
                        footer_right = '{}v {}s'.format(vectorSize, steps_done)

                        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                        captioned_image = insert_image_data_embed(captioned_image, data)

                        captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                        embedding_yet_to_be_embedded = False
                    if unload:
                        shared.sd_model.first_stage_model.to(devices.cpu)
                    torch.set_rng_state(rng_state)
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt,
                                                                         shared.opts.samples_format,
                                                                         processed.infotexts[0], p=p,
                                                                         forced_filename=forced_filename,
                                                                         save_to_dirs=False)
                    last_saved_image += f", prompt: {preview_text}"


                shared.state.job_no = embedding.step

                shared.state.textinfo = f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
        filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
        save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        pass
    finally:
        pbar.leave = False
        pbar.close()
        shared.sd_model.first_stage_model.to(devices.device)
        shared.parallel_processing_allowed = old_parallel_processing_allowed
    return embedding, filename