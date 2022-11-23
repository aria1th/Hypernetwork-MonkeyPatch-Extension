import csv
import datetime
import html
import os
import sys
import traceback

import torch
import tqdm
from PIL import PngImagePlugin

from .dataset import PersonalizedBase, PersonalizedDataLoader
from modules import shared, devices, sd_models, images, processing, sd_samplers, sd_hijack
from modules.textual_inversion.image_embedding import caption_image_overlay, insert_image_data_embed, embedding_to_b64
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from modules.textual_inversion.textual_inversion import save_embedding

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


def train_embedding(embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width,
                    training_height, steps, shuffle_tags, tag_drop_out, latent_sampling_method, create_image_every,
                    save_embedding_every, template_file, save_image_with_stored_embedding, preview_from_txt2img,
                    preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale,
                    preview_seed, preview_width, preview_height):
    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, steps,
                          save_embedding_every, create_image_every, log_directory, name="embedding")

    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    unload = shared.opts.unload_models_when_training

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
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

    pin_memory = shared.opts.pin_memory

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
        shared.sd_model.first_stage_model.to(devices.cpu)

    embedding.vec.requires_grad = True
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate)
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
                scheduler.apply(optimizer, embedding.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                with torch.autocast("cuda"):
                    # c = stack_conds(batch.cond).to(devices.device)
                    # mask = torch.tensor(batch.emb_index).to(devices.device, non_blocking=pin_memory)
                    # print(mask)
                    # c[:, 1:1+embedding.vec.shape[0]] = embedding.vec.to(devices.device, non_blocking=pin_memory)
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    c = shared.sd_model.cond_stage_model(batch.cond_text)
                    loss = shared.sd_model(x, c)[0] / gradient_step
                    del x

                    _loss_step += loss.item()
                scaler.scale(loss).backward()

                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue
                scaler.step(optimizer)
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
                if embedding_dir is not None and steps_done % save_embedding_every == 0:
                    # Before saving, change name to match current checkpoint.
                    embedding_name_every = f'{embedding_name}-{steps_done}'
                    last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
                    # if shared.opts.save_optimizer_state:
                    # embedding.optimizer_state_dict = optimizer.state_dict()
                    save_embedding(embedding, checkpoint, embedding_name_every, last_saved_file,
                                   remove_cached_checksum=True)
                    embedding_yet_to_be_embedded = True

                write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, steps_per_epoch, {
                    "loss": f"{loss_step:.7f}",
                    "learn_rate": scheduler.learn_rate
                })

                if images_dir is not None and steps_done % create_image_every == 0:
                    forced_filename = f'{embedding_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)

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

                    if unload:
                        shared.sd_model.first_stage_model.to(devices.cpu)

                    if image is not None:
                        shared.state.current_image = image
                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt,
                                                                             shared.opts.samples_format,
                                                                             processed.infotexts[0], p=p,
                                                                             forced_filename=forced_filename,
                                                                             save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"

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
                        footer_mid = '[{}]'.format(checkpoint.hash)
                        footer_right = '{}v {}s'.format(vectorSize, steps_done)

                        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                        captioned_image = insert_image_data_embed(captioned_image, data)

                        captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                        embedding_yet_to_be_embedded = False

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
        save_embedding(embedding, checkpoint, embedding_name, filename, remove_cached_checksum=True)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        pass
    finally:
        pbar.leave = False
        pbar.close()
        shared.sd_model.first_stage_model.to(devices.device)

    return embedding, filename