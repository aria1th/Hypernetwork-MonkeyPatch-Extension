import gc
import html
import json
import os
import random

from modules import shared, sd_hijack, devices
from modules.call_queue import wrap_gradio_call
from modules.paths import script_path
from modules.ui import create_refresh_button, gr_show
from webui import wrap_gradio_gpu_call
from .textual_inversion import train_embedding as train_embedding_external
from .hypernetwork import train_hypernetwork as train_hypernetwork_external, train_hypernetwork_tuning
import gradio as gr


def train_hypernetwork_ui(*args):
    initial_hypernetwork = None
    if hasattr(shared, 'loaded_hypernetwork'):
        initial_hypernetwork = shared.loaded_hypernetwork
    else:
        shared.loaded_hypernetworks = []
    assert not shared.cmd_opts.lowvram, 'Training models with lowvram is not possible'

    try:
        sd_hijack.undo_optimizations()

        hypernetwork, filename = train_hypernetwork_external(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {hypernetwork.step} steps.
Hypernetwork saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        if hasattr(shared, 'loaded_hypernetwork'):
            shared.loaded_hypernetwork = initial_hypernetwork
        else:
            shared.loaded_hypernetworks = []
        del hypernetwork
        gc.collect()
        shared.sd_model.cond_stage_model.to(devices.device)
        shared.sd_model.first_stage_model.to(devices.device)
        sd_hijack.apply_optimizations()


def train_hypernetwork_ui_tuning(*args):
    initial_hypernetwork = None
    if hasattr(shared, 'loaded_hypernetwork'):
        initial_hypernetwork = shared.loaded_hypernetwork
    else:
        shared.loaded_hypernetworks = []

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram is not possible'

    try:
        sd_hijack.undo_optimizations()

        train_hypernetwork_tuning(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'}.
"""
        return res, ""
    except Exception:
        raise
    finally:
        if hasattr(shared, 'loaded_hypernetwork'):
            shared.loaded_hypernetwork = initial_hypernetwork
        else:
            shared.loaded_hypernetworks = []
        shared.sd_model.cond_stage_model.to(devices.device)
        shared.sd_model.first_stage_model.to(devices.device)
        sd_hijack.apply_optimizations()


def save_training_setting(*args):
    save_file_name, learn_rate, batch_size, gradient_step, training_width, \
    training_height, steps, shuffle_tags, tag_drop_out, latent_sampling_method, \
    template_file, use_beta_scheduler, beta_repeat_epoch, epoch_mult, warmup, min_lr, \
    gamma_rate, use_beta_adamW_checkbox, save_when_converge, create_when_converge, \
    adamw_weight_decay, adamw_beta_1, adamw_beta_2, adamw_eps, show_gradient_clip_checkbox, \
    gradient_clip_opt, optional_gradient_clip_value, optional_gradient_norm_type, latent_sampling_std,\
    noise_training_scheduler_enabled, noise_training_scheduler_repeat, noise_training_scheduler_cycle, loss_opt = args
    dumped_locals = locals()
    dumped_locals.pop('args')
    filename = (str(random.randint(0, 1024)) if save_file_name == '' else save_file_name) + '_train_' + '.json'
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, filename)
    with open(filename, 'w') as file:
        print(dumped_locals)
        json.dump(dumped_locals, file)
        print(f"File saved as {filename}")
    return filename, ""


def save_hypernetwork_setting(*args):
    save_file_name, enable_sizes, overwrite_old, layer_structure, activation_func, weight_init, add_layer_norm, use_dropout, dropout_structure, optional_info, weight_init_seed, normal_std, skip_connection = args
    dumped_locals = locals()
    dumped_locals.pop('args')
    filename = (str(random.randint(0, 1024)) if save_file_name == '' else save_file_name) + '_hypernetwork_' + '.json'
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, filename)
    with open(filename, 'w') as file:
        print(dumped_locals)
        json.dump(dumped_locals, file)
        print(f"File saved as {filename}")
    return filename, ""


def on_train_gamma_tab(params=None):
    dummy_component = gr.Label(visible=False)
    with gr.Tab(label="Train_Gamma") as train_gamma:
        gr.HTML(
            value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
        with gr.Row():
            train_embedding_name = gr.Dropdown(label='Embedding', elem_id="train_embedding", choices=sorted(
                sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
            create_refresh_button(train_embedding_name,
                                  sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {
                    "choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())},
                                  "refresh_train_embedding_name")
        with gr.Row():
            train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork",
                                                  choices=[x for x in shared.hypernetworks.keys()])
            create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks,
                                  lambda: {"choices": sorted([x for x in shared.hypernetworks.keys()])},
                                  "refresh_train_hypernetwork_name")
        with gr.Row():
            embedding_learn_rate = gr.Textbox(label='Embedding Learning rate',
                                              placeholder="Embedding Learning rate", value="0.005")
            hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate',
                                                 placeholder="Hypernetwork Learning rate", value="0.00004")
            use_beta_scheduler_checkbox = gr.Checkbox(
                label='Show advanced learn rate scheduler options')
            use_beta_adamW_checkbox = gr.Checkbox(
                label='Show advanced adamW parameter options)')
            show_gradient_clip_checkbox = gr.Checkbox(
                label='Show Gradient Clipping Options(for both)')
            show_noise_options = gr.Checkbox(
                label='Show Noise Scheduler Options(for both)')
        with gr.Row(visible=False) as adamW_options:
            adamw_weight_decay = gr.Textbox(label="AdamW weight decay parameter", placeholder="default = 0.01",
                                            value="0.01")
            adamw_beta_1 = gr.Textbox(label="AdamW beta1 parameter", placeholder="default = 0.9", value="0.9")
            adamw_beta_2 = gr.Textbox(label="AdamW beta2 parameter", placeholder="default = 0.99", value="0.99")
            adamw_eps = gr.Textbox(label="AdamW epsilon parameter", placeholder="default = 1e-8", value="1e-8")
        with gr.Row(visible=False) as beta_scheduler_options:
            use_beta_scheduler = gr.Checkbox(label='Use CosineAnnealingWarmupRestarts Scheduler')
            beta_repeat_epoch = gr.Textbox(label='Steps for cycle', placeholder="Cycles every nth Step", value="64")
            epoch_mult = gr.Textbox(label='Step multiplier per cycle', placeholder="Step length multiplier every cycle",
                                    value="1")
            warmup = gr.Textbox(label='Warmup step per cycle', placeholder="CosineAnnealing lr increase step",
                                value="5")
            min_lr = gr.Textbox(label='Minimum learning rate',
                                placeholder="restricts decay value, but does not restrict gamma rate decay",
                                value="6e-7")
            gamma_rate = gr.Textbox(label='Decays learning rate every cycle',
                                    placeholder="Value should be in (0-1]", value="1")
        with gr.Row(visible=False) as beta_scheduler_options2:
            save_converge_opt = gr.Checkbox(label="Saves when every cycle finishes")
            generate_converge_opt = gr.Checkbox(label="Generates image when every cycle finishes")
        with gr.Row(visible=False) as gradient_clip_options:
            gradient_clip_opt = gr.Radio(label="Gradient Clipping Options", choices=["None", "limit", "norm"])
            optional_gradient_clip_value = gr.Textbox(label="Limiting value", value="1e-1")
            optional_gradient_norm_type = gr.Textbox(label="Norm type", value="2")
        with gr.Row(visible=False) as noise_scheduler_options:
            noise_training_scheduler_enabled = gr.Checkbox(label="Use Noise training scheduler(test)")
            noise_training_scheduler_repeat = gr.Checkbox(label="Restarts noise scheduler, or linear")
            noise_training_scheduler_cycle = gr.Number(label="Restarts noise scheduler every nth epoch")
        # change by feedback
        show_noise_options.change(
            fn = lambda show:gr_show(show),
            inputs = [show_noise_options],
            outputs = [noise_scheduler_options]
        )
        use_beta_adamW_checkbox.change(
            fn=lambda show: gr_show(show),
            inputs=[use_beta_adamW_checkbox],
            outputs=[adamW_options],
        )
        use_beta_scheduler_checkbox.change(
            fn=lambda show: gr_show(show),
            inputs=[use_beta_scheduler_checkbox],
            outputs=[beta_scheduler_options],
        )
        use_beta_scheduler_checkbox.change(
            fn=lambda show: gr_show(show),
            inputs=[use_beta_scheduler_checkbox],
            outputs=[beta_scheduler_options2],
        )
        show_gradient_clip_checkbox.change(
            fn=lambda show: gr_show(show),
            inputs=[show_gradient_clip_checkbox],
            outputs=[gradient_clip_options],
        )
        move_optim_when_generate = gr.Checkbox(label="Unload Optimizer when generating preview(hypernetwork)",
                                               value=True)
        batch_size = gr.Number(label='Batch size', value=1, precision=0)
        gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0)
        dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images")
        log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs",
                                   value="textual_inversion")
        template_file = gr.Textbox(label='Prompt template file',
                                   value=os.path.join(script_path, "textual_inversion_templates",
                                                      "style_filewords.txt"))
        training_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
        training_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
        steps = gr.Number(label='Max steps', value=100000, precision=0)
        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable',
                                       value=500, precision=0)
        save_embedding_every = gr.Number(
            label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0)
        save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True)
        preview_from_txt2img = gr.Checkbox(
            label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False)
        with gr.Row():
            shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False)
            tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.",
                                     value=0)
        with gr.Row():
            latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once",
                                              choices=['once', 'deterministic', 'random'])
            latent_sampling_std_value = gr.Number(label="Standard deviation for sampling", value=-1)
        with gr.Row():
            loss_opt = gr.Radio(label="loss type", value="loss",
                                choices=['loss', 'loss_simple', 'loss_vlb'])
        with gr.Row():
            save_training_option = gr.Button(value="Save training setting")
            save_file_name = gr.Textbox(label="File name to save setting as", value="")
            load_training_option = gr.Textbox(
                label="Load training option from saved json file. This will override settings above", value="")
        with gr.Row():
            interrupt_training = gr.Button(value="Interrupt")
            train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary')
            train_embedding = gr.Button(value="Train Embedding", variant='primary')
        ti_output = gr.Text(elem_id="ti_output3", value="", show_label=False)
        ti_outcome = gr.HTML(elem_id="ti_error3", value="")

    # Full path to .json or simple names are recommended.
    save_training_option.click(
        fn=wrap_gradio_call(save_training_setting),
        inputs=[
            save_file_name,
            hypernetwork_learn_rate,
            batch_size,
            gradient_step,
            training_width,
            training_height,
            steps,
            shuffle_tags,
            tag_drop_out,
            latent_sampling_method,
            template_file,
            use_beta_scheduler,
            beta_repeat_epoch,
            epoch_mult,
            warmup,
            min_lr,
            gamma_rate,
            use_beta_adamW_checkbox,
            save_converge_opt,
            generate_converge_opt,
            adamw_weight_decay,
            adamw_beta_1,
            adamw_beta_2,
            adamw_eps,
            show_gradient_clip_checkbox,
            gradient_clip_opt,
            optional_gradient_clip_value,
            optional_gradient_norm_type,
            latent_sampling_std_value,
        noise_training_scheduler_enabled,
        noise_training_scheduler_repeat,
        noise_training_scheduler_cycle,
        loss_opt],
        outputs=[
            ti_output,
            ti_outcome,
        ]
    )
    train_embedding.click(
        fn=wrap_gradio_gpu_call(train_embedding_external, extra_outputs=[gr.update()]),
        _js="start_training_textual_inversion",
        inputs=[
            dummy_component,
            train_embedding_name,
            embedding_learn_rate,
            batch_size,
            gradient_step,
            dataset_directory,
            log_directory,
            training_width,
            training_height,
            steps,
            shuffle_tags,
            tag_drop_out,
            latent_sampling_method,
            create_image_every,
            save_embedding_every,
            template_file,
            save_image_with_stored_embedding,
            preview_from_txt2img,
            *params.txt2img_preview_params,
            use_beta_scheduler,
            beta_repeat_epoch,
            epoch_mult,
            warmup,
            min_lr,
            gamma_rate,
            save_converge_opt,
            generate_converge_opt,
            move_optim_when_generate,
            use_beta_adamW_checkbox,
            adamw_weight_decay,
            adamw_beta_1,
            adamw_beta_2,
            adamw_eps,
            show_gradient_clip_checkbox,
            gradient_clip_opt,
            optional_gradient_clip_value,
            optional_gradient_norm_type,
            latent_sampling_std_value
        ],
        outputs=[
            ti_output,
            ti_outcome,
        ]
    )

    train_hypernetwork.click(
        fn=wrap_gradio_gpu_call(train_hypernetwork_ui, extra_outputs=[gr.update()]),
        _js="start_training_textual_inversion",
        inputs=[
            dummy_component,
            train_hypernetwork_name,
            hypernetwork_learn_rate,
            batch_size,
            gradient_step,
            dataset_directory,
            log_directory,
            training_width,
            training_height,
            steps,
            shuffle_tags,
            tag_drop_out,
            latent_sampling_method,
            create_image_every,
            save_embedding_every,
            template_file,
            preview_from_txt2img,
            *params.txt2img_preview_params,
            use_beta_scheduler,
            beta_repeat_epoch,
            epoch_mult,
            warmup,
            min_lr,
            gamma_rate,
            save_converge_opt,
            generate_converge_opt,
            move_optim_when_generate,
            use_beta_adamW_checkbox,
            adamw_weight_decay,
            adamw_beta_1,
            adamw_beta_2,
            adamw_eps,
            show_gradient_clip_checkbox,
            gradient_clip_opt,
            optional_gradient_clip_value,
            optional_gradient_norm_type,
            latent_sampling_std_value,
        noise_training_scheduler_enabled,
        noise_training_scheduler_repeat,
        noise_training_scheduler_cycle,
            load_training_option

        ],
        outputs=[
            ti_output,
            ti_outcome,
        ]
    )

    interrupt_training.click(
        fn=lambda: shared.state.interrupt(),
        inputs=[],
        outputs=[],
    )
    return [(train_gamma, "Train Gamma", "train_gamma")]


def on_train_tuning(params=None):
    dummy_component = gr.Label(visible=False)
    with gr.Tab(label="Train_Tuning") as train_tuning:
        gr.HTML(
            value="<p style='margin-bottom: 0.7em'>Train Hypernetwork; you must specify a directory <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
        with gr.Row():
            train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork",
                                                  choices=[x for x in shared.hypernetworks.keys()])
            create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks,
                                  lambda: {"choices": sorted([x for x in shared.hypernetworks.keys()])},
                                  "refresh_train_hypernetwork_name")
            optional_new_hypernetwork_name = gr.Textbox(
                label="Hypernetwork name to create, leave it empty to use selected", value="")
        with gr.Row():
            load_hypernetworks_option = gr.Textbox(
                label="Load Hypernetwork creation option from saved json file",
                placeholder=". filename cannot have ',' inside, and files should be splitted by ','.", value="")
        with gr.Row():
            load_training_options = gr.Textbox(
                label="Load training option(s) from saved json file",
                placeholder=". filename cannot have ',' inside, and files should be splitted by ','.", value="")
        move_optim_when_generate = gr.Checkbox(label="Unload Optimizer when generating preview(hypernetwork)",
                                               value=True)
        dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images")
        log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs",
                                   value="textual_inversion")
        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable',
                                       value=500, precision=0)
        save_model_every = gr.Number(
            label='Save a copy of model to log directory every N steps, 0 to disable', value=500, precision=0)
        preview_from_txt2img = gr.Checkbox(
            label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False)
        manual_dataset_seed = gr.Number(
            label="Manual dataset seed", value=-1, precision=0
        )
        with gr.Row():
            interrupt_training = gr.Button(value="Interrupt")
            train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary')
        ti_output = gr.Text(elem_id="ti_output4", value="", show_label=False)
        ti_outcome = gr.HTML(elem_id="ti_error4", value="")
    train_hypernetwork.click(
        fn=wrap_gradio_gpu_call(train_hypernetwork_ui_tuning, extra_outputs=[gr.update()]),
        _js="start_training_textual_inversion",
        inputs=[
            dummy_component,
            train_hypernetwork_name,
            dataset_directory,
            log_directory,
            create_image_every,
            save_model_every,
            preview_from_txt2img,
            *params.txt2img_preview_params,
            move_optim_when_generate,
            optional_new_hypernetwork_name,
            load_hypernetworks_option,
            load_training_options,
            manual_dataset_seed
        ],
        outputs=[
            ti_output,
            ti_outcome,
        ]
    )

    interrupt_training.click(
        fn=lambda: shared.state.interrupt(),
        inputs=[],
        outputs=[],
    )
    return [(train_tuning, "Train Tuning", "train_tuning")]
