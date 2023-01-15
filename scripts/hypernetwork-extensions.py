import os

from modules.call_queue import wrap_gradio_call
from modules.hypernetworks.ui import keys
import modules.scripts as scripts
from modules import script_callbacks, shared, sd_hijack
import gradio as gr

from modules.paths import script_path
from modules.ui import create_refresh_button, gr_show
import patches.clip_hijack as clip_hijack
import patches.textual_inversion as textual_inversion
import patches.ui as ui
import patches.shared as shared_patch
import patches.external_pr.ui as external_patch_ui
from webui import wrap_gradio_gpu_call

setattr(shared.opts,'pin_memory', False)


def create_training_tab(params: script_callbacks.UiTrainTabParams = None):
    with gr.Tab(label="Train_Beta") as train_beta:
        gr.HTML(
            value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
        with gr.Row():
            train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork",
                                                  choices=[x for x in shared.hypernetworks.keys()])
            create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks,
                                  lambda: {"choices": sorted([x for x in shared.hypernetworks.keys()])},
                                  "refresh_train_hypernetwork_name")
        with gr.Row():
            hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate',
                                                 placeholder="Hypernetwork Learning rate", value="0.00001")
            use_beta_scheduler_checkbox = gr.Checkbox(
                label='Show advanced learn rate scheduler options(for Hypernetworks)')
        with gr.Row(visible=False) as beta_scheduler_options:
            use_beta_scheduler = gr.Checkbox(label='Uses CosineAnnealingWarmRestarts Scheduler')
            beta_repeat_epoch = gr.Textbox(label='Epoch for cycle', placeholder="Cycles every nth epoch", value="4000")
            epoch_mult = gr.Textbox(label='Epoch multiplier per cycle', placeholder="Cycles length multiplier every cycle", value="1")
            warmup = gr.Textbox(label='Warmup step per cycle', placeholder="CosineAnnealing lr increase step", value="1")
            min_lr = gr.Textbox(label='Minimum learning rate for beta scheduler',
                                placeholder="restricts decay value, but does not restrict gamma rate decay",
                                value="1e-7")
            gamma_rate = gr.Textbox(label='Separate learning rate decay for ExponentialLR',
                                    placeholder="Value should be in (0-1]", value="1")
        batch_size = gr.Number(label='Batch size', value=1, precision=0)
        dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images")
        log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs",
                                   value="textual_inversion")
        template_file = gr.Textbox(label='Prompt template file',
                                   value=os.path.join(script_path, "textual_inversion_templates",
                                                      "style_filewords.txt"))
        training_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
        training_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
        steps = gr.Number(label='Max steps', value=100000, precision=0)
        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500,
                                       precision=0)
        save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable',
                                         value=500, precision=0)
        preview_from_txt2img = gr.Checkbox(
            label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False)

        with gr.Row():
            interrupt_training = gr.Button(value="Interrupt")
            train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary')
        ti_output = gr.Text(elem_id="ti_output2", value="", show_label=False)
        ti_outcome = gr.HTML(elem_id="ti_error2", value="")
        use_beta_scheduler_checkbox.change(
            fn=lambda show: gr_show(show),
            inputs=[use_beta_scheduler_checkbox],
            outputs=[beta_scheduler_options],
        )
    interrupt_training.click(
        fn=lambda: shared.state.interrupt(),
        inputs=[],
        outputs=[],
    )
    train_hypernetwork.click(
        fn=wrap_gradio_gpu_call(ui.train_hypernetwork_ui, extra_outputs=[gr.update()]),
        _js="start_training_textual_inversion",
        inputs=[
            train_hypernetwork_name,
            hypernetwork_learn_rate,
            batch_size,
            dataset_directory,
            log_directory,
            training_width,
            training_height,
            steps,
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
            gamma_rate
        ],
        outputs=[
            ti_output,
            ti_outcome,
        ]
    )
    return [(train_beta, "Train_beta", "train_beta")]

def create_extension_tab(params=None):
    with gr.Tab(label="Create Beta hypernetwork") as create_beta:
        new_hypernetwork_name = gr.Textbox(label="Name")
        new_hypernetwork_sizes = gr.CheckboxGroup(label="Modules", value=["768", "320", "640", "1024", "1280"],
                                                  choices=["768", "320", "640", "1024", "1280"])
        new_hypernetwork_layer_structure = gr.Textbox("1, 2, 1", label="Enter hypernetwork layer structure",
                                                      placeholder="1st and last digit must be 1. ex:'1, 2, 1'")
        new_hypernetwork_activation_func = gr.Dropdown(value="linear",
                                                       label="Select activation function of hypernetwork. Recommended : Swish / Linear(none)",
                                                       choices=keys)
        new_hypernetwork_initialization_option = gr.Dropdown(value="Normal",
                                                             label="Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise",
                                                             choices=["Normal", "KaimingUniform", "KaimingNormal",
                                                                      "XavierUniform", "XavierNormal"])
        show_additional_options = gr.Checkbox(
            label='Show advanced options')
        with gr.Row(visible=False) as weight_options:
            generation_seed = gr.Number(label='Weight initialization seed, set -1 for default', value=-1, precision=0)
            normal_std = gr.Textbox(label="Standard Deviation for Normal weight initialization", placeholder="must be positive float", value="0.01")
        show_additional_options.change(
            fn=lambda show: gr_show(show),
            inputs=[show_additional_options],
            outputs=[weight_options],)
        new_hypernetwork_add_layer_norm = gr.Checkbox(label="Add layer normalization")
        new_hypernetwork_use_dropout = gr.Checkbox(
            label="Use dropout. Might improve training when dataset is small / limited.")
        new_hypernetwork_dropout_structure = gr.Textbox("0, 0, 0",
                                                        label="Enter hypernetwork Dropout structure (or empty). Recommended : 0~0.35 incrementing sequence: 0, 0.05, 0.15",
                                                        placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'")
        optional_info = gr.Textbox("", label="Optional information about Hypernetwork", placeholder="Training information, dateset, etc")
        overwrite_old_hypernetwork = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork")

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(value="")

            with gr.Column():
                create_hypernetwork = gr.Button(value="Create hypernetwork", variant='primary')
            save_setting = gr.Button(value="Save hypernetwork setting to file")
            setting_name = gr.Textbox(label="Setting file name", value="")
            ti_output = gr.Text(elem_id="ti_output2", value="", show_label=False)
            ti_outcome = gr.HTML(elem_id="ti_error2", value="")

        save_setting.click(
            fn=wrap_gradio_call(external_patch_ui.save_hypernetwork_setting),
            inputs=[
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure,
                optional_info,
                generation_seed if generation_seed.visible else None,
                normal_std if normal_std.visible else 0.01],
            outputs=[]
        )
        create_hypernetwork.click(
            fn=ui.create_hypernetwork,
            inputs=[
                new_hypernetwork_name,
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure,
                optional_info,
                generation_seed if generation_seed.visible else None,
                normal_std if normal_std.visible else 0.01
            ],
            outputs=[
                new_hypernetwork_name,
                ti_output,
                ti_outcome,
            ]
        )
    return [(create_beta, "Create_beta", "create_beta")]


def create_extension_tab2(params=None):
    with gr.Blocks(analytics_enabled=False) as CLIP_test_interface:
        with gr.Tab(label="CLIP-test") as clip_test:
            with gr.Row():
                clipTextModelPath = gr.Textbox("openai/clip-vit-large-patch14", label="CLIP Text models. Set to empty to not change.")
                # see https://huggingface.co/openai/clip-vit-large-patch14 and related pages to find model.
                change_model = gr.Checkbox(label="Enable clip model change. This will be triggered from next model changes.")
                change_model.change(
                    fn=clip_hijack.trigger_sd_hijack,
                    inputs=[
                        change_model,
                        clipTextModelPath
                    ],
                    outputs=[]
                )
    return [(CLIP_test_interface, "CLIP_test", "clip_test")]

def on_ui_settings():
    shared.opts.add_option("disable_ema",
        shared.OptionInfo(False, "Detach grad from conditioning models",
        section=('training', "Training")))

#script_callbacks.on_ui_train_tabs(create_training_tab)   # Deprecate Beta Training
script_callbacks.on_ui_train_tabs(create_extension_tab)
script_callbacks.on_ui_train_tabs(external_patch_ui.on_train_gamma_tab)
script_callbacks.on_ui_tabs(create_extension_tab2)
script_callbacks.on_ui_settings(on_ui_settings)
class Script(scripts.Script):
    def title(self):
        return "Hypernetwork Monkey Patch"

    def show(self, _):
        return scripts.AlwaysVisible
