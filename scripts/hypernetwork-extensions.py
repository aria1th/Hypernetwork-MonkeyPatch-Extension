import os

from modules.call_queue import wrap_gradio_call
from modules.hypernetworks.ui import keys
import modules.scripts as scripts
from modules import script_callbacks, shared
import gradio as gr

from modules.ui import  gr_show
import patches.clip_hijack as clip_hijack
import patches.textual_inversion as textual_inversion
import patches.ui as ui
import patches.shared as shared_patch
import patches.external_pr.ui as external_patch_ui

setattr(shared.opts,'pin_memory', False)

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
        skip_connection = gr.Checkbox(label="Use skip-connection. Won't work without extension!")
        optional_info = gr.Textbox("", label="Optional information about Hypernetwork", placeholder="Training information, dateset, etc")
        overwrite_old_hypernetwork = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork")

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(value="")

            with gr.Column():
                create_hypernetwork = gr.Button(value="Create hypernetwork", variant='primary')
        setting_name = gr.Textbox(label="Setting file name", value="")
        save_setting = gr.Button(value="Save hypernetwork setting to file")
        ti_output = gr.Text(elem_id="ti_output2", value="", show_label=False)
        ti_outcome = gr.HTML(elem_id="ti_error2", value="")



        save_setting.click(
            fn=wrap_gradio_call(external_patch_ui.save_hypernetwork_setting),
            inputs=[
                setting_name,
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
                normal_std if normal_std.visible else 0.01,
                skip_connection],
            outputs=[
                ti_output,
                ti_outcome,
            ]
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
                normal_std if normal_std.visible else 0.01,
                skip_connection
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
            with gr.Row():
                def track_vram_usage(*args):
                    import torch
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                if obj.is_cuda:
                                    print(type(obj), obj.size())
                        except: pass
                track_vram_usage_button = gr.Button(value="Track VRAM usage")
                track_vram_usage_button.click(
                    fn = track_vram_usage,
                    inputs=[],
                    outputs=[]
                )
    return [(CLIP_test_interface, "CLIP_test", "clip_test")]

def on_ui_settings():
    shared.opts.add_option("disable_ema",
        shared.OptionInfo(False, "Detach grad from conditioning models",
        section=('training', "Training")))
    if not hasattr(shared.opts, 'training_enable_tensorboard'):
        shared.opts.add_option("training_enable_tensorboard",
                               shared.OptionInfo(False, "Enable tensorboard logging",
                                                 section=('training', "Training")))

#script_callbacks.on_ui_train_tabs(create_training_tab)   # Deprecate Beta Training
script_callbacks.on_ui_train_tabs(create_extension_tab)
script_callbacks.on_ui_train_tabs(external_patch_ui.on_train_gamma_tab)
script_callbacks.on_ui_train_tabs(external_patch_ui.on_train_tuning)
script_callbacks.on_ui_tabs(create_extension_tab2)
script_callbacks.on_ui_settings(on_ui_settings)
class Script(scripts.Script):
    def title(self):
        return "Hypernetwork Monkey Patch"

    def show(self, _):
        return scripts.AlwaysVisible
