from modules import sd_hijack_clip, sd_hijack, shared
from modules.sd_hijack import StableDiffusionModelHijack, EmbeddingsWithFixes, apply_optimizations
try:
    from modules.sd_hijack import fix_checkpoint
    def clear_any_hijacks():
        StableDiffusionModelHijack.hijack = default_hijack
except (ModuleNotFoundError, ImportError):
    from modules.sd_hijack_checkpoint import add, remove
    def fix_checkpoint():
        add()

    def clear_any_hijacks():
        remove()
        StableDiffusionModelHijack.hijack = default_hijack


import ldm.modules.encoders.modules

default_hijack = StableDiffusionModelHijack.hijack

def trigger_sd_hijack(enabled, pretrained_key):
    clear_any_hijacks()
    if not enabled or pretrained_key == '':
        pretrained_key = 'openai/clip-vit-large-patch14'
    StableDiffusionModelHijack.hijack = create_lambda(pretrained_key)
    print("Hijacked clip text model!")
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    sd_hijack.model_hijack.hijack(shared.sd_model)
    if not enabled:
        StableDiffusionModelHijack.hijack = default_hijack




def create_lambda(model):
    def hijack_lambda(self, m):
        if type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            from transformers import CLIPTextModel, CLIPTokenizer
            print(f"Changing CLIP model to {model}")
            try:
                m.cond_stage_model.transformer = CLIPTextModel.from_pretrained(
                    model).to(m.cond_stage_model.transformer.device)
                m.cond_stage_model.transformer.requires_grad_(False)
                m.cond_stage_model.tokenizer = CLIPTokenizer.from_pretrained(
                    model)
            except:
                print(f"Cannot initiate from given model key {model}!")

            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

            self.optimization_method = apply_optimizations()

            self.clip = m.cond_stage_model

            fix_checkpoint()


            def flatten(el):
                flattened = [flatten(children) for children in el.children()]
                res = [el]
                for c in flattened:
                    res += c
                return res

            self.layers = flatten(m)
        else:
            print("CLIP change can be only applied to FrozenCLIPEmbedder class")
            return default_hijack(self, m)
    return hijack_lambda
