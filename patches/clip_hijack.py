from modules import sd_hijack_clip, sd_hijack
from modules.sd_hijack import StableDiffusionModelHijack, EmbeddingsWithFixes, apply_optimizations, fix_checkpoint
import ldm.modules.encoders.modules

default_hijack = StableDiffusionModelHijack.hijack

def trigger_sd_hijack(pretrained_key):
    clear_any_hijacks()
    StableDiffusionModelHijack.hijack = create_lambda(pretrained_key)
    print("Hijacked clip text model!")
    sd_hijack.model_hijack.undo_hijack(sd_model)
    sd_hijack.model_hijack.hijack(sd_model)

def clear_any_hijacks():
    StableDiffusionModelHijack.hijack = default_hijack

def create_lambda(model):
    def hijack_lambda(self, m):
        if type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            from transformers import CLIPTextModel, CLIPTokenizer
            print(f"Changing CLIP model to f{model}")
            try:
                m.cond_stage_model.transformer = CLIPTextModel.from_pretrained(
                    model).to(m.cond_stage_model.transformer.device)
                m.cond_stage_model.transformer.requires_grad_(False)
                m.cond_stage_model.tokenizer = CLIPTokenizer.from_pretrained(
                    model)
            except:
                print(f"Cannot initiate from given model key f{model}!")

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
