from torch.utils.checkpoint import checkpoint


def BasicTransformerBlock_forward(self, x, context=None):
    return checkpoint(self._forward, x, context)

def AttentionBlock_forward(self, x):
    return checkpoint(self._forward, x)

def ResBlock_forward(self, x, emb):
    return checkpoint(self._forward, x, emb)


try:
    import ldm.modules.attention
    import ldm.modules.diffusionmodules.model
    import ldm.modules.diffusionmodules.openaimodel
    ldm.modules.attention.BasicTransformerBlock.forward = BasicTransformerBlock_forward
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = ResBlock_forward
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = AttentionBlock_forward
except:
    pass
