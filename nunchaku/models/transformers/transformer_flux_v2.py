from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)


class NunchakuFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, block: FluxSingleTransformerBlock):
        super(FluxSingleTransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm = block.norm
        self.proj_mlp = block.proj_mlp
        self.act_mlp = block.act_mlp
        self.proj_out = block.proj_out
        self.attn = block.attn


class NunchakuFluxTransformerBlock(FluxTransformerBlock):

    def __init__(self, block: FluxTransformerBlock):
        super(FluxTransformerBlock, self).__init__()

        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context
        self.attn = block.attn
        self.norm2 = block.norm2
        self.norm2_context = block.norm2_context
        self.ff = block.ff
        self.ff_context = block.ff_context


class NunchakuFluxTransformer2DModelV2(FluxTransformer2DModel):

    def _patch_model(self):
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFluxTransformerBlock(block)
        for i, block in enumerate(self.transformer_blocks_single):
            self.transformer_blocks_single[i] = NunchakuFluxSingleTransformerBlock(block)
