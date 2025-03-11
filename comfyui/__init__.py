# only import if running as a custom node

from .nodes.lora import SVDQuantFluxLoraLoader
from .nodes.models import SVDQuantFluxDiTLoader, SVDQuantTextEncoderLoader
from .nodes.preprocessors import FluxDepthPreprocessor

NODE_CLASS_MAPPINGS = {
    "SVDQuantFluxDiTLoader": SVDQuantFluxDiTLoader,
    "SVDQuantTextEncoderLoader": SVDQuantTextEncoderLoader,
    "SVDQuantFluxLoraLoader": SVDQuantFluxLoraLoader,
    "SVDQuantDepthPreprocessor": FluxDepthPreprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
