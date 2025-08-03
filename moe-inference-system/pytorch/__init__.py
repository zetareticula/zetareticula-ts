import torch
from torch import nn
import os
import json
from typing import Dict, Optional, Tuple, Union
import numpy as np

class ZetaReticulaWrapper(nn.Module):
    """
    PyTorch wrapper for Zeta Reticula TS MoE engine.
    Enables seamless integration with PyTorch models.
    """
    
    def __init__(
        self,
        ts_module_path: str = "dist/integration/diffusion_transformer.js",
        num_experts: int = 8,
        attention_heads: int = 12,
        embedding_dim: int = 768,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.ts_module_path = os.path.abspath(ts_module_path)
        self.num_experts = num_experts
        self.attention_heads = attention_heads
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        
        # Load the compiled Node.js module
        self._init_node_bridge()
        
    def _init_node_bridge(self):
        """Initialize the Node.js bridge for TypeScript interop."""
        try:
            import nodebridge
            self._node = nodebridge.NodeProcess()
            
            # Load the TypeScript module
            self._node.eval(f'''
            const {{ ZetaDiffusionEngine }} = require('{self.ts_module_path}');
            const engine = new ZetaDiffusionEngine({{
                numExperts: {self.num_experts},
                attentionHeads: {self.attention_heads},
                embeddingDim: {self.embedding_dim}
            }});
            
            global.quantizeLatents = async (tensor) => {{
                const result = await engine.quantizeLatents(tensor);
                return result;
            }};
            
            global.crossAttention = async (x, context) => {{
                const result = await engine.crossAttention(x, context);
                return result;
            }};
            ''')
            
        except ImportError:
            raise ImportError(
                "nodebridge package is required. Install with: "
                "pip install nodebridge"
            )
    
    def forward(
        self, 
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the Zeta Reticula engine.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            context: Optional context tensor for cross-attention
            
        Returns:
            Processed tensor(s)
        """
        # Convert tensors to CPU for Node.js bridge
        x_np = x.cpu().numpy()
        
        if context is not None:
            ctx_np = context.cpu().numpy()
            # Call cross-attention
            result = self._node.call(
                'crossAttention',
                {"data": x_np.tolist(), "shape": list(x.shape)},
                {"data": ctx_np.tolist(), "shape": list(context.shape)}
            )
        else:
            # Call quantization
            result = self._node.call(
                'quantizeLatents',
                {"data": x_np.tolist(), "shape": list(x.shape)}
            )
        
        # Convert back to PyTorch tensor
        output = torch.tensor(
            result['data'], 
            dtype=x.dtype, 
            device=x.device
        ).view(*result['shape'])
        
        return output
    
    def measure_alignment(
        self, 
        modality_a: torch.Tensor, 
        modality_b: torch.Tensor
    ) -> float:
        """Measure alignment between two modalities using GW distance."""
        # This uses the Python implementation directly for better performance
        from ..src.evaluation.gromov_wasserstein import gromovWassersteinDistance
        
        return gromovWassersteinDistance(
            {"data": modality_a.cpu().numpy(), "modality": "A"},
            {"data": modality_b.cpu().numpy(), "modality": "B"}
        )

# Example usage in a PyTorch model
class ZetaDiffusionWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.zeta_engine = ZetaReticulaWrapper(**kwargs)
        
    def forward(self, x, context=None):
        return self.zeta_engine(x, context)
