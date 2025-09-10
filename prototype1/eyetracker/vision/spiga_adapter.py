from __future__ import annotations
from typing import Dict, Any
import numpy as np

class SpigaAdapter:
    """
    Light wrapper that *would* call SPIGA (BMVC 2022) to get dense landmarks + head pose.
    This is a placeholder to keep the interface identical to MediaPipeIris.
    Implementers can pipe their SPIGA inference here and populate the same dict keys.
    """
    def __init__(self, device: str = "cpu", model_name: str = "spiga-face"):
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError("PyTorch not available. Install torch or use the MediaPipe backend.") from e
        
        
        # TODO: load SPIGA weights / model here when integrating the real package
       
       
        self.device = device
        self.model_name = model_name

    def process(self, frame_bgr) -> Dict[str, Any]:
        
        # TODO: replace with real SPIGA inference; below is a stub that reports not-ok.
        
        return {'ok': False, 'score': 0.0}
