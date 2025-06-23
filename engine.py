class EngineInterface:
    """Abstract base class for OCR engines."""

    def predict(self, images):
        raise NotImplementedError

from typing import List, Dict

try:
    import torch
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - optional dependency
    torch = None
    hf_pipeline = None


class HFEngine(EngineInterface):
    """Hugging Face OCR engine using TrOCR."""

    def __init__(self, model: str | None = None):
        if hf_pipeline is None:
            raise RuntimeError("transformers is not available")
        if model is None:
            try:
                import config
                model = getattr(config, "HF_MODEL", "microsoft/trocr-base-stage1")
            except Exception:
                model = "microsoft/trocr-base-stage1"
        self.model = model
        self.device = 0 if torch and torch.cuda.is_available() else -1
        dtype = torch.float16 if self.device == 0 else None
        self._init_pipeline(self.device, dtype)

    def _init_pipeline(self, device: int, dtype):
        self.pipe = hf_pipeline(
            "image-to-text",
            model=self.model,
            device=device,
            torch_dtype=dtype,
        )

    def predict(self, images: List) -> List[Dict]:
        out = []
        for img in images:
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            try:
                res = self.pipe(img)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg and self.device == 0:
                    if torch:
                        torch.cuda.empty_cache()
                    self.device = -1
                    self._init_pipeline(-1, None)
                    res = self.pipe(img)
                else:
                    raise
            text = res[0]["generated_text"] if res else ""
            out.append({"text": text, "boxes": []})
        return out
