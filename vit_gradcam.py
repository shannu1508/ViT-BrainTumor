import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from typing import List, Optional

# Try to import the model class from local transformer implementation
from transformer import TumorClassifierViT


class GradCAM:
    """Grad-CAM for Vision Transformer using attention rollout and gradients.

    - Hooks all self-attention "attend" modules from `model.vit.transformer.layers`.
    - Captures attention maps for each layer during forward pass.
    - Captures gradients of the last attention layer via a backward hook.
    - Computes attention rollout across all layers (Abnar & Zuidema).
    - Averages across heads as required.
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()

        # Storage for attention maps and gradients across forward/backward
        self.attentions: List[torch.Tensor] = []
        self.last_attn_grads: Optional[torch.Tensor] = None
        self.handles = []

        # Register hooks on attention modules
        self._register_attention_hooks()

    def _find_attention_modules(self) -> List[torch.nn.Module]:
        """Discover attention modules in the ViT implementation.

        The provided `transformer.py` uses `self.vit.transformer.layers[i][0].attend`.
        We try this pattern first and fallback to scanning modules for attribute `attend`.
        """
        attn_modules = []
        try:
            layers = getattr(self.model.vit.transformer, "layers")
            for i, layer in enumerate(layers):
                # many ViT implementations pack attention into layer[0].attend
                try:
                    attn = layer[0].attend
                    attn_modules.append(attn)
                except Exception:
                    # fallback: scan submodules
                    for m in layer.modules():
                        if hasattr(m, "attend"):
                            attn_modules.append(getattr(m, "attend"))
        except Exception:
            # Generic fallback: scan entire model for modules with attribute 'attend'
            for m in self.model.modules():
                if hasattr(m, "attend"):
                    attn_modules.append(getattr(m, "attend"))

        # Last resort: include modules that are instances of torch.nn.MultiheadAttention
        if not attn_modules:
            for m in self.model.modules():
                if isinstance(m, torch.nn.MultiheadAttention):
                    attn_modules.append(m)

        return attn_modules

    def _register_attention_hooks(self):
        """Attach forward hooks to store attention maps and a backward hook on the last-attention output to capture gradients."""
        attn_modules = self._find_attention_modules()
        if not attn_modules:
            raise RuntimeError("No attention modules found in model to hook.")

        # forward hook: capture attention outputs (probabilities)
        def forward_hook(module, input, output):
            # Robustly find an attention tensor inside the output (handles Tensor or tuple/list)
            attn_tensor = None
            if isinstance(output, torch.Tensor) and getattr(output, 'ndim', 0) >= 2:
                attn_tensor = output
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor) and getattr(item, 'ndim', 0) >= 2:
                        attn_tensor = item
                        break

            # fallback: sometimes attention maps can be in the input to the module
            if attn_tensor is None and isinstance(input, (list, tuple)):
                for item in input:
                    if isinstance(item, torch.Tensor) and getattr(item, 'ndim', 0) >= 2:
                        attn_tensor = item
                        break

            if attn_tensor is None:
                return

            # store a detached CPU copy for later analysis
            try:
                self.attentions.append(attn_tensor.detach().cpu())
            except Exception:
                # as a last resort convert to tensor then store
                self.attentions.append(torch.tensor(attn_tensor).detach().cpu())

            # On the last attention module, register a backward hook to capture gradients
            if module is attn_modules[-1]:
                def _backward_hook(grad):
                    self.last_attn_grads = grad.detach().cpu()

                try:
                    attn_tensor.register_hook(_backward_hook)
                except Exception:
                    pass

        # Register forward hooks for all attention modules
        for m in attn_modules:
            try:
                h = m.register_forward_hook(forward_hook)
                self.handles.append(h)
            except Exception:
                # ignore modules that don't accept hooks
                pass

    def _clear(self):
        self.attentions = []
        self.last_attn_grads = None

    def _remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _prep_image(self, pil_image: Image.Image):
        """Convert PIL image to model-input tensor using common ImageNet normalization."""
        # Ensure RGB
        img = pil_image.convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        # Normalize with ImageNet statistics (common default for ViT pretrained models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # to CHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def compute_attention_rollout(self, discard_ratio: float = 0.0) -> np.ndarray:
        """Compute attention rollout across all captured attention maps.

        - Expects `self.attentions` to contain attention tensors from each layer in forward order.
        - Each attention tensor can have shape (B, heads, tokens, tokens) or (heads, tokens, tokens).
        - We average across heads for each layer and multiply (I + A) across layers.
        """
        if not self.attentions:
            raise RuntimeError("No attentions captured. Run a forward pass first.")

        # Convert all attention tensors to numpy with shape (tokens, tokens)
        attn_mats = []
        for a in self.attentions:
            at = a.numpy()
            # handle shapes: (B, H, T, T) or (H, T, T) or (T, T)
            if at.ndim == 4:
                # take batch 0
                at = at[0]
            if at.ndim == 3:
                # average across heads
                at = at.mean(axis=0)
            # at is now (T, T)
            attn_mats.append(at)

        # Optionally zero out small attentions per layer (discard_ratio)
        augmented = []
        for mat in attn_mats:
            # mat shape (T, T)
            if discard_ratio > 0:
                flat = mat.flatten()
                threshold = np.quantile(flat, discard_ratio)
                mat = np.where(mat < threshold, 0.0, mat)
            # add identity
            mat = mat + np.eye(mat.shape[0])
            # normalize rows
            mat = mat / mat.sum(axis=-1, keepdims=True)
            augmented.append(mat)

        # Multiply matrices (rollout): A_L * ... * A_1
        rollout = augmented[0]
        for mat in augmented[1:]:
            rollout = mat @ rollout

        return rollout

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Run forward/backward and produce heatmap resized to input image (224x224).

        Returns heatmap as float32 numpy array in range [0,1].
        """
        # Clear prev captures
        self._clear()

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass through model to get logits
        logits = self.model(input_tensor)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        if target_class is None:
            target_class = pred

        # Compute scalar score for target class and backprop
        score = logits[0, target_class]
        # Zero grads
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Now we should have attentions filled and last_attn_grads captured
        # Compute rollout (average across heads already handled in compute_attention_rollout)
        rollout = self.compute_attention_rollout()

        # Extract cls token row to get importance of each token to CLS
        # rollout shape: (T, T) where T = num_tokens (cls + patches)
        # We take rollout[0, 1:] to get influence from CLS to each patch
        cls_attention = rollout[0, 1:]

        # If gradient info from last attn exists, use it to modulate final map (simple weighting)
        if self.last_attn_grads is not None:
            g = self.last_attn_grads.numpy()
            # possible shapes: (B, H, T, T) or (H, T, T)
            if g.ndim == 4:
                g = g[0]
            if g.ndim == 3:
                g = g.mean(axis=0)  # average heads -> (T, T)
            # take positive gradients only and aggregate effect on CLS->patches
            g_pos = np.maximum(g, 0.0)
            weight = g_pos[0, 1:].mean(axis=0) if g_pos.ndim == 2 else g_pos[0, 1:]
            # combine by elementwise multiply then renormalize
            cls_attention = cls_attention * weight

        # Normalize
        mask = cls_attention
        mask = mask.reshape(int(np.sqrt(mask.shape[0])), -1)  # from (49,) -> (7,7)
        mask = mask - mask.min()
        if mask.max() > 0:
            mask = mask / mask.max()

        # Upsample to image size (224x224)
        mask_img = Image.fromarray(np.uint8(mask * 255)).resize((224, 224), resample=Image.BILINEAR)
        mask_arr = np.array(mask_img).astype(np.float32) / 255.0
        return mask_arr

    @staticmethod
    def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """Overlay heatmap (H,W in [0,1]) onto input PIL image and return resulting PIL image."""
        img = pil_img.convert("RGB").resize((224, 224))
        img_arr = np.array(img).astype(np.float32) / 255.0

        # Use matplotlib colormap
        cmap = plt.get_cmap("jet")
        colored_heat = cmap(heatmap)[:, :, :3]

        overlay = (1.0 - alpha) * img_arr + alpha * colored_heat
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray(np.uint8(overlay * 255))
        return overlay_img


def run_gradcam(image_path: str, model_path: str = "model.pth", save_path: str = None):
    """Load model, run Grad-CAM on an input image and show/save the overlay.

    Example usage: run_gradcam("test_mri.jpg")
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate model and load weights (assumes classifier uses 4 classes)
    model = TumorClassifierViT(num_classes=4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Prepare image
    pil = Image.open(image_path)

    gradcam = GradCAM(model, device=device)
    input_tensor = gradcam._prep_image(pil)

    heatmap = gradcam.generate_heatmap(input_tensor)
    overlay = GradCAM.overlay_heatmap_on_image(pil, heatmap, alpha=0.5)

    # Display with matplotlib
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(pil.convert('RGB'))

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.axis('off')
    plt.imshow(heatmap, cmap='jet')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.axis('off')
    plt.imshow(overlay)

    plt.tight_layout()

    if save_path:
        overlay.save(save_path)
    plt.show()


if __name__ == "__main__":
    # Simple test entrypoint. Replace the filename below as needed.
    test_image = "test_mri.jpg"
    if os.path.exists(test_image):
        run_gradcam(test_image)
    else:
        print(f"Place a test MRI named '{test_image}' next to this script or pass a path to run_gradcam().")
