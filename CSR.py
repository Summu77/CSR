import torch
import torch.nn as nn
import torch.fft
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Optional, Union, List, Tuple

DEFAULT_CONFIG = {
    "FILTER_TYPE": "gaussian", "LPF_RADIUS": 40, "DETECT_THRESH": 0.85,
    "PURIFY_STEPS": 3, "PURIFY_EPS": 4/255, "PURIFY_ALPHA": 2/255,
    "BUTTERWORTH_ORDER": 2
}

class RobustCLIP(nn.Module):
    def __init__(
        self, 
        model_name_or_obj: Union[str, CLIPModel] = "openai/clip-vit-base-patch32", 
        config: Dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.config = DEFAULT_CONFIG.copy()
        if config: self.config.update(config)
        
        # --- Load model ---
        if isinstance(model_name_or_obj, str):
            print(f"[RobustCLIP] Loading base model: {model_name_or_obj}...")
            self.base_model = CLIPModel.from_pretrained(model_name_or_obj).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj)
        else:
            self.base_model = model_name_or_obj.to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name_or_obj.config.name_or_path)

        self.clip_norm = T.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
        self.enable_defense = True
        
        # [New] Used to store detection results from the most recent inference
        self.last_detection_mask = None 

    # ================= Internal Logic =================

    def _apply_lpf(self, img_tensor):
        """Low-pass filter implementation"""
        B, C, H, W = img_tensor.shape
        radius = self.config["LPF_RADIUS"]
        
        fft = torch.fft.fft2(img_tensor)
        fft_shift = torch.fft.fftshift(fft)
        
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=self.device) - cy
        x = torch.arange(W, device=self.device) - cx
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        dist_sq = y_grid**2 + x_grid**2
        
        if self.config["FILTER_TYPE"] == 'gaussian':
            mask = torch.exp(-(dist_sq) / (2 * (radius**2)))
        elif self.config["FILTER_TYPE"] == 'butterworth':
            n = self.config["BUTTERWORTH_ORDER"]
            mask = 1.0 / (1.0 + (torch.sqrt(dist_sq) / radius)**(2 * n))
        else:
            mask = (dist_sq <= radius**2).float()
            
        f_filtered = fft_shift * mask.view(1, 1, H, W)
        img_back = torch.fft.ifft2(torch.fft.ifftshift(f_filtered))
        return torch.clamp(img_back.real, 0.0, 1.0)

    def _get_feats(self, img):
        normed = self.clip_norm(img)
        feats = self.base_model.get_image_features(normed)
        return feats / feats.norm(dim=-1, keepdim=True)
    
    def _detect_logic(self, pixel_values):
        """
        [New] Independent detection logic
        Returns: (is_adv_mask, lpf_feats)
        """
        orig_feats = self._get_feats(pixel_values)
        lpf_imgs = self._apply_lpf(pixel_values)
        lpf_feats = self._get_feats(lpf_imgs)
        
        sims = (orig_feats * lpf_feats).sum(dim=1)
        is_adv = sims < self.config["DETECT_THRESH"]
        return is_adv, lpf_feats

    def _purify(self, img_tensor, target_feats):

        # target_feats is already detached before being passed in, no need to worry about gradients
        with torch.enable_grad():
            img_in = img_tensor.detach().clone()
            img_in.requires_grad = True
            
            # Features of the original adversarial sample (as a repulsion target), need to detach
            orig_adv_feats = self._get_feats(img_in).detach()
            
            # --- [New] Initialize best result container ---
            best_img = img_tensor.detach().clone()
            # Set initial score to negative infinity (or calculate the score of the initial state)
            best_score = torch.full((img_tensor.shape[0],), -float('inf'), device=self.device)

            for _ in range(self.config["PURIFY_STEPS"]):
                # 1. Forward pass to compute current features
                curr_feats = self._get_feats(img_in)
                
                # 2. Calculate score (per-sample)
                # We want: close to target (sim_lpf large) and far from adversarial (sim_adv small)
                sim_lpf = (curr_feats * target_feats).sum(dim=1)
                sim_adv = (curr_feats * orig_adv_feats).sum(dim=1)
                current_score = sim_lpf - sim_adv
                # current_score = - sim_adv
                
                # 3. [Key optimization] Update best result
                # If the result of the current step is better than all previous ones, save it
                better_mask = current_score > best_score
                if better_mask.any():
                    best_score[better_mask] = current_score[better_mask].detach()
                    best_img[better_mask] = img_in[better_mask].detach()
                
                # 4. Calculate Loss and backpropagate
                # We want to maximize Score, i.e., minimize -Score
                loss = -current_score.sum()
                
                self.base_model.zero_grad()
                loss.backward()
                
                # 5. PGD update
                grad = img_in.grad.data
                img_in.data = img_in.data - self.config["PURIFY_ALPHA"] * grad.sign()
                
                # Project into epsilon ball (constrain near the original input image img_tensor)
                eta = torch.clamp(img_in.data - img_tensor.data, -self.config["PURIFY_EPS"], self.config["PURIFY_EPS"])
                img_in.data = torch.clamp(img_tensor.data + eta, 0.0, 1.0)
                img_in.grad = None
            
            # --- [New] Final check ---
            # The img_in at the end of the loop is the result of the last update, but the score hasn't been calculated yet
            # Perform an additional forward pass to ensure we don't miss potentially better results from the last step
            with torch.no_grad():
                final_feats = self._get_feats(img_in)
                sim_lpf = (final_feats * target_feats).sum(dim=1)
                sim_adv = (final_feats * orig_adv_feats).sum(dim=1)
                final_score = sim_lpf - sim_adv
                
                better_mask = final_score > best_score
                if better_mask.any():
                    best_img[better_mask] = img_in[better_mask]

        return best_img.detach()
        
    def process_images(self, pixel_values: torch.Tensor):
        
        if not self.enable_defense:
            self.last_detection_mask = torch.zeros(pixel_values.shape[0], dtype=torch.bool, device=self.device)
            return pixel_values

        pixel_values = pixel_values.to(self.device)
        
        # 1. Detect
        is_adv, lpf_feats = self._detect_logic(pixel_values)
        self.last_detection_mask = is_adv # Save state
        
        # 2. Purify (if no attack, return directly)
        if not is_adv.any():
            return pixel_values
        
        final_imgs = pixel_values.clone()
        idx = torch.where(is_adv)[0]
        
        purified = self._purify(pixel_values[idx], lpf_feats[idx])
        final_imgs[idx] = purified
        
        return final_imgs

    # ================= New External Interfaces =================

    def detect(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        [New feature] Explicitly detect whether it is an adversarial sample
        :param pixel_values: [B, 3, H, W] range [0, 1]
        :return: Boolean Tensor [B] (True=adversarial sample, False=normal)
        """
        pixel_values = pixel_values.to(self.device)
        is_adv, _ = self._detect_logic(pixel_values)
        return is_adv

    # ================= Compatible with CLIP API =================

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        """
        Automatic defense Forward
        Note: After calling, you can check which images in the batch were detected through model.last_detection_mask
        """
        if pixel_values is not None:
            pixel_values = self.process_images(pixel_values) # Will internally update last_detection_mask
            pixel_values = self.clip_norm(pixel_values)      # Normalize
        
        return self.base_model(input_ids=input_ids, pixel_values=pixel_values, **kwargs)

    def predict_zero_shot(self, image_tensor, text_labels):
        """
        Zero-Shot prediction + return attack detection results
        :return: (probs, logits, is_attack_mask)
        """
        text_inputs = self.processor(text=text_labels, return_tensors="pt", padding=True).to(self.device)
        
        # Run Forward (will automatically trigger process_images and update last_detection_mask)
        outputs = self.forward(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=image_tensor
        )
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Get the detection results just now from self.last_detection_mask
        is_attack = self.last_detection_mask
        
        return probs, logits_per_image, is_attack

    def get_image_features(self, pixel_values=None, **kwargs):
        clean_pixels = self.process_images(pixel_values)
        normed_pixels = self.clip_norm(clean_pixels)
        return self.base_model.get_image_features(pixel_values=normed_pixels, **kwargs)
        
    def get_text_features(self, input_ids=None, **kwargs):
        return self.base_model.get_text_features(input_ids=input_ids, **kwargs)
