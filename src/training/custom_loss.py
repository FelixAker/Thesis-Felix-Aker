"""
The selective knowledge distillation loss (Section 3.5 of the thesis).

Combines cross-entropy on the ground-truth token with a top-K KL divergence
against the cached teacher distribution. The KD term contributes only where the
teacher entropy exceeds the threshold, and is normalized over all valid tokens
rather than over the active ones, so raising the threshold removes signal
instead of re-weighting what is left (Section 3.6).
"""

import torch
import torch.nn as nn

class EntropySelectiveKDLoss(nn.Module):
    """
    Custom loss combining Cross-Entropy and Top-K KL Divergence weighted by an entropy mask.
    """
    def __init__(self, alpha=0.5, temperature=1.0, baseline_mode="selective", entropy_threshold=1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.baseline_mode = baseline_mode
        self.entropy_threshold = entropy_threshold
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, student_logits, teacher_topk_logits, teacher_topk_indices, targets, entropy_score, is_premask=False):
        """
        Compute the selective distillation loss.
        
        Args:
            student_logits: Tensor of shape (batch_size, seq_len, vocab_size)
            teacher_topk_logits: Tensor of shape (batch_size, seq_len, K)
            teacher_topk_indices: Tensor of shape (batch_size, seq_len, K) of int64
            targets: Tensor of shape (batch_size, seq_len) with true labels (-100 for padding)
            entropy_score: Tensor of shape (batch_size, seq_len) — either continuous entropy
                           values (new datasets, is_premask=False) or pre-binarized 0/1 mask
                           (old datasets, is_premask=True).
            is_premask: If True, entropy_score is already a binary mask — skip thresholding.
            
        Returns:
            total_loss: Scalar loss tensor combining CE and selective KD loss.
        """
        # Shift so that tokens < n predict n
        student_logits = student_logits[..., :-1, :].contiguous()
        targets = targets[..., 1:].contiguous()
        
        # Teacher logits and entropy masks were computed from the same aligned inputs
        # so they also need to be shifted to match the targets.
        teacher_topk_logits = teacher_topk_logits[..., :-1, :].contiguous()
        teacher_topk_indices = teacher_topk_indices[..., :-1, :].contiguous()
        entropy_score = entropy_score[..., :-1].contiguous()

        B, L, V = student_logits.shape
        flat_student_logits = student_logits.view(-1, V)
        flat_targets = targets.view(-1)
        
        # Valid tokens mask (ignore -100 padding)
        valid_mask = flat_targets != -100
        
        # 1. Standard Cross-Entropy Loss
        ce_loss = self.ce_loss(flat_student_logits, flat_targets)
        ce_loss = ce_loss * valid_mask.float()
        mean_ce_loss = ce_loss.sum() / (valid_mask.float().sum() + 1e-8)
        
        # 2. Top-K KL Divergence Loss
        # Compute teacher probabilities over the Top-K subset
        # IMPORTANT: teacher_topk_logits were saved as fp16 and can contain inf (fp16 max ~65504).
        # nan_to_num replaces inf/-inf with finite values before softmax to prevent nan.
        teacher_topk_logits = teacher_topk_logits.float()
        teacher_topk_logits = torch.nan_to_num(teacher_topk_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        teacher_probs = torch.softmax(teacher_topk_logits / self.temperature, dim=-1)
        
        # Compute student log probabilities over the entire vocabulary
        # Use float32 to avoid bfloat16 underflow on 256k softmax.
        student_log_probs_full = torch.log_softmax(student_logits.float() / self.temperature, dim=-1)
        
        # Gather student log probabilities corresponding to teacher Top-K indices
        # Index must be int64 for torch.gather.
        student_log_probs_topk = torch.gather(student_log_probs_full, dim=-1, index=teacher_topk_indices.long())
        # Clamp to prevent -inf (sparse 256k softmax) causing 0 * inf = nan in KL formula.
        student_log_probs_topk = student_log_probs_topk.clamp(min=-100.0)
        
        # KL Divergence per token: sum over K dimension
        teacher_log_probs = torch.log(teacher_probs + 1e-8)
        teacher_log_probs = teacher_log_probs.clamp(min=-100.0)
        kd_loss_per_token = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs_topk), dim=-1)
        # Final safety: replace any residual nan/inf with 0
        kd_loss_per_token = torch.nan_to_num(kd_loss_per_token, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale by temperature squared to keep gradient magnitudes consistent
        kd_loss_per_token = kd_loss_per_token * (self.temperature ** 2)
        
        # Flatten kd_loss and apply valid_mask and entropy_mask
        flat_kd_loss = kd_loss_per_token.view(-1)
        
        # Apply dynamic thresholding to entropy_score, or use pre-binarized mask directly
        if self.baseline_mode == "full_kd":
            entropy_mask = torch.ones_like(entropy_score)
        elif is_premask:
            # Old dataset format: mask already binarized at Phase 1 time, use as-is
            entropy_mask = entropy_score.float()
        else:
            entropy_mask = (entropy_score > self.entropy_threshold).float()
            
        flat_entropy_mask = entropy_mask.view(-1).float()
        
        # Only apply KD loss where valid_mask is true AND entropy_mask is active
        selective_kd_loss = flat_kd_loss * flat_entropy_mask * valid_mask.float()
        
        # Average over valid tokens that are active in the entropy mask
        # Average over ALL valid tokens so the KD gradient is properly scaled
        mean_kd_loss = selective_kd_loss.sum() / (valid_mask.float().sum() + 1e-8)
        
        # Combine losses
        total_loss = self.alpha * mean_ce_loss + (1.0 - self.alpha) * mean_kd_loss
        
        total_valid_tokens = valid_mask.float().sum()
        active_kd_tokens = (flat_entropy_mask * valid_mask.float()).sum()
        kd_fraction = active_kd_tokens / (total_valid_tokens + 1e-8)
        
        # Return detached TENSORS, not Python floats. Calling .item() here forced
        # four CUDA synchronisations on every micro-batch, stalling the pipeline
        # even though these values are only read every logging_steps. The caller
        # converts them (see train_student.py) only when it actually logs.
        metrics = {
            "train/ce_loss": mean_ce_loss.detach(),
            "train/kd_loss": mean_kd_loss.detach(),
            "train/kd_fraction": kd_fraction.detach(),
            "train/active_kd_tokens": active_kd_tokens.detach()
        }
        
        return total_loss, metrics
