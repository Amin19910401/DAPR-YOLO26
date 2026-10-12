# -*- coding: utf-8 -*-
"""
DAPR: Dynamic Adaptive Prototype-based Refinement
================================================================

Custom modules implementing the DAPR module and its ablation variants.

This file should be placed inside the Ultralytics package directory:

    ultralytics/nn/modules/dapr.py

and imported in:

    ultralytics/nn/modules/__init__.py
    ultralytics/nn/tasks.py

Classes:
    DAPR_only_v3      : Full DAPR module (main contribution)
    DAPR_noDelta      : Ablation without adaptive prototype offset
    DAPR_noPosition   : Ablation without positional embedding
    DAPR_noGate       : Ablation without confidence gate
    DAPR_noClassPart  : Ablation without class-part conditioning
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


__all__ = (
    "DAPR_only_v3",
    "DAPR_noDelta",
    "DAPR_noPosition",
    "DAPR_noGate",
    "DAPR_noClassPart",
)


# ============================================================
# Auto-fix for ModelEMA with THOP float64 counters
# ============================================================

def _dapr_auto_fix_ema():
    """Remove float64 buffers (from THOP) before EMA update."""
    try:
        from ultralytics.utils.torch_utils import ModelEMA

        if getattr(ModelEMA.update, "_dapr_patched", False):
            return

        _original_update = ModelEMA.update

        def _clean_float64(model):
            for m in model.modules():
                if hasattr(m, "_buffers"):
                    for key in list(m._buffers.keys()):
                        try:
                            v = m._buffers[key]
                            if v is not None and v.dtype == torch.float64:
                                del m._buffers[key]
                        except Exception:
                            pass
                if hasattr(m, "_parameters"):
                    for key in list(m._parameters.keys()):
                        try:
                            v = m._parameters[key]
                            if v is not None and v.dtype == torch.float64:
                                del m._parameters[key]
                        except Exception:
                            pass
                for key in ["total_ops", "total_params"]:
                    if key in m.__dict__:
                        try:
                            del m.__dict__[key]
                        except Exception:
                            pass

        def _patched_update(self, model):
            _clean_float64(model)
            if hasattr(self, "ema"):
                _clean_float64(self.ema)
            return _original_update(self, model)

        _patched_update._dapr_patched = True
        ModelEMA.update = _patched_update
        print("[DAPR-AUTO-FIX] ModelEMA.update patched successfully")
    except Exception as e:
        print(f"[DAPR-AUTO-FIX] Failed: {e}")


_dapr_auto_fix_ema()


# ============================================================
# DAPR_only_v3: Full DAPR module
# ============================================================

class DAPR_only_v3(nn.Module):
    """
    DAPR-only v3.

    Dynamic Adaptive Prototype-based Refinement module.

    Input:
        x : [B, c1, H, W]

    Output:
        y : [B, c2, H, W]

    Components:
        (1) Split-Process-Concat preprocessing
        (2) Soft part assignment and aggregation
        (3) Adaptive class-part-conditioned prototype generation
        (4) Prototype matching via temperature-scaled cross-attention
        (5) Spatial reconstruction and confidence-gated fusion
    """

    def __init__(
        self,
        c1,
        c2,
        num_classes=2,
        parts_per_class=4,
        proto_dim=48,
        reduction=4,
        temperature=0.50,
    ):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.num_classes = num_classes
        self.parts_per_class = parts_per_class
        self.num_parts = num_classes * parts_per_class
        self.proto_dim = proto_dim
        self.temperature = temperature

        # ---------- 1. Split-Process-Concat ----------
        self.c_half = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c_half, 1, 1)

        # ---------- 2. Internal representation ----------
        self.internal_dim = min(64, max(self.c_half // reduction, 32))
        self.input_proj = Conv(self.c_half, self.internal_dim, 1, p=0)
        hidden = max(self.internal_dim // 2, 32)

        # ---------- 3. Local encoder ----------
        self.local_encoder = nn.Sequential(
            Conv(self.internal_dim, hidden, 3, p=1),
            nn.ReLU(inplace=True),
            Conv(hidden, proto_dim, 1, p=0),
        )

        # ---------- 4. Part assignment ----------
        self.part_assignment = nn.Sequential(
            Conv(self.internal_dim, hidden, 3, p=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.num_parts, 1),
        )

        # ---------- 5. Position encoding ----------
        pos_hidden = max(proto_dim // 4, 8)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, proto_dim),
        )
        self._coords_cache = None
        self._cached_HW = None

        # ---------- 6. Class-aware base prototypes ----------
        self.base_prototypes = nn.Parameter(
            torch.empty(num_classes, parts_per_class, proto_dim)
        )
        nn.init.normal_(self.base_prototypes, mean=0.0, std=0.02)

        # ---------- 7. Adaptive prototype generator ----------
        self.prototype_generator = nn.Sequential(
            nn.Linear(proto_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.num_parts * proto_dim),
        )

        # ---------- 8. Matching projections ----------
        self.part_query = nn.Linear(proto_dim, proto_dim, bias=False)
        self.prototype_key = nn.Linear(proto_dim, proto_dim, bias=False)

        # ---------- 9. Class-part classifier ----------
        # Output: C * P logits per part slot
        self.class_classifier = nn.Linear(
            proto_dim, num_classes * parts_per_class
        )

        # ---------- 10. Depthwise reconstruction ----------
        self.reconstruction_proj = Conv(proto_dim, self.c_half, 1, p=0)
        self.reconstruction_dw = nn.Conv2d(
            self.c_half, self.c_half,
            kernel_size=3, padding=1, groups=self.c_half, bias=False,
        )
        self.reconstruction_bn = nn.BatchNorm2d(self.c_half)
        self.reconstruction_act = nn.ReLU(inplace=True)

        # ---------- 11. Confidence gate ----------
        gate_hidden = max(self.internal_dim // 4, 16)
        self.gate = nn.Sequential(
            Conv(self.internal_dim + proto_dim + 1, gate_hidden, 1, p=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1),
            nn.Sigmoid(),
        )

        # ---------- 12. Output projection ----------
        self.cv2 = Conv(2 * self.c_half, c2, 1, 1)
        self.norm = nn.BatchNorm2d(self.c_half)

    def forward(self, x):
        B, C1, H, W = x.shape
        device, dtype = x.device, x.dtype
        N = H * W

        # ---------- Split-Process ----------
        a, b = self.cv1(x).chunk(2, 1)

        # ---------- DAPR branch on b ----------
        x_internal = self.input_proj(b)

        local_feat = self.local_encoder(x_internal)
        D = local_feat.shape[1]
        local_flat = local_feat.flatten(2).transpose(1, 2)

        # Part assignment
        assign_logits = self.part_assignment(x_internal).flatten(2)
        assignment = F.softmax(assign_logits, dim=1)

        # Part pooling (unnormalized aggregation)
        part_features = torch.bmm(assignment, local_flat)

        # ---------- Position encoding (cached) ----------
        if self._cached_HW != (H, W):
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing="ij",
            )
            self._coords_cache = torch.stack([xx, yy], dim=-1).view(1, N, 2)
            self._cached_HW = (H, W)

        coords = self._coords_cache.to(device=device, dtype=dtype).expand(B, -1, -1)
        part_centers = torch.bmm(assignment, coords)
        pos_embed = self.position_mlp(part_centers)

        # ---------- Adaptive prototypes ----------
        inst_desc = part_features.mean(dim=1)
        delta = self.prototype_generator(inst_desc).view(
            B, self.num_parts, self.proto_dim
        )

        # ---------- Class-part conditioning ----------
        class_logits = self.class_classifier(part_features)
        class_logits = class_logits.view(
            B, self.num_parts, self.num_classes, self.parts_per_class
        )
        class_probs = F.softmax(
            class_logits.view(B, self.num_parts, -1), dim=-1
        ).view(B, self.num_parts, self.num_classes, self.parts_per_class)

        class_prototypes = torch.einsum(
            "bkcp,cpd->bkd", class_probs, self.base_prototypes
        )

        prototypes = class_prototypes + 0.10 * delta + 0.10 * pos_embed

        # ---------- Prototype matching ----------
        q = F.normalize(self.part_query(part_features), p=2, dim=-1)
        k = F.normalize(self.prototype_key(prototypes), p=2, dim=-1)
        sim = torch.matmul(q, k.transpose(1, 2)) / self.temperature
        proto_attn = F.softmax(sim, dim=-1)
        matched_parts = torch.matmul(proto_attn, prototypes)

        # ---------- Spatial reconstruction ----------
        assign_t = assignment.transpose(1, 2)
        spatial_feat = torch.bmm(assign_t, matched_parts)
        spatial_feat = spatial_feat.transpose(1, 2).contiguous().view(B, D, H, W)

        reconstructed = self.reconstruction_proj(spatial_feat)
        reconstructed = self.reconstruction_dw(reconstructed)
        reconstructed = self.reconstruction_act(
            self.reconstruction_bn(reconstructed)
        )

        # ---------- Class confidence spatial ----------
        class_probs_marginal = class_probs.sum(dim=-1)
        class_conf = class_probs_marginal.max(dim=-1, keepdim=True)[0]
        class_conf_spatial = torch.bmm(assign_t, class_conf)
        class_conf_spatial = class_conf_spatial.transpose(1, 2).contiguous().view(
            B, 1, H, W
        )

        # ---------- Confidence gate ----------
        confidence = self.gate(torch.cat(
            [x_internal, spatial_feat, class_conf_spatial], dim=1
        ))

        # ---------- Residual + Concat ----------
        b_out = b + confidence * reconstructed
        b_out = self.norm(b_out)

        return self.cv2(torch.cat([a, b_out], dim=1))


# ============================================================
# DAPR_noDelta: Ablation without adaptive prototype offset
# ============================================================

class DAPR_noDelta(DAPR_only_v3):
    """Ablation variant: adaptive prototype offset is disabled."""

    def __init__(self, c1, c2, num_classes=2, parts_per_class=4,
                 proto_dim=48, reduction=4, temperature=0.50):
        super().__init__(c1, c2, num_classes, parts_per_class,
                         proto_dim, reduction, temperature)

    def forward(self, x):
        B, C1, H, W = x.shape
        device, dtype = x.device, x.dtype
        N = H * W

        a, b = self.cv1(x).chunk(2, 1)
        x_internal = self.input_proj(b)
        local_feat = self.local_encoder(x_internal)
        D = local_feat.shape[1]
        local_flat = local_feat.flatten(2).transpose(1, 2)

        assign_logits = self.part_assignment(x_internal).flatten(2)
        assignment = F.softmax(assign_logits, dim=1)
        part_features = torch.bmm(assignment, local_flat)

        if self._cached_HW != (H, W):
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing="ij",
            )
            self._coords_cache = torch.stack([xx, yy], dim=-1).view(1, N, 2)
            self._cached_HW = (H, W)
        coords = self._coords_cache.to(device=device, dtype=dtype).expand(B, -1, -1)
        part_centers = torch.bmm(assignment, coords)
        pos_embed = self.position_mlp(part_centers)

        # Delta disabled
        class_logits = self.class_classifier(part_features)
        class_logits = class_logits.view(
            B, self.num_parts, self.num_classes, self.parts_per_class
        )
        class_probs = F.softmax(
            class_logits.view(B, self.num_parts, -1), dim=-1
        ).view(B, self.num_parts, self.num_classes, self.parts_per_class)
        class_prototypes = torch.einsum(
            "bkcp,cpd->bkd", class_probs, self.base_prototypes
        )

        prototypes = class_prototypes + 0.10 * pos_embed

        q = F.normalize(self.part_query(part_features), p=2, dim=-1)
        k = F.normalize(self.prototype_key(prototypes), p=2, dim=-1)
        sim = torch.matmul(q, k.transpose(1, 2)) / self.temperature
        proto_attn = F.softmax(sim, dim=-1)
        matched_parts = torch.matmul(proto_attn, prototypes)

        assign_t = assignment.transpose(1, 2)
        spatial_feat = torch.bmm(assign_t, matched_parts)
        spatial_feat = spatial_feat.transpose(1, 2).contiguous().view(B, D, H, W)

        reconstructed = self.reconstruction_proj(spatial_feat)
        reconstructed = self.reconstruction_dw(reconstructed)
        reconstructed = self.reconstruction_act(
            self.reconstruction_bn(reconstructed)
        )

        class_probs_marginal = class_probs.sum(dim=-1)
        class_conf = class_probs_marginal.max(dim=-1, keepdim=True)[0]
        class_conf_spatial = torch.bmm(assign_t, class_conf)
        class_conf_spatial = class_conf_spatial.transpose(1, 2).contiguous().view(
            B, 1, H, W
        )

        confidence = self.gate(torch.cat(
            [x_internal, spatial_feat, class_conf_spatial], dim=1
        ))

        b_out = b + confidence * reconstructed
        b_out = self.norm(b_out)
        return self.cv2(torch.cat([a, b_out], dim=1))


# ============================================================
# DAPR_noPosition: Ablation without positional embedding
# ============================================================

class DAPR_noPosition(DAPR_only_v3):
    """Ablation variant: positional embedding is disabled."""

    def __init__(self, c1, c2, num_classes=2, parts_per_class=4,
                 proto_dim=48, reduction=4, temperature=0.50):
        super().__init__(c1, c2, num_classes, parts_per_class,
                         proto_dim, reduction, temperature)

    def forward(self, x):
        B, C1, H, W = x.shape
        device, dtype = x.device, x.dtype
        N = H * W

        a, b = self.cv1(x).chunk(2, 1)
        x_internal = self.input_proj(b)
        local_feat = self.local_encoder(x_internal)
        D = local_feat.shape[1]
        local_flat = local_feat.flatten(2).transpose(1, 2)

        assign_logits = self.part_assignment(x_internal).flatten(2)
        assignment = F.softmax(assign_logits, dim=1)
        part_features = torch.bmm(assignment, local_flat)

        # Positional embedding disabled

        inst_desc = part_features.mean(dim=1)
        delta = self.prototype_generator(inst_desc).view(
            B, self.num_parts, self.proto_dim
        )

        class_logits = self.class_classifier(part_features)
        class_logits = class_logits.view(
            B, self.num_parts, self.num_classes, self.parts_per_class
        )
        class_probs = F.softmax(
            class_logits.view(B, self.num_parts, -1), dim=-1
        ).view(B, self.num_parts, self.num_classes, self.parts_per_class)
        class_prototypes = torch.einsum(
            "bkcp,cpd->bkd", class_probs, self.base_prototypes
        )

        prototypes = class_prototypes + 0.10 * delta

        q = F.normalize(self.part_query(part_features), p=2, dim=-1)
        k = F.normalize(self.prototype_key(prototypes), p=2, dim=-1)
        sim = torch.matmul(q, k.transpose(1, 2)) / self.temperature
        proto_attn = F.softmax(sim, dim=-1)
        matched_parts = torch.matmul(proto_attn, prototypes)

        assign_t = assignment.transpose(1, 2)
        spatial_feat = torch.bmm(assign_t, matched_parts)
        spatial_feat = spatial_feat.transpose(1, 2).contiguous().view(B, D, H, W)

        reconstructed = self.reconstruction_proj(spatial_feat)
        reconstructed = self.reconstruction_dw(reconstructed)
        reconstructed = self.reconstruction_act(
            self.reconstruction_bn(reconstructed)
        )

        class_probs_marginal = class_probs.sum(dim=-1)
        class_conf = class_probs_marginal.max(dim=-1, keepdim=True)[0]
        class_conf_spatial = torch.bmm(assign_t, class_conf)
        class_conf_spatial = class_conf_spatial.transpose(1, 2).contiguous().view(
            B, 1, H, W
        )

        confidence = self.gate(torch.cat(
            [x_internal, spatial_feat, class_conf_spatial], dim=1
        ))

        b_out = b + confidence * reconstructed
        b_out = self.norm(b_out)
        return self.cv2(torch.cat([a, b_out], dim=1))


# ============================================================
# DAPR_noGate: Ablation without confidence gate
# ============================================================

class DAPR_noGate(DAPR_only_v3):
    """Ablation variant: confidence gate is disabled (fixed unit weight)."""

    def __init__(self, c1, c2, num_classes=2, parts_per_class=4,
                 proto_dim=48, reduction=4, temperature=0.50):
        super().__init__(c1, c2, num_classes, parts_per_class,
                         proto_dim, reduction, temperature)

    def forward(self, x):
        B, C1, H, W = x.shape
        device, dtype = x.device, x.dtype
        N = H * W

        a, b = self.cv1(x).chunk(2, 1)
        x_internal = self.input_proj(b)
        local_feat = self.local_encoder(x_internal)
        D = local_feat.shape[1]
        local_flat = local_feat.flatten(2).transpose(1, 2)

        assign_logits = self.part_assignment(x_internal).flatten(2)
        assignment = F.softmax(assign_logits, dim=1)
        part_features = torch.bmm(assignment, local_flat)

        if self._cached_HW != (H, W):
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing="ij",
            )
            self._coords_cache = torch.stack([xx, yy], dim=-1).view(1, N, 2)
            self._cached_HW = (H, W)
        coords = self._coords_cache.to(device=device, dtype=dtype).expand(B, -1, -1)
        part_centers = torch.bmm(assignment, coords)
        pos_embed = self.position_mlp(part_centers)

        inst_desc = part_features.mean(dim=1)
        delta = self.prototype_generator(inst_desc).view(
            B, self.num_parts, self.proto_dim
        )

        class_logits = self.class_classifier(part_features)
        class_logits = class_logits.view(
            B, self.num_parts, self.num_classes, self.parts_per_class
        )
        class_probs = F.softmax(
            class_logits.view(B, self.num_parts, -1), dim=-1
        ).view(B, self.num_parts, self.num_classes, self.parts_per_class)
        class_prototypes = torch.einsum(
            "bkcp,cpd->bkd", class_probs, self.base_prototypes
        )

        prototypes = class_prototypes + 0.10 * delta + 0.10 * pos_embed

        q = F.normalize(self.part_query(part_features), p=2, dim=-1)
        k = F.normalize(self.prototype_key(prototypes), p=2, dim=-1)
        sim = torch.matmul(q, k.transpose(1, 2)) / self.temperature
        proto_attn = F.softmax(sim, dim=-1)
        matched_parts = torch.matmul(proto_attn, prototypes)

        assign_t = assignment.transpose(1, 2)
        spatial_feat = torch.bmm(assign_t, matched_parts)
        spatial_feat = spatial_feat.transpose(1, 2).contiguous().view(B, D, H, W)

        reconstructed = self.reconstruction_proj(spatial_feat)
        reconstructed = self.reconstruction_dw(reconstructed)
        reconstructed = self.reconstruction_act(
            self.reconstruction_bn(reconstructed)
        )

        # Gate disabled: confidence = 1 (residual only)
        b_out = b + reconstructed
        b_out = self.norm(b_out)
        return self.cv2(torch.cat([a, b_out], dim=1))


# ============================================================
# DAPR_noClassPart: Ablation without class-part conditioning
# ============================================================

class DAPR_noClassPart(DAPR_only_v3):
    """
    Ablation variant: class-part conditioning is replaced by a uniform
    distribution over all (class, part) combinations.
    """

    def __init__(self, c1, c2, num_classes=2, parts_per_class=4,
                 proto_dim=48, reduction=4, temperature=0.50):
        super().__init__(c1, c2, num_classes, parts_per_class,
                         proto_dim, reduction, temperature)

    def forward(self, x):
        B, C1, H, W = x.shape
        device, dtype = x.device, x.dtype
        N = H * W

        a, b = self.cv1(x).chunk(2, 1)
        x_internal = self.input_proj(b)
        local_feat = self.local_encoder(x_internal)
        D = local_feat.shape[1]
        local_flat = local_feat.flatten(2).transpose(1, 2)

        assign_logits = self.part_assignment(x_internal).flatten(2)
        assignment = F.softmax(assign_logits, dim=1)
        part_features = torch.bmm(assignment, local_flat)

        if self._cached_HW != (H, W):
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing="ij",
            )
            self._coords_cache = torch.stack([xx, yy], dim=-1).view(1, N, 2)
            self._cached_HW = (H, W)
        coords = self._coords_cache.to(device=device, dtype=dtype).expand(B, -1, -1)
        part_centers = torch.bmm(assignment, coords)
        pos_embed = self.position_mlp(part_centers)

        inst_desc = part_features.mean(dim=1)
        delta = self.prototype_generator(inst_desc).view(
            B, self.num_parts, self.proto_dim
        )

        # Class-part conditioning disabled: uniform distribution
        class_probs = torch.full(
            (B, self.num_parts, self.num_classes, self.parts_per_class),
            1.0 / (self.num_classes * self.parts_per_class),
            device=device, dtype=dtype,
        )

        class_prototypes = torch.einsum(
            "bkcp,cpd->bkd", class_probs, self.base_prototypes
        )

        prototypes = class_prototypes + 0.10 * delta + 0.10 * pos_embed

        q = F.normalize(self.part_query(part_features), p=2, dim=-1)
        k = F.normalize(self.prototype_key(prototypes), p=2, dim=-1)
        sim = torch.matmul(q, k.transpose(1, 2)) / self.temperature
        proto_attn = F.softmax(sim, dim=-1)
        matched_parts = torch.matmul(proto_attn, prototypes)

        assign_t = assignment.transpose(1, 2)
        spatial_feat = torch.bmm(assign_t, matched_parts)
        spatial_feat = spatial_feat.transpose(1, 2).contiguous().view(B, D, H, W)

        reconstructed = self.reconstruction_proj(spatial_feat)
        reconstructed = self.reconstruction_dw(reconstructed)
        reconstructed = self.reconstruction_act(
            self.reconstruction_bn(reconstructed)
        )

        # Uniform class confidence
        class_conf = torch.ones(B, self.num_parts, 1, device=device, dtype=dtype)
        class_conf_spatial = torch.bmm(assign_t, class_conf)
        class_conf_spatial = class_conf_spatial.transpose(1, 2).contiguous().view(
            B, 1, H, W
        )

        confidence = self.gate(torch.cat(
            [x_internal, spatial_feat, class_conf_spatial], dim=1
        ))

        b_out = b + confidence * reconstructed
        b_out = self.norm(b_out)
        return self.cv2(torch.cat([a, b_out], dim=1))