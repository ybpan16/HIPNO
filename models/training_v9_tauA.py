#!/usr/bin/env python3
from __future__ import annotations

import copy
import gc
import math
import os
from itertools import islice

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from . import training_v9_core as v9c
from data.dataset_vitaldb4 import (
    TauDecayTargetConfig,
    _detect_dicrotic_notch_index,
    _fit_log_decay_tau_profile,
)


PRESSURE_NORM_SBP_CENTER = 120.0
PRESSURE_NORM_SBP_SCALE = 40.0
PRESSURE_NORM_DBP_CENTER = 70.0
PRESSURE_NORM_DBP_SCALE = 25.0
PRESSURE_NORM_MAP_CENTER = 90.0
PRESSURE_NORM_MAP_SCALE = 30.0
PRESSURE_NORM_PP_CENTER = 50.0
PRESSURE_NORM_PP_SCALE = 30.0


# =============================================================
# v9 model: direct waveform branch + physics branch
# =============================================================


class DirectABPWaveformHead(nn.Module):
    """Supervised ECG/PPG -> ABP waveform head.

    This branch predicts pressure directly from the learned waveform context.
    It is not a correction on the Windkessel rollout.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        nfreq_theta: int = 10,
        nibp_embed_dim: int = 16,
        pressure_bias_mmHg: float = 80.0,
    ):
        super().__init__()
        self.emb_theta = v9c.FourierAngle(nfreq_theta)
        self.nibp_embed_dim = int(nibp_embed_dim)
        in_dim = 2 * nfreq_theta + 1 + 2 * latent_dim + self.nibp_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.net[-1].bias, float(pressure_bias_mmHg))

    def forward(
        self,
        theta_bt1: torch.Tensor,
        prog_bt1: torch.Tensor,
        abp_context_bt: torch.Tensor,
        z_b: torch.Tensor,
        nibp_embed_b: torch.Tensor,
    ) -> torch.Tensor:
        B, T = theta_bt1.shape[:2]
        theta_feat = self.emb_theta(theta_bt1)
        z_bt = z_b.unsqueeze(1).expand(B, T, -1)
        nibp_bt = nibp_embed_b.unsqueeze(1).expand(B, T, -1)
        features = torch.cat([theta_feat, prog_bt1, abp_context_bt, z_bt, nibp_bt], dim=-1)
        return self.net(features).squeeze(-1)


class NIBPConditioner(nn.Module):
    """Encode scalar cuff context and apply a learned in-model pressure calibration."""

    def __init__(
        self,
        *,
        feature_dim: int = 6,
        latent_dim: int = 64,
        embed_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self.affine_head = nn.Sequential(
            nn.Linear(latent_dim + self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)

    def encode(self, nibp_features: torch.Tensor) -> torch.Tensor:
        if nibp_features.ndim != 2 or nibp_features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"nibp_features must have shape [B,{self.feature_dim}], "
                f"got {tuple(nibp_features.shape)}."
            )
        return self.encoder(nibp_features)

    def calibrate(self, abp_bt: torch.Tensor, z_b: torch.Tensor, nibp_embed_b: torch.Tensor) -> torch.Tensor:
        raw = self.affine_head(torch.cat([z_b, nibp_embed_b], dim=-1))
        scale = torch.exp(raw[:, 0:1])
        shift = raw[:, 1:2]
        return scale * abp_bt + shift


class HemodynamicCOHead(nn.Module):
    """Compatibility CO head retained for v6 checkpoint/API shape only."""

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        nibp_embed_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        in_dim = latent_dim + nibp_embed_dim + 9
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(math.exp(5.0) - 1.0))

    def forward(
        self,
        z_b: torch.Tensor,
        nibp_embed_b: torch.Tensor,
        abp_bt: torch.Tensor,
        wk: dict[str, torch.Tensor],
        qscale_b: torch.Tensor,
    ) -> torch.Tensor:
        sbp = abp_bt.max(dim=1).values
        dbp = abp_bt.min(dim=1).values
        map_p = abp_bt.mean(dim=1)
        pp = sbp - dbp
        pressure_features = torch.stack(
            [
                (sbp - PRESSURE_NORM_SBP_CENTER) / PRESSURE_NORM_SBP_SCALE,
                (dbp - PRESSURE_NORM_DBP_CENTER) / PRESSURE_NORM_DBP_SCALE,
                (map_p - PRESSURE_NORM_MAP_CENTER) / PRESSURE_NORM_MAP_SCALE,
                (pp - PRESSURE_NORM_PP_CENTER) / PRESSURE_NORM_PP_SCALE,
            ],
            dim=1,
        )
        wk_features = torch.stack(
            [
                torch.log(wk["R1"].clamp_min(1e-8)),
                torch.log(wk["R2"].clamp_min(1e-8)),
                torch.log(wk["tau"].clamp_min(1e-8)),
                (wk["Pv"] - 5.0) / 5.0,
                torch.log(qscale_b.clamp_min(1e-8)),
            ],
            dim=1,
        )
        raw = self.net(torch.cat([z_b, nibp_embed_b, pressure_features, wk_features], dim=1)).squeeze(-1)
        return F.softplus(raw) + 1e-6


class DirectWaveformDeepONet1DFlowPINN(v9c.DeepONet1DFlowPINN):
    """Direct waveform + identifiable pressure dynamics.

    The pressure branch is parameterized by tau, Pv, and compliance-normalized
    flow drive u(t)=Q(t)/C. Internal C is a unit gauge, not physical arterial
    compliance. A downstream calibrator can later supply the sole physical
    compliance scale C_cal and recover Q_abs=C_cal*u, R=tau/C_cal.
    """

    def __init__(
        self,
        branch_dim=2,
        hidden_dim=256,
        latent_dim=64,
        nx=16,
        Lx=1.0,
        encoder_temporal_stride: int = 1,
        nibp_feature_dim: int = 6,
        nibp_embed_dim: int = 16,
    ):
        super().__init__(
            branch_dim=branch_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            nx=nx,
            Lx=Lx,
            encoder_temporal_stride=encoder_temporal_stride,
        )
        self.nibp_conditioner = NIBPConditioner(
            feature_dim=nibp_feature_dim,
            latent_dim=latent_dim,
            embed_dim=nibp_embed_dim,
            hidden_dim=hidden_dim // 2,
        )
        self.direct_abp_head = DirectABPWaveformHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            nfreq_theta=10,
            nibp_embed_dim=nibp_embed_dim,
        )
        self.co_head = HemodynamicCOHead(
            latent_dim=latent_dim,
            nibp_embed_dim=nibp_embed_dim,
            hidden_dim=hidden_dim // 2,
        )
        # Kept for checkpoint/API compatibility; training_v9_tauA freezes it and leaves CO undefined.
        self.register_buffer("direct_blend_alpha", torch.tensor(1.0))
        self.detach_slow_wk_rollout = True

    def _wk_from_raw(self, raw):
        rho1 = self._bounded_sigmoid(raw[..., 0], 0.02, 0.15)
        tau = self._bounded_sigmoid(raw[..., 2], 0.30, 2.50)
        Pv = self._bounded_sigmoid(raw[..., 3], 2.00, 20.00)
        C_gauge = torch.ones_like(tau)
        R2_gauge = tau
        return {
            "R1": rho1,
            "rho1": rho1,
            "R2": R2_gauge,
            "tau": tau,
            "Pv": Pv,
            "C": C_gauge,
            "C_gauge": C_gauge,
            "R_total": rho1 + R2_gauge,
        }

    def windkessel_prior_loss(self, wk=None):
        if wk is None:
            wk = self._reference_windkessel_params()
        rho1 = wk["rho1"] if "rho1" in wk else wk["R1"]
        L_tau = (torch.log(wk["tau"]) - math.log(1.0)).pow(2) / (2 * 0.5**2)
        L_rho1 = (torch.log(rho1) - math.log(0.06)).pow(2) / (2 * 0.6**2)
        L_Pv = (wk["Pv"] - wk["Pv"].new_tensor(5.0)).pow(2) / (2 * 3.0**2)
        return (L_tau + L_rho1 + 0.2 * L_Pv).mean()

    def physiologic_barrier_loss(self, wk=None):
        if wk is None:
            wk = self._reference_windkessel_params()
        tau = wk["tau"]
        rho1 = wk["rho1"] if "rho1" in wk else wk["R1"]
        loss = tau.new_tensor(0.0)
        loss = loss + F.relu(tau.new_tensor(0.45) - tau).pow(2).mean()
        loss = loss + 0.5 * F.relu(tau - tau.new_tensor(2.20)).pow(2).mean()
        loss = loss + 0.25 * F.relu(rho1.new_tensor(0.02) - rho1).pow(2).mean()
        loss = loss + 0.25 * F.relu(rho1 - rho1.new_tensor(0.15)).pow(2).mean()
        loss = loss + 0.1 * (self.excess_gain - self.excess_gain.new_tensor(0.25)).pow(2)
        return loss

    def operator_fields(self, theta, prog, z_b, x_grid=None):
        op = super().operator_fields(theta, prog, z_b, x_grid=x_grid)
        op["U"] = op["Q"]
        op["U_in"] = op["Q_in"]
        op["U_in_bc"] = op["Q_in_bc"]
        op["Uscale"] = op["Qscale"]
        return op

    def set_direct_blend_alpha(self, alpha: float) -> None:
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"direct blend alpha must be in [0, 1], got {alpha}.")
        self.direct_blend_alpha.fill_(float(alpha))

    def set_slow_wk_rollout_detach(self, enabled: bool) -> None:
        self.detach_slow_wk_rollout = bool(enabled)

    def _wk_for_rollout(self, wk: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.detach_slow_wk_rollout:
            return wk
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in wk.items()}

    def encode_waveform_context(self, input_func: torch.Tensor, dt_seconds=None) -> dict[str, torch.Tensor]:
        B = input_func.shape[0]
        ecg = input_func[..., 0:1]
        ppg = input_func[..., 1:2]
        ecg_memory, ecg_summary = self.ecg_branch(ecg)
        ppg_memory, ppg_summary = self.ppg_branch(ppg)
        phi = self.phase_net(ecg, dt_seconds)

        u = phi / (2.0 * math.pi)
        theta = 2.0 * math.pi * torch.frac(u)
        final_cycle_count = u[:, -1:, :]
        torch._assert(
            (final_cycle_count > 0).all(),
            "Phase network produced a non-positive final cycle count.",
        )
        prog = u / final_cycle_count
        abp_context_bt, z_b = self.abp_decoder(
            theta,
            prog,
            ecg_memory,
            ppg_memory,
            ecg_summary,
            ppg_summary,
        )
        return {
            "phi": phi,
            "theta": theta,
            "prog": prog,
            "abp_context": abp_context_bt,
            "global_latent": z_b,
            "ecg_summary": ecg_summary,
            "ppg_summary": ppg_summary,
        }

    def _nibp_embed(self, ctx: dict[str, torch.Tensor], nibp_features: torch.Tensor | None) -> torch.Tensor:
        z_b = ctx["global_latent"]
        if nibp_features is None:
            nibp_features = z_b.new_zeros((z_b.shape[0], self.nibp_conditioner.feature_dim))
        return self.nibp_conditioner.encode(nibp_features.to(device=z_b.device, dtype=z_b.dtype))

    def direct_abp_from_context(
        self,
        ctx: dict[str, torch.Tensor],
        nibp_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nibp_embed = self._nibp_embed(ctx, nibp_features)
        abp = self.direct_abp_head(
            ctx["theta"],
            ctx["prog"],
            ctx["abp_context"],
            ctx["global_latent"],
            nibp_embed,
        )
        abp = self.nibp_conditioner.calibrate(abp, ctx["global_latent"], nibp_embed)
        return abp, nibp_embed

    def forward_waveform(
        self,
        input_func: torch.Tensor,
        t_unused=None,
        dt_seconds=None,
        nibp_features: torch.Tensor | None = None,
    ):
        del t_unused
        ctx = self.encode_waveform_context(input_func, dt_seconds=dt_seconds)
        abp_direct, nibp_embed = self.direct_abp_from_context(ctx, nibp_features=nibp_features)
        fields = {
            "phi": ctx["phi"],
            "theta": ctx["theta"],
            "prog": ctx["prog"],
            "abp_context": ctx["abp_context"],
            "ecg_summary": ctx["ecg_summary"],
            "ppg_summary": ctx["ppg_summary"],
            "abp_direct": abp_direct,
            "abp_physics": abp_direct.new_zeros(abp_direct.shape),
            "abp_bt": abp_direct,
            "nibp_embed": nibp_embed,
        }
        aux = {"global_latent": ctx["global_latent"]}
        return abp_direct.unsqueeze(-1), fields, aux

    def forward(self, input_func, t_unused=None, dt_seconds=None, nibp_features: torch.Tensor | None = None):
        del t_unused

        B, T = input_func.shape[0], input_func.shape[1]
        ctx = self.encode_waveform_context(input_func, dt_seconds=dt_seconds)
        phi = ctx["phi"]
        theta = ctx["theta"]
        prog = ctx["prog"]
        abp_context_bt = ctx["abp_context"]
        z_b = ctx["global_latent"]
        abp_direct, nibp_embed = self.direct_abp_from_context(ctx, nibp_features=nibp_features)

        wk = self.patient_windkessel_params(z_b)
        wk_rollout = self._wk_for_rollout(wk)
        op = self.operator_fields(theta, prog, z_b)
        A = op["A"]
        U = op["Q"]
        P = op["P"]
        U_in = op["Q_in"]
        U_in_bc = op["Q_in_bc"]
        Uscale_b = op["Qscale"]
        P_prox = P[..., 0]
        P_dist = P[..., -1]
        U_L = U[..., -1]
        dt_bt = v9c._as_bt(dt_seconds if dt_seconds is not None else 1.0, B, T, U.device, U.dtype)

        Pc0_b = self.predict_Pc0(z_b, wk=wk_rollout)
        P_wk, Pc = self.windkessel_forward(U_L, dt_bt, wk=wk_rollout, Pc0=Pc0_b)
        P_excess = P_prox - P_dist

        # The old residual path is intentionally inactive; keep the field for
        # checkpoint/evaluation code that expects the v6 key to exist.
        abp_residual = P_wk.new_zeros(P_wk.shape)
        abp_physics = P_wk + self.excess_gain * P_excess
        alpha = self.direct_blend_alpha.to(dtype=abp_direct.dtype, device=abp_direct.device)
        abp_bt = alpha * abp_direct + (1.0 - alpha) * abp_physics
        abp_pred = abp_bt.unsqueeze(-1)

        u_mean = v9c.mean_flow(U_L, dt_bt)
        undefined_co_l_min = torch.full_like(u_mean, float("nan"))
        undefined_co_ml_s = torch.full_like(u_mean, float("nan"))

        fields = {
            "A": A,
            "Q": U,
            "U": U,
            "P": P,
            "phi": phi,
            "theta": theta,
            "prog": prog,
            "P_wk": P_wk,
            "P_prox": P_prox,
            "P_dist": P_dist,
            "P_excess": P_excess,
            "abp_residual": abp_residual,
            "abp_physics": abp_physics,
            "abp_direct": abp_direct,
            "abp_bt": abp_bt,
            "abp_blend_alpha": alpha.expand(()),
            "abp_context": abp_context_bt,
            "ecg_summary": ctx["ecg_summary"],
            "ppg_summary": ctx["ppg_summary"],
            "nibp_embed": nibp_embed,
        }
        aux = {
            "Q_in": U_in,
            "Q_in_bc": U_in_bc,
            "U_in": U_in,
            "U_in_bc": U_in_bc,
            "Pc": Pc,
            "Qscale": Uscale_b,
            "Uscale": Uscale_b,
            "U_L": U_L,
            "U_mean": u_mean,
            "CO_physics_ml_s": undefined_co_ml_s,
            "CO_physics_L_min": undefined_co_l_min,
            "CO_head_L_min": undefined_co_l_min,
            "CO_ml_s": undefined_co_ml_s,
            "CO_L_min": undefined_co_l_min,
            "global_latent": z_b,
            "wk": wk,
            "wk_rollout": wk_rollout,
        }
        return abp_pred, fields, aux


# =============================================================
# Curriculum helpers
# =============================================================


def _is_direct_waveform_param(name: str) -> bool:
    name = v9c._canonical_param_name(name)
    waveform_modules = (
        "ecg_branch",
        "ppg_branch",
        "abp_decoder",
        "direct_abp_head",
        "nibp_conditioner",
    )
    return any(token in name for token in waveform_modules)


def _is_physics_param(name: str) -> bool:
    name = v9c._canonical_param_name(name)
    physics_modules = (
        "trunk",
        "head_scale_A",
        "head_bias_A",
        "head_scale_Q",
        "head_bias_Q",
        "head_logQscale",
        "inlet_param_head",
        "wk_param_head",
        "_Pc0_raw",
        "_beta_raw",
        "Pext",
        "_cp_raw",
        "_Cf_raw",
        "_excess_gain_raw",
    )
    return any(token in name for token in physics_modules)


def _set_trainable_stage_v6(model: nn.Module, stage: str) -> None:
    for name, p in model.named_parameters():
        canonical_name = v9c._canonical_param_name(name)
        if "co_head" in canonical_name:
            trainable = False
        elif stage == "phase":
            trainable = v9c._is_phase_param(name)
        elif stage == "waveform":
            trainable = _is_direct_waveform_param(name)
        elif stage == "physics":
            trainable = _is_direct_waveform_param(name) or _is_physics_param(name)
        elif stage == "joint":
            trainable = True
        else:
            raise ValueError(f"Unsupported training stage: {stage!r}")
        p.requires_grad_(trainable)


def _curriculum_stage_v6(
    epoch: int,
    phase_end_epoch: int,
    waveform_end_epoch: int,
    physics_end_epoch: int,
) -> str:
    if epoch <= phase_end_epoch:
        return "phase"
    if epoch <= waveform_end_epoch:
        return "waveform"
    if epoch <= physics_end_epoch:
        return "physics"
    return "joint"


def _linear_epoch_value(epoch: int, start_epoch: int, end_epoch: int, start_value: float, end_value: float) -> float:
    if end_epoch < start_epoch:
        raise ValueError(f"end_epoch must be greater than or equal to start_epoch, got {start_epoch}->{end_epoch}.")
    if end_epoch == start_epoch:
        return float(end_value) if epoch >= end_epoch else float(start_value)
    if epoch <= start_epoch:
        return float(start_value)
    if epoch >= end_epoch:
        return float(end_value)
    frac = (int(epoch) - int(start_epoch)) / float(int(end_epoch) - int(start_epoch))
    return float(start_value) + frac * (float(end_value) - float(start_value))


def _direct_alpha_for_epoch(
    epoch: int,
    stage: str,
    *,
    waveform_end_epoch: int,
    physics_end_epoch: int,
    physics_warm_end_epoch: int | None,
    physics_start_alpha: float,
    physics_warm_end_alpha: float | None,
    physics_end_alpha: float,
    joint_alpha: float,
    joint_alpha_ramp_epochs: int,
) -> float:
    if stage in {"phase", "waveform"}:
        return 1.0
    if stage == "physics":
        if physics_warm_end_epoch is not None:
            if physics_warm_end_alpha is None:
                raise ValueError("physics_warm_end_alpha is required when physics_warm_end_epoch is set.")
            if epoch <= int(physics_warm_end_epoch):
                return _linear_epoch_value(
                    epoch,
                    int(waveform_end_epoch) + 1,
                    int(physics_warm_end_epoch),
                    float(physics_start_alpha),
                    float(physics_warm_end_alpha),
                )
            return _linear_epoch_value(
                epoch,
                int(physics_warm_end_epoch) + 1,
                int(physics_end_epoch),
                float(physics_warm_end_alpha),
                float(physics_end_alpha),
            )
        ramp = v9c._linear_ramp_between_epochs(epoch, waveform_end_epoch, physics_end_epoch)
        return 1.0 + ramp * (float(physics_end_alpha) - 1.0)
    if stage == "joint":
        return _linear_epoch_value(
            epoch,
            int(physics_end_epoch) + 1,
            int(physics_end_epoch) + max(1, int(joint_alpha_ramp_epochs)),
            float(physics_end_alpha),
            float(joint_alpha),
        )
    raise ValueError(f"Unsupported training stage: {stage!r}")


def _make_waveform_teacher(model: nn.Module, device: torch.device) -> DirectWaveformDeepONet1DFlowPINN:
    teacher = copy.deepcopy(v9c._unwrap_compiled_model(model)).to(device)
    teacher.set_direct_blend_alpha(1.0)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _strict_co_prior_from_batch(subject_ids, co_prior_map, device, dtype=torch.float32):
    missing = [sid for sid in subject_ids if sid is None or int(sid) not in co_prior_map]
    if missing:
        raise KeyError(
            "CO prior is enabled, but one or more subject IDs are missing from the CO prior map."
        )
    vals = [float(co_prior_map[int(sid)]) for sid in subject_ids]
    return torch.tensor(vals, device=device, dtype=dtype)


def _nibp_features_from_batch(
    batch: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    require_all_valid: bool = False,
) -> torch.Tensor:
    v9c.require_batch_keys(
        batch,
        [
            "nibp_sbp_label",
            "nibp_dbp_label",
            "nibp_map_label",
            "nibp_valid",
            "nearest_nibp_delta_sec",
        ],
        "NIBP input",
    )
    sbp = batch["nibp_sbp_label"].to(device=device, dtype=dtype).view(-1)
    dbp = batch["nibp_dbp_label"].to(device=device, dtype=dtype).view(-1)
    map_p = batch["nibp_map_label"].to(device=device, dtype=dtype).view(-1)
    delta = batch["nearest_nibp_delta_sec"].to(device=device, dtype=dtype).view(-1)
    valid = (
        batch["nibp_valid"].to(device=device).view(-1).bool()
        & torch.isfinite(sbp)
        & torch.isfinite(dbp)
        & torch.isfinite(map_p)
        & torch.isfinite(delta)
    )
    if require_all_valid and not bool(valid.all().item()):
        raise RuntimeError("NIBP input is required, but at least one batch element has invalid NIBP context.")

    zero = torch.zeros_like(sbp)
    sbp_in = torch.where(valid, sbp, zero)
    dbp_in = torch.where(valid, dbp, zero)
    map_in = torch.where(valid, map_p, zero)
    delta_in = torch.where(valid, delta, zero)
    pp_in = sbp_in - dbp_in
    present = valid.to(dtype=dtype)
    return torch.stack(
        [
            (sbp_in - PRESSURE_NORM_SBP_CENTER) / PRESSURE_NORM_SBP_SCALE,
            (dbp_in - PRESSURE_NORM_DBP_CENTER) / PRESSURE_NORM_DBP_SCALE,
            (map_in - PRESSURE_NORM_MAP_CENTER) / PRESSURE_NORM_MAP_SCALE,
            (pp_in - PRESSURE_NORM_PP_CENTER) / PRESSURE_NORM_PP_SCALE,
            delta_in / 600.0,
            present,
        ],
        dim=1,
    )


# =============================================================
# Loss blocks
# =============================================================


def _phase_loss(
    phi_hat_bt: torch.Tensor,
    phi_ref_bt: torch.Tensor,
    phi_mask: torch.Tensor,
    r_peaks_local,
    *,
    lambda_phase_anchor: float,
    lambda_phase_smooth: float,
    lambda_phase_mono: float,
) -> torch.Tensor:
    L_ref = v9c.phase_reference_loss_offset_invariant(phi_hat_bt, phi_ref_bt, phi_mask)
    L_peak = v9c.phase_peak_anchor_loss_local(phi_hat_bt, r_peaks_local)
    dphi = phi_hat_bt[:, 1:] - phi_hat_bt[:, :-1]
    L_mono = F.relu(-dphi).mean()
    L_smooth = (dphi[:, 1:] - dphi[:, :-1]).pow(2).mean()
    return L_ref + lambda_phase_anchor * L_peak + lambda_phase_mono * L_mono + lambda_phase_smooth * L_smooth


def _waveform_reconstruction_loss(
    model: nn.Module,
    y_hat_bt: torch.Tensor,
    y_bt: torch.Tensor,
    mask_bt: torch.Tensor,
    dt_bt: torch.Tensor,
    beat_bounds,
    *,
    w_abs: float,
    w_z: float,
    w_der: float,
    w_beat: float,
    w_abp_stats: float,
    w_corr: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    L_abs = v9c.masked_mse(y_hat_bt, y_bt, mask_bt)
    mu, sd = v9c._masked_mean_std(y_bt, mask_bt)
    z_pred = (y_hat_bt - mu) / (sd + 1e-6)
    z_true = (y_bt - mu) / (sd + 1e-6)
    L_z = v9c.masked_huber(z_pred, z_true, mask_bt, delta=1.0)
    dy_t = model.dt_centered(y_bt, dt_bt)
    dy_p = model.dt_centered(y_hat_bt, dt_bt)
    L_der = v9c.masked_mse(dy_p, dy_t, mask_bt)
    L_corr = v9c.corr_loss(y_hat_bt, y_bt, mask_bt)
    L_abp_stats = v9c.window_sys_dia_map_loss_soft(y_hat_bt, y_bt, mask_bt)

    beat_pack = v9c.pack_valid_beat_tensors(
        torch.stack([y_hat_bt, y_bt], dim=-1),
        mask_bt,
        beat_bounds,
        min_points=6,
    )
    if beat_pack is None or w_beat <= 0:
        L_beat = y_hat_bt.new_tensor(0.0)
    else:
        L_beat = v9c.beatwise_sys_dia_loss_soft_vectorized(
            beat_pack.values[:, :, 0],
            beat_pack.values[:, :, 1],
            beat_pack.valid,
        )

    rec_loss = (
        w_abs * L_abs
        + w_z * L_z
        + w_der * L_der
        + w_corr * L_corr
        + w_beat * L_beat
        + w_abp_stats * L_abp_stats
    )
    parts = {
        "abs": L_abs,
        "z": L_z,
        "der": L_der,
        "corr": L_corr,
        "beat": L_beat,
        "abp_stats": L_abp_stats,
    }
    return rec_loss, parts


def _nibp_loss_if_enabled(batch, y_hat_bt, mask_bt, lambda_nibp_eff: float):
    if lambda_nibp_eff <= 0:
        return y_hat_bt.new_tensor(0.0), torch.zeros((), device=y_hat_bt.device, dtype=torch.long)
    v9c.require_batch_keys(
        batch,
        ["nibp_sbp_label", "nibp_dbp_label", "nibp_map_label", "nibp_valid"],
        "NIBP",
    )
    nibp_sbp_t = batch["nibp_sbp_label"].to(device=y_hat_bt.device, dtype=y_hat_bt.dtype).view(-1)
    nibp_dbp_t = batch["nibp_dbp_label"].to(device=y_hat_bt.device, dtype=y_hat_bt.dtype).view(-1)
    nibp_map_t = batch["nibp_map_label"].to(device=y_hat_bt.device, dtype=y_hat_bt.dtype).view(-1)
    nibp_valid_t = batch["nibp_valid"].to(device=y_hat_bt.device).view(-1).bool()
    return v9c.window_nibp_loss_soft(
        y_hat_bt,
        mask_bt,
        nibp_sbp_t,
        nibp_dbp_t,
        nibp_map_t,
        nibp_valid_t,
    )


def _co_supervised_loss_if_enabled(batch, co_pred, lambda_co_supervised_eff: float):
    if lambda_co_supervised_eff <= 0:
        return co_pred.new_tensor(0.0), torch.zeros((), device=co_pred.device, dtype=torch.long)
    v9c.require_batch_keys(batch, ["co_ref_l_min", "co_valid"], "CO")
    co_ref_t = batch["co_ref_l_min"].to(device=co_pred.device, dtype=co_pred.dtype).view(-1)
    co_valid_t = batch["co_valid"].to(device=co_pred.device).view(-1).bool()
    return v9c.masked_scalar_anchor_loss(co_pred.view(-1), co_ref_t, co_valid_t, scale=2.0)


def _cvp_loss_if_enabled(batch, pv_pred, lambda_cvp_eff: float):
    if lambda_cvp_eff <= 0:
        return pv_pred.new_tensor(0.0), torch.zeros((), device=pv_pred.device, dtype=torch.long)
    v9c.require_batch_keys(batch, ["cvp_label", "cvp_valid"], "CVP")
    cvp_ref_t = batch["cvp_label"].to(device=pv_pred.device, dtype=pv_pred.dtype).view(-1)
    cvp_valid_t = batch["cvp_valid"].to(device=pv_pred.device).view(-1).bool()
    return v9c.masked_scalar_anchor_loss(pv_pred.view(-1), cvp_ref_t, cvp_valid_t, scale=5.0)


def _tau_loss_if_enabled(batch, log_tau_pred, lambda_tau_eff: float):
    if lambda_tau_eff <= 0:
        zero_n = torch.zeros((), device=log_tau_pred.device, dtype=torch.long)
        nan = log_tau_pred.new_tensor(float("nan"))
        return log_tau_pred.new_tensor(0.0), zero_n, nan, nan
    v9c.require_batch_keys(
        batch,
        ["tau_obs_log", "tau_obs_se", "tau_weight", "tau_valid"],
        "diastolic tau",
    )
    tau_obs_log_t = batch["tau_obs_log"].to(device=log_tau_pred.device, dtype=log_tau_pred.dtype).view(-1)
    tau_obs_se_t = batch["tau_obs_se"].to(device=log_tau_pred.device, dtype=log_tau_pred.dtype).view(-1)
    tau_weight_t = batch["tau_weight"].to(device=log_tau_pred.device, dtype=log_tau_pred.dtype).view(-1)
    tau_valid_t = batch["tau_valid"].to(device=log_tau_pred.device).view(-1).bool()
    valid_tau = (
        tau_valid_t
        & torch.isfinite(tau_obs_log_t)
        & torch.isfinite(tau_obs_se_t)
        & torch.isfinite(tau_weight_t)
        & torch.isfinite(log_tau_pred)
        & (tau_obs_se_t > 0)
        & (tau_weight_t > 0)
    )
    n_tau = valid_tau.sum()
    if not valid_tau.any():
        nan = log_tau_pred.new_tensor(float("nan"))
        return log_tau_pred.new_tensor(0.0), n_tau, nan, nan
    err = log_tau_pred[valid_tau] - tau_obs_log_t[valid_tau]
    weight = tau_weight_t[valid_tau] / tau_obs_se_t[valid_tau].pow(2)
    loss = (weight * err.pow(2)).sum() / weight.sum()
    tau_obs_mean = torch.exp(tau_obs_log_t[valid_tau]).mean()
    tau_pred_mean = torch.exp(log_tau_pred[valid_tau]).mean()
    return loss, n_tau, tau_obs_mean, tau_pred_mean


def _pv_anchor_loss_if_enabled(
    wk: dict[str, torch.Tensor],
    lambda_pv_anchor_eff: float,
    *,
    anchor_mmHg: float,
    sigma_mmHg: float,
) -> torch.Tensor:
    pv = wk["Pv"].view(-1)
    if lambda_pv_anchor_eff <= 0.0:
        return pv.new_tensor(0.0)
    if sigma_mmHg <= 0.0:
        raise ValueError(f"sigma_mmHg must be positive, got {sigma_mmHg}.")
    target = pv.new_tensor(float(anchor_mmHg))
    sigma = pv.new_tensor(float(sigma_mmHg))
    return ((pv - target) / sigma).pow(2).mean()


def _qscale_gauge_loss_if_enabled(qscale: torch.Tensor, lambda_qscale_gauge_eff: float) -> torch.Tensor:
    qscale = qscale.view(-1)
    if lambda_qscale_gauge_eff <= 0.0:
        return qscale.new_tensor(0.0)
    return torch.log(qscale).pow(2).mean()


def _as_numpy_peak_array(peaks) -> np.ndarray:
    if peaks is None:
        return np.asarray([], dtype=np.int64)
    if isinstance(peaks, torch.Tensor):
        arr = peaks.detach().cpu().view(-1).numpy()
    else:
        arr = np.asarray(peaks)
    if arr.size == 0:
        return np.asarray([], dtype=np.int64)
    arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        arr = arr[finite]
    return arr.astype(np.int64, copy=False)


def _rescaled_matched_abp_peaks(batch, b: int, source_T: int, target_T: int) -> list[int]:
    if "matched_abp_peaks" not in batch:
        raise RuntimeError("Tau morphology supervision requires batch['matched_abp_peaks'].")
    peaks_raw = batch["matched_abp_peaks"][int(b)]
    return v9c._rescale_indices_to_length(_as_numpy_peak_array(peaks_raw), source_T, target_T)


def _batch_window_value(batch, name: str, b: int):
    value = batch[name]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value[int(b)].item() if value[int(b)].numel() == 1 else value[int(b)].tolist()
    if isinstance(value, (list, tuple)):
        return value[int(b)]
    return value


def _key_scalar(value) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        value = value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return format(value, ".17g")
    return str(value)


def _tau_morphology_window_key(batch, b: int) -> str:
    fields = ("subject_id", "record_id", "hadm_id", "clean_window_order", "start", "end")
    missing = [name for name in fields if name not in batch]
    if missing:
        raise RuntimeError(f"Tau morphology precompute key requires batch fields: {missing}")
    return "|".join(f"{name}={_key_scalar(_batch_window_value(batch, name, b))}" for name in fields)


def _load_tau_morphology_segments(path: str | None):
    if path is None or not str(path).strip():
        return None
    segment_path = os.path.abspath(str(path).strip())
    if not os.path.isfile(segment_path):
        raise RuntimeError(f"Tau morphology segment file not found: {segment_path}")
    if segment_path.endswith(".parquet"):
        df = pd.read_parquet(segment_path)
    elif segment_path.endswith(".csv"):
        df = pd.read_csv(segment_path)
    else:
        raise RuntimeError("PINN_TAU_MORPH_SEGMENTS must point to a .parquet or .csv file.")
    required = {"window_key", "dia_start", "dia_end", "p_inf", "target_T"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Tau morphology segment file missing columns: {sorted(missing)}")
    segments: dict[str, list[tuple[int, int, float, int]]] = {}
    for row in df.to_dict(orient="records"):
        key = str(row["window_key"])
        start = int(row["dia_start"])
        end = int(row["dia_end"])
        p_inf = float(row["p_inf"])
        target_T = int(row["target_T"])
        if end <= start or target_T <= 0 or not math.isfinite(p_inf):
            raise RuntimeError(f"Invalid tau morphology segment row for key={key}: {row}")
        segments.setdefault(key, []).append((start, end, p_inf, target_T))
    if not segments:
        raise RuntimeError(f"Tau morphology segment file contains no usable rows: {segment_path}")
    n_rows = int(sum(len(v) for v in segments.values()))
    print(
        f"[training_v9_tau] Loaded precomputed tau morphology segments: "
        f"{n_rows} segments across {len(segments)} windows from {segment_path}"
    )
    return segments


def _true_abp_tau_segments_from_matched_peaks(
    true_abp_np: np.ndarray,
    peaks: list[int],
    fs: float,
    cfg: TauDecayTargetConfig,
) -> list[tuple[int, int, float]]:
    """Return true-ABP post-notch segments and fitted asymptote for waveform tau loss."""

    x = np.asarray(true_abp_np, dtype=np.float64)
    if x.ndim != 1 or x.size < 8 or not np.isfinite(x).all():
        return []
    clean_peaks = sorted(set(int(p) for p in peaks if 0 <= int(p) < x.size))
    if len(clean_peaks) < 3:
        return []

    start_margin = int(round(float(cfg.diastolic_start_margin_sec) * float(fs)))
    end_margin = int(round(float(cfg.diastolic_end_margin_sec) * float(fs)))
    min_segment = max(3, int(round(float(cfg.min_segment_sec) * float(fs))))
    out: list[tuple[int, int, float]] = []

    for k in range(1, len(clean_peaks) - 1):
        prev_peak = clean_peaks[k - 1]
        peak = clean_peaks[k]
        next_peak = clean_peaks[k + 1]
        if not (0 <= prev_peak < peak < next_peak <= x.size):
            continue

        prev_seg = x[prev_peak:peak]
        next_seg = x[peak:next_peak]
        if prev_seg.size < 3 or next_seg.size < 3:
            continue
        prev_trough = int(prev_peak + int(np.argmin(prev_seg)))
        next_trough = int(peak + int(np.argmin(next_seg)))
        if not (prev_trough < peak < next_trough):
            continue

        pulse_pressure = float(x[peak] - min(x[prev_trough], x[next_trough]))
        if not np.isfinite(pulse_pressure) or pulse_pressure <= 0.0:
            continue

        notch = _detect_dicrotic_notch_index(
            x,
            peak_idx=peak,
            next_trough_idx=next_trough,
            fs=fs,
            pulse_pressure=pulse_pressure,
            cfg=cfg,
        )
        if notch is None:
            continue
        dia_start = int(notch) + start_margin
        dia_end = int(next_trough) - end_margin
        if dia_end - dia_start < min_segment:
            continue

        fit = _fit_log_decay_tau_profile(x[dia_start:dia_end], fs=fs, cfg=cfg)
        if fit is None:
            continue
        out.append((dia_start, dia_end, float(fit["p_inf"])))
    return out


def _segment_log_decay_tau(
    pred_segment: torch.Tensor,
    dt_segment: torch.Tensor,
    p_inf: float,
    *,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    if pred_segment.ndim != 1 or dt_segment.ndim != 1 or pred_segment.numel() < 3:
        return None
    pred_segment = pred_segment.float()
    dt_segment = dt_segment.float()
    shifted = pred_segment - pred_segment.new_tensor(float(p_inf))
    if not bool(torch.isfinite(shifted).all().detach().cpu()) or not bool((shifted > eps).all().detach().cpu()):
        return None
    if not bool(torch.isfinite(dt_segment).all().detach().cpu()):
        return None
    if pred_segment.numel() == 1:
        return None
    t = torch.cat([dt_segment.new_zeros(1), torch.cumsum(dt_segment[:-1], dim=0)], dim=0)
    t = t - t.mean()
    sxx = t.pow(2).sum()
    if not bool(torch.isfinite(sxx).detach().cpu()) or float(sxx.detach().cpu()) <= 0.0:
        return None
    y = torch.log(shifted)
    y = y - y.mean()
    slope = (t * y).sum() / sxx
    if not bool(torch.isfinite(slope).detach().cpu()) or float(slope.detach().cpu()) >= 0.0:
        return None
    tau = -1.0 / slope
    if not bool(torch.isfinite(tau).detach().cpu()):
        return None
    return torch.log(tau)


def _tau_morphology_loss_if_enabled(
    batch,
    pred_bt: torch.Tensor,
    true_bt: torch.Tensor,
    mask_bt: torch.Tensor,
    dt_bt: torch.Tensor,
    *,
    source_T: int,
    target_T: int,
    target_log_tau: torch.Tensor,
    lambda_tau_morph_eff: float,
    beta: float,
    cfg: TauDecayTargetConfig,
    precomputed_segments: dict[str, list[tuple[int, int, float, int]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if lambda_tau_morph_eff <= 0.0:
        zero_n = torch.zeros((), device=pred_bt.device, dtype=torch.long)
        nan = pred_bt.new_tensor(float("nan"))
        return pred_bt.new_tensor(0.0), zero_n, zero_n, nan
    v9c.require_batch_keys(
        batch,
        ["matched_abp_peaks", "tau_obs_log", "tau_obs_se", "tau_weight", "tau_valid"],
        "tau morphology",
    )
    target_log_tau = target_log_tau.to(device=pred_bt.device, dtype=pred_bt.dtype).view(-1)
    tau_obs_se = batch["tau_obs_se"].to(device=pred_bt.device, dtype=pred_bt.dtype).view(-1)
    tau_weight = batch["tau_weight"].to(device=pred_bt.device, dtype=pred_bt.dtype).view(-1)
    tau_valid = batch["tau_valid"].to(device=pred_bt.device).view(-1).bool()

    window_losses: list[torch.Tensor] = []
    window_weights: list[torch.Tensor] = []
    window_tau_means: list[torch.Tensor] = []
    n_segments = 0

    B = int(pred_bt.shape[0])
    for b in range(B):
        if not bool(tau_valid[b].detach().cpu()):
            continue
        if not bool(torch.isfinite(target_log_tau[b]).detach().cpu()):
            continue
        if not bool(torch.isfinite(tau_obs_se[b]).detach().cpu()) or float(tau_obs_se[b].detach().cpu()) <= 0.0:
            continue
        if not bool(torch.isfinite(tau_weight[b]).detach().cpu()) or float(tau_weight[b].detach().cpu()) <= 0.0:
            continue

        valid_mask = mask_bt[b].detach().bool().cpu().numpy()
        if valid_mask.size != int(target_T):
            continue
        dt_row = dt_bt[b].detach()
        if not bool(torch.isfinite(dt_row).all().cpu()):
            continue
        fs = float(1.0 / torch.median(dt_row.float()).detach().cpu().item())
        if not np.isfinite(fs) or fs <= 0.0:
            continue

        if precomputed_segments is not None:
            key = _tau_morphology_window_key(batch, b)
            segment_rows = precomputed_segments.get(key, [])
            segments = []
            for start, end, p_inf, segment_target_T in segment_rows:
                if int(segment_target_T) != int(target_T):
                    raise RuntimeError(
                        f"Precomputed tau morphology target_T mismatch for key={key}: "
                        f"file has {segment_target_T}, training uses {target_T}."
                    )
                segments.append((int(start), int(end), float(p_inf)))
        else:
            peaks = _rescaled_matched_abp_peaks(batch, b, source_T, target_T)
            true_np = true_bt[b].detach().float().cpu().numpy()
            segments = _true_abp_tau_segments_from_matched_peaks(true_np, peaks, fs, cfg)
        segment_losses: list[torch.Tensor] = []
        segment_log_tau: list[torch.Tensor] = []
        for start, end, p_inf in segments:
            if not bool(valid_mask[start:end].all()):
                continue
            log_tau_wave = _segment_log_decay_tau(
                pred_bt[b, start:end],
                dt_bt[b, start:end],
                p_inf,
            )
            if log_tau_wave is None:
                continue
            seg_loss = F.smooth_l1_loss(
                log_tau_wave,
                target_log_tau[b],
                beta=float(beta),
                reduction="mean",
            )
            segment_losses.append(seg_loss)
            segment_log_tau.append(log_tau_wave.detach())
        if not segment_losses:
            continue

        window_losses.append(torch.stack(segment_losses).mean())
        window_weights.append(tau_weight[b] / tau_obs_se[b].pow(2))
        window_tau_means.append(torch.exp(torch.stack(segment_log_tau)).mean())
        n_segments += len(segment_losses)

    if not window_losses:
        zero_n = torch.zeros((), device=pred_bt.device, dtype=torch.long)
        nan = pred_bt.new_tensor(float("nan"))
        return pred_bt.new_tensor(0.0), zero_n, zero_n, nan

    losses = torch.stack(window_losses)
    weights = torch.stack(window_weights)
    loss = (weights * losses).sum() / weights.sum()
    n_windows = torch.tensor(len(window_losses), device=pred_bt.device, dtype=torch.long)
    n_segments_t = torch.tensor(int(n_segments), device=pred_bt.device, dtype=torch.long)
    tau_mean = torch.stack(window_tau_means).mean()
    return loss, n_windows, n_segments_t, tau_mean


# =============================================================
# Training loop
# =============================================================


def train_pinn_deeponet_v6_tau(
    dataloader,
    *,
    co_prior_map=None,
    n_epochs=90,
    lr=1e-4,
    device="cpu",
    nx=16,
    model_factory=None,
    encoder_temporal_stride=1,
    operator_time_points=None,
    grad_clip=1.0,
    phase_pretrain_epochs=8,
    waveform_end_epoch=32,
    physics_end_epoch=60,
    lambda_phase_anchor=1.0,
    lambda_phase_smooth=0.1,
    lambda_phase_mono=0.5,
    lambda_phys=0.20,
    lambda_param=0.05,
    lambda_distill=2.0,
    lambda_co=0.0,
    lambda_co_supervised=0.0,
    lambda_nibp=0.0,
    lambda_cvp=0.0,
    lambda_tau_dia=0.25,
    lambda_tau_morph=0.50,
    lambda_tau_morph_internal=0.0,
    tau_morph_start_epoch=None,
    tau_morph_full_epoch=None,
    tau_morph_beta=0.20,
    tau_morph_internal_detach_ref=True,
    w_abs=1.0,
    w_z=0.25,
    w_der=2.0,
    w_beat=0.5,
    w_abp_stats=6.5,
    w_corr=0.5,
    w_couple=10.0,
    w_wk=0.1,
    w_rtot=0.5,
    w_pc_periodic=0.1,
    w_mass=0.10,
    w_flow_smooth=0.12,
    w_latent_smooth=0.05,
    w_pressure_aux=0.25,
    w_subject_param_consistency=0.05,
    w_prior=0.75,
    w_pde_max=0.05,
    pde_collocation_time=24,
    pde_collocation_x=6,
    group_ramp_epochs=12,
    wk_delta_initial_scale=0.02,
    wk_delta_full_epoch=None,
    physics_warm_end_epoch=None,
    direct_alpha_physics_start=1.0,
    direct_alpha_physics_warm_end=None,
    direct_alpha_physics_end=0.50,
    direct_alpha_joint=0.35,
    direct_alpha_joint_ramp_epochs=8,
    route_slow_wk_rollout=True,
    lambda_pv_anchor=0.10,
    pv_anchor_mmHg=5.0,
    pv_anchor_sigma_mmHg=3.0,
    lambda_qscale_gauge=0.10,
    use_nibp_input=True,
    require_nibp_input_valid=False,
    nibp_full_epoch=None,
    co_supervised_start_epoch=None,
    co_supervised_full_epoch=None,
    cvp_full_epoch=None,
    tau_dia_start_epoch=None,
    tau_dia_full_epoch=None,
    use_amp=False,
    max_steps_per_epoch=None,
    warmup_steps=0,
    min_lr_ratio=0.01,
    ema_decay=0.995,
    epoch_end_callback=None,
    eval_dataloader=None,
    eval_max_steps=None,
    eval_every_n_epochs=1,
    best_model_metric="val_tau_log_mae",
    non_blocking=False,
    tau_morph_segments_by_key=None,
    **kwargs,
):
    if kwargs:
        raise TypeError(f"Unexpected train_pinn_deeponet_v6_tau kwargs: {sorted(kwargs)}")

    if isinstance(device, str):
        device = torch.device(device)
    v9c.configure_gpu_runtime(device)

    if not (0 <= int(phase_pretrain_epochs) < int(waveform_end_epoch) < int(physics_end_epoch) < int(n_epochs)):
        raise ValueError(
            "Stage boundaries must satisfy "
            "0 <= phase_pretrain_epochs < waveform_end_epoch < physics_end_epoch < n_epochs."
        )
    alpha_values = {
        "direct_alpha_physics_start": direct_alpha_physics_start,
        "direct_alpha_physics_end": direct_alpha_physics_end,
        "direct_alpha_joint": direct_alpha_joint,
    }
    if direct_alpha_physics_warm_end is not None:
        alpha_values["direct_alpha_physics_warm_end"] = direct_alpha_physics_warm_end
    for name, value in alpha_values.items():
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    if physics_warm_end_epoch is not None:
        physics_warm_end_epoch = int(physics_warm_end_epoch)
        if not (int(waveform_end_epoch) < physics_warm_end_epoch < int(physics_end_epoch)):
            raise ValueError(
                "physics_warm_end_epoch must satisfy "
                "waveform_end_epoch < physics_warm_end_epoch < physics_end_epoch."
            )
        if direct_alpha_physics_warm_end is None:
            raise ValueError("direct_alpha_physics_warm_end is required when physics_warm_end_epoch is set.")
    if float(lambda_co) != 0.0 or float(lambda_co_supervised) != 0.0 or float(lambda_cvp) != 0.0:
        raise ValueError("training_v9_tauA.py trains tau and waveform only; CO/CVP losses must be zero.")
    if float(lambda_tau_morph) < 0.0:
        raise ValueError("lambda_tau_morph must be non-negative.")
    if float(lambda_tau_morph_internal) != 0.0:
        raise ValueError("training_v9_tauA.py includes morphology target A only; internal morphology loss must be zero.")
    if float(tau_morph_beta) <= 0.0:
        raise ValueError("tau_morph_beta must be positive.")

    if model_factory is None:
        model = DirectWaveformDeepONet1DFlowPINN(
            nx=nx,
            encoder_temporal_stride=encoder_temporal_stride,
        )
    else:
        model = model_factory(
            nx=nx,
            encoder_temporal_stride=encoder_temporal_stride,
        )
    model = model.to(device)
    model = v9c.optional_compile_model(model, device)

    if max_steps_per_epoch is not None and max_steps_per_epoch > 0:
        steps_per_epoch = int(max_steps_per_epoch)
    else:
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            raise ValueError("dataloader has no len() but max_steps_per_epoch is not set.")
    total_steps = int(n_epochs) * int(steps_per_epoch)

    if wk_delta_full_epoch is None:
        wk_delta_full_epoch = physics_end_epoch
    if nibp_full_epoch is None:
        nibp_full_epoch = waveform_end_epoch
    if co_supervised_start_epoch is None:
        co_supervised_start_epoch = waveform_end_epoch + 1
    if co_supervised_full_epoch is None:
        co_supervised_full_epoch = physics_end_epoch
    if cvp_full_epoch is None:
        cvp_full_epoch = physics_end_epoch
    if tau_dia_start_epoch is None:
        tau_dia_start_epoch = waveform_end_epoch + 1
    if tau_dia_full_epoch is None:
        tau_dia_full_epoch = physics_end_epoch
    if tau_morph_start_epoch is None:
        tau_morph_start_epoch = tau_dia_start_epoch
    if tau_morph_full_epoch is None:
        tau_morph_full_epoch = tau_dia_full_epoch

    print(f"[training_v9_tau] Schedule: {steps_per_epoch} steps/epoch x {n_epochs} epochs = {total_steps} total")
    print(f"[training_v9_tau] Warmup: {warmup_steps} steps | phase pretrain: epochs 1-{phase_pretrain_epochs}")
    print(
        "[training_v9_tau] Curriculum: "
        f"waveform={phase_pretrain_epochs + 1}-{waveform_end_epoch}, "
        f"physics={waveform_end_epoch + 1}-{physics_end_epoch}, "
        f"joint={physics_end_epoch + 1}-{n_epochs}"
    )
    if physics_warm_end_epoch is not None:
        print(
            "[training_v9_tau] Physics alpha knot: "
            f"warm_start={waveform_end_epoch + 1}-{physics_warm_end_epoch}, "
            f"ramp={physics_warm_end_epoch + 1}-{physics_end_epoch}"
        )
    if direct_alpha_physics_warm_end is not None:
        print(
            "[training_v9_tau] Direct alpha: "
            "waveform=1.000, "
            f"physics_start={direct_alpha_physics_start:.3f}, "
            f"physics_warm_end={float(direct_alpha_physics_warm_end):.3f}, "
            f"physics_end={direct_alpha_physics_end:.3f}, "
            f"joint={direct_alpha_joint:.3f}"
        )
    else:
        print(
            "[training_v9_tau] Direct alpha: "
            "waveform=1.000, physics_start=1.000, "
            f"physics_end={direct_alpha_physics_end:.3f}, "
            f"joint={direct_alpha_joint:.3f}"
        )
    print(
        "[training_v9_tau] NIBP forward input: "
        f"enabled={bool(use_nibp_input)} require_all_valid={bool(require_nibp_input_valid)}"
    )
    print(
        "[training_v9_tauA] slow-state routing: "
        f"detach_wk_rollout={int(bool(route_slow_wk_rollout))}; "
        f"Pv_anchor={lambda_pv_anchor} target={pv_anchor_mmHg} sigma={pv_anchor_sigma_mmHg}; "
        f"Qscale_gauge={lambda_qscale_gauge}"
    )

    opt = v9c.build_fused_optimizer(model, lr=lr, weight_decay=1e-5, device=device)
    scheduler = v9c.build_scheduler(opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio)

    use_cuda_amp = use_amp and device.type == "cuda"
    amp_dtype = v9c.resolve_amp_dtype(device) if use_cuda_amp else torch.float32
    use_grad_scaler = use_cuda_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    if use_cuda_amp:
        print(f"[training_v9_tau] AMP dtype: {amp_dtype}; grad_scaler={use_grad_scaler}")
    if eval_dataloader is not None and int(eval_every_n_epochs) <= 0:
        raise ValueError("eval_every_n_epochs must be positive when eval_dataloader is provided.")
    if lambda_pv_anchor < 0.0:
        raise ValueError("lambda_pv_anchor must be non-negative.")
    if lambda_qscale_gauge < 0.0:
        raise ValueError("lambda_qscale_gauge must be non-negative.")
    if pv_anchor_sigma_mmHg <= 0.0:
        raise ValueError("pv_anchor_sigma_mmHg must be positive.")
    v9c._unwrap_compiled_model(model).set_slow_wk_rollout_detach(bool(route_slow_wk_rollout))

    ema = v9c.ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    teacher_model = None

    METRIC_KEYS = [
        "loss", "wave", "phase", "distill", "phys", "param", "aux",
        "nibp_aux", "co_sup", "cvp_aux", "tau_dia", "tau_morph", "tau_morph_internal",
        "wk", "rtot", "pc_periodic", "couple", "pde", "mass",
        "flow_smooth", "latent_smooth", "prior", "pressure_aux",
        "pv_anchor", "qscale_gauge",
        "abp_stats", "beat", "abs", "corr", "der",
        "wave_w", "phys_w", "param_w", "distill_w", "nibp_w",
        "co_sup_w", "cvp_w", "tau_dia_w", "tau_morph_w", "tau_morph_internal_w",
        "pv_anchor_w", "qscale_gauge_w",
        "nibp_n", "co_sup_n", "cvp_n", "tau_dia_n",
        "tau_morph_n", "tau_morph_segments", "tau_morph_internal_n", "tau_morph_internal_segments",
        "tau_obs_mean", "tau_pred_mean", "tau_morph_mean", "tau_morph_internal_mean",
        "grad_norm", "direct_alpha",
    ]
    PARAM_KEYS = [
        "R1", "R2", "C", "tau", "Pv",
        "Pext", "beta", "c_p", "C_f", "excess_gain", "abp_residual_scale",
        "Qscale_mean", "Qscale_p05", "Qscale_p50", "Qscale_p95",
        "wk_delta_scale", "lr",
    ]
    VAL_METRIC_KEYS = [
        "val_mae", "val_rmse", "val_corr",
        "val_sbp_mae", "val_dbp_mae", "val_map_mae",
        "val_nibp_sbp_mae", "val_nibp_dbp_mae", "val_nibp_map_mae",
        "val_co_mae", "val_cvp_mae",
        "val_tau_log_mae", "val_tau_mae", "val_tau_n",
        "val_nibp_n", "val_co_n", "val_cvp_n",
        "val_n_windows", "val_loss",
    ]
    history = {k: [] for k in METRIC_KEYS + PARAM_KEYS + VAL_METRIC_KEYS}
    history["wk_epoch_summary"] = []
    history["stage"] = []

    lower_is_better = {
        "val_mae", "val_rmse", "val_loss", "val_sbp_mae", "val_dbp_mae", "val_map_mae",
        "val_nibp_sbp_mae", "val_nibp_dbp_mae", "val_nibp_map_mae",
        "val_co_mae", "val_cvp_mae", "val_tau_log_mae", "val_tau_mae",
    }
    best_val_score = float("inf") if best_model_metric in lower_is_better else float("-inf")
    best_epoch = -1
    best_model_state = None
    best_ema_state = None
    print(
        f"[training_v9_tau] Best checkpoint metric: {best_model_metric}; "
        "non-finite validation is skipped, and tau metrics are not eligible "
        "until the supervised tau loss is active."
    )
    tau_morph_cfg = TauDecayTargetConfig()

    for epoch in range(1, int(n_epochs) + 1):
        model.train()
        stage = _curriculum_stage_v6(epoch, phase_pretrain_epochs, waveform_end_epoch, physics_end_epoch)
        _set_trainable_stage_v6(model, stage)

        alpha = _direct_alpha_for_epoch(
            epoch,
            stage,
            waveform_end_epoch=waveform_end_epoch,
            physics_end_epoch=physics_end_epoch,
            physics_warm_end_epoch=physics_warm_end_epoch,
            physics_start_alpha=direct_alpha_physics_start,
            physics_warm_end_alpha=direct_alpha_physics_warm_end,
            physics_end_alpha=direct_alpha_physics_end,
            joint_alpha=direct_alpha_joint,
            joint_alpha_ramp_epochs=direct_alpha_joint_ramp_epochs,
        )
        v9c._unwrap_compiled_model(model).set_direct_blend_alpha(alpha)

        if stage in {"phase", "waveform"}:
            wk_delta_scale = 0.0
            active_wk = (False, False, False, False)
            train_reference = (False, False, False, False)
            pc0_active = False
            physics_ramp = 0.0
        else:
            if teacher_model is None:
                teacher_model = _make_waveform_teacher(model, device)
                print(f"[training_v9_tau] Frozen waveform teacher captured at epoch {epoch - 1}.")
            wk_delta_scale = v9c._linear_ramp_between_epochs(
                epoch,
                waveform_end_epoch,
                int(wk_delta_full_epoch),
                initial_scale=wk_delta_initial_scale,
            )
            active_wk = (True, True, True, True)
            train_reference = (True, True, True, True)
            pc0_active = True
            physics_ramp = v9c._post_recon_ramp(epoch, waveform_end_epoch, group_ramp_epochs)

        v9c._unwrap_compiled_model(model).set_wk_adaptation(
            delta_scale=wk_delta_scale,
            active_wk=active_wk,
            train_reference=train_reference,
            pc0_active=pc0_active,
        )

        lambda_phys_eff = lambda_phys * physics_ramp
        lambda_param_eff = lambda_param * physics_ramp
        lambda_distill_eff = lambda_distill if stage in {"physics", "joint"} else 0.0
        lambda_nibp_eff = lambda_nibp * v9c._linear_ramp_between_epochs(
            epoch,
            max(1, phase_pretrain_epochs + 1),
            int(nibp_full_epoch),
        )
        lambda_co_eff = lambda_co * physics_ramp
        lambda_co_supervised_eff = lambda_co_supervised * v9c._linear_ramp_between_epochs(
            epoch,
            int(co_supervised_start_epoch),
            int(co_supervised_full_epoch),
        )
        lambda_cvp_eff = lambda_cvp * v9c._linear_ramp_between_epochs(
            epoch,
            waveform_end_epoch + 1,
            int(cvp_full_epoch),
        )
        lambda_tau_dia_eff = lambda_tau_dia * v9c._linear_ramp_between_epochs(
            epoch,
            int(tau_dia_start_epoch),
            int(tau_dia_full_epoch),
        )
        lambda_tau_morph_eff = lambda_tau_morph * v9c._linear_ramp_between_epochs(
            epoch,
            int(tau_morph_start_epoch),
            int(tau_morph_full_epoch),
        )
        lambda_tau_morph_internal_eff = lambda_tau_morph_internal * v9c._linear_ramp_between_epochs(
            epoch,
            int(tau_morph_start_epoch),
            int(tau_morph_full_epoch),
        )
        lambda_pv_anchor_eff = lambda_pv_anchor * physics_ramp
        lambda_qscale_gauge_eff = lambda_qscale_gauge * physics_ramp
        w_pde = w_pde_max * physics_ramp

        run = {k: 0.0 for k in METRIC_KEYS}
        param_run = {k: 0.0 for k in ["R1", "R2", "C", "tau", "Pv"]}
        param_batches = 0
        n_batches = 0
        wk_epoch_rows = []
        qscale_epoch_values = []

        cap = max_steps_per_epoch
        epoch_iter = islice(dataloader, cap) if cap else dataloader
        pbar = tqdm(epoch_iter, total=cap, desc=f"Epoch {epoch}/{n_epochs}", leave=False)

        for _, batch in enumerate(pbar, start=1):
            x = batch["input"].to(device, non_blocking=non_blocking)
            y = batch["output"].to(device, non_blocking=non_blocking)
            mask = batch["output_mask"].to(device, non_blocking=non_blocking)
            phi_ref = batch["phase_ref"].to(device, non_blocking=non_blocking)
            phi_mask = batch["phase_mask"].to(device, non_blocking=non_blocking)
            dt_in = batch["dt"]

            B, source_T = x.shape[0], x.shape[1]
            raw_dt_b = v9c._broadcast_dt(dt_in, B, source_T, x.device, x.dtype)
            raw_dt_b = v9c._calibrate_dt(raw_dt_b)
            raw_r_peaks_list = v9c.extract_r_peaks_per_sample(batch, B)

            target_T = int(operator_time_points) if operator_time_points is not None else source_T
            x, y, mask, dt_b = v9c._resample_to_operator_grid(x, y, mask, raw_dt_b, target_T)
            v9c._assert_operator_grid_dt_consistency(raw_dt_b, dt_b, source_T, target_T)
            phi_ref = v9c._resample_time_tensor(phi_ref, target_T)
            phi_mask = v9c._resample_time_mask(phi_mask, target_T)
            nibp_features = (
                _nibp_features_from_batch(
                    batch,
                    device=x.device,
                    dtype=x.dtype,
                    require_all_valid=bool(require_nibp_input_valid),
                )
                if use_nibp_input
                else None
            )

            T = x.shape[1]
            dt_bt = dt_b.squeeze(-1)
            r_peaks_local = [
                v9c._rescale_indices_to_length(peaks, source_T, T)
                for peaks in raw_r_peaks_list
            ]
            beat_bounds = v9c._beat_bounds_from_local_peaks(r_peaks_local, T)
            phi_ref_bt = phi_ref.squeeze(-1)

            opt.zero_grad(set_to_none=True)

            if stage == "phase":
                ecg = x[..., 0:1]
                with torch.amp.autocast("cuda", enabled=use_cuda_amp, dtype=amp_dtype):
                    phi_hat_bt = model.phase_net(ecg, dt_b).squeeze(-1)
                    phase_loss = _phase_loss(
                        phi_hat_bt,
                        phi_ref_bt,
                        phi_mask,
                        r_peaks_local,
                        lambda_phase_anchor=lambda_phase_anchor,
                        lambda_phase_smooth=lambda_phase_smooth,
                        lambda_phase_mono=lambda_phase_mono,
                    )
                    total_loss = phase_loss
                    zero = phase_loss.new_tensor(0.0)
                    rec_loss = zero
                    L_distill = zero
                    phys_loss = zero
                    param_loss = zero
                    aux_loss = zero
                    L_nibp = zero
                    Lco_sup = zero
                    L_cvp = zero
                    L_tau_dia = zero
                    L_tau_morph = zero
                    L_tau_morph_internal = zero
                    n_nibp = torch.zeros((), device=x.device, dtype=torch.long)
                    n_co_sup = torch.zeros((), device=x.device, dtype=torch.long)
                    n_cvp = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_dia = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph_segments = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph_internal = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph_internal_segments = torch.zeros((), device=x.device, dtype=torch.long)
                    tau_obs_epoch_mean = zero.new_tensor(float("nan"))
                    tau_pred_epoch_mean = zero.new_tensor(float("nan"))
                    tau_morph_epoch_mean = zero.new_tensor(float("nan"))
                    tau_morph_internal_epoch_mean = zero.new_tensor(float("nan"))
                    loss_parts = {"abs": zero, "corr": zero, "der": zero, "beat": zero, "abp_stats": zero}
                    Lcouple = Lwk = L_rtot = L_pc_periodic = Lpde = Lmass = zero
                    L_smooth_total = L_latent_smooth = L_prior = L_pressure_aux = zero
                    L_pv_anchor = L_qscale_gauge = zero
            else:
                with torch.amp.autocast("cuda", enabled=use_cuda_amp, dtype=amp_dtype):
                    if stage == "waveform":
                        y_hat, fields, aux = v9c._unwrap_compiled_model(model).forward_waveform(
                            x,
                            dt_seconds=dt_b,
                            nibp_features=nibp_features,
                        )
                    else:
                        y_hat, fields, aux = model(x, dt_seconds=dt_b, nibp_features=nibp_features)

                    phi_hat_bt = fields["phi"].squeeze(-1)
                    phase_loss = _phase_loss(
                        phi_hat_bt,
                        phi_ref_bt,
                        phi_mask,
                        r_peaks_local,
                        lambda_phase_anchor=lambda_phase_anchor,
                        lambda_phase_smooth=lambda_phase_smooth,
                        lambda_phase_mono=lambda_phase_mono,
                    )

                    y_hat_bt = v9c._to_bt2d(y_hat)
                    y_bt = v9c._to_bt2d(y)
                    y_hat_bt, y_bt, mask = v9c._align_bt(y_hat_bt, y_bt, mask)
                    rec_loss, loss_parts = _waveform_reconstruction_loss(
                        v9c._unwrap_compiled_model(model),
                        y_hat_bt,
                        y_bt,
                        mask,
                        dt_bt,
                        beat_bounds,
                        w_abs=w_abs,
                        w_z=w_z,
                        w_der=w_der,
                        w_beat=w_beat,
                        w_abp_stats=w_abp_stats,
                        w_corr=w_corr,
                    )
                    L_nibp, n_nibp = _nibp_loss_if_enabled(batch, y_hat_bt, mask, lambda_nibp_eff)

                    zero = y_hat_bt.new_tensor(0.0)
                    L_distill = zero
                    phys_loss = zero
                    param_loss = zero
                    Lco_sup = zero
                    L_cvp = zero
                    L_tau_dia = zero
                    L_tau_morph = zero
                    L_tau_morph_internal = zero
                    n_co_sup = torch.zeros((), device=x.device, dtype=torch.long)
                    n_cvp = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_dia = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph_segments = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph_internal = torch.zeros((), device=x.device, dtype=torch.long)
                    n_tau_morph_internal_segments = torch.zeros((), device=x.device, dtype=torch.long)
                    tau_obs_epoch_mean = zero.new_tensor(float("nan"))
                    tau_pred_epoch_mean = zero.new_tensor(float("nan"))
                    tau_morph_epoch_mean = zero.new_tensor(float("nan"))
                    tau_morph_internal_epoch_mean = zero.new_tensor(float("nan"))
                    Lcouple = Lwk = L_rtot = L_pc_periodic = Lpde = Lmass = zero
                    L_smooth_total = L_latent_smooth = L_prior = L_pressure_aux = zero
                    L_pv_anchor = L_qscale_gauge = zero

                    if stage in {"physics", "joint"}:
                        if teacher_model is None:
                            raise RuntimeError("Waveform teacher must exist before physics or joint stages.")
                        with torch.no_grad():
                            y_teacher, _, _ = teacher_model.forward_waveform(
                                x,
                                dt_seconds=dt_b,
                                nibp_features=nibp_features,
                            )
                            y_teacher_bt = v9c._to_bt2d(y_teacher)
                            y_teacher_bt, _, _ = v9c._align_bt(y_teacher_bt, y_bt, mask)
                        L_distill = v9c.masked_mse(y_hat_bt, y_teacher_bt, mask)

                        Qin = aux["Q_in"]
                        Qin_bc = aux["Q_in_bc"]
                        Q = fields["Q"]
                        P = fields["P"]
                        wk = aux["wk"]
                        wk_rollout = aux["wk_rollout"]
                        P_L = P[..., -1]
                        P_wk = fields["P_wk"]
                        Lcouple = v9c.masked_mse(P_L, P_wk, mask)
                        _, Lwk = model.bc_losses(Q, Qin_bc, P, dt_bt, wk=wk_rollout, mask_bt=mask)
                        if w_pde > 0 or w_pc_periodic > 0:
                            Lpde, L_pc_periodic = model.pde_and_periodic_collocation_loss(
                                aux["global_latent"],
                                wk_rollout,
                                fields["phi"],
                                dt_bt,
                                mask_bt=mask,
                                n_time=pde_collocation_time,
                                n_x=pde_collocation_x,
                            )
                            if w_pde <= 0:
                                Lpde = zero
                        else:
                            Lpde = zero
                            L_pc_periodic = zero

                        Q_L_bt = Q[..., -1]
                        L_rtot = model.rtotal_balance_loss(P_L, Q_L_bt, dt_bt, wk=wk_rollout, mask_bt=mask)
                        Q_mean_in = v9c.mean_flow(Qin, dt_bt, mask)
                        Q_mean_out = v9c.mean_flow(Q_L_bt, dt_bt, mask)
                        scale_Q = Q_mean_in.detach().abs().clamp_min(1.0)
                        Lmass = ((Q_mean_in - Q_mean_out) / scale_Q).pow(2).mean()

                        L_flow_smooth = v9c.l2_second_derivative_time(Q_L_bt, dt_bt, mask_bt=mask)
                        L_Qin_smooth = v9c.l2_second_derivative_time(Qin, dt_bt, mask_bt=mask)
                        L_smooth_total = L_flow_smooth + L_Qin_smooth
                        L_latent_smooth = v9c.latent_smoothness_loss(fields["abp_context"], mask)
                        L_pressure_aux = (
                            v9c.normalized_masked_mse(fields["P_prox"], y_bt, mask)
                            + 0.5 * v9c.window_sys_dia_map_loss_soft(fields["P_prox"], y_bt, mask)
                        )
                        L_prior = model.windkessel_prior_loss(wk) + model.physiologic_barrier_loss(wk)
                        L_pv_anchor = _pv_anchor_loss_if_enabled(
                            wk,
                            lambda_pv_anchor_eff,
                            anchor_mmHg=pv_anchor_mmHg,
                            sigma_mmHg=pv_anchor_sigma_mmHg,
                        )
                        L_qscale_gauge = _qscale_gauge_loss_if_enabled(aux["Qscale"], lambda_qscale_gauge_eff)

                        sids = v9c.get_subject_ids_from_batch(batch, B)
                        L_subject_param = v9c.within_subject_parameter_consistency_loss(sids, wk)
                        if lambda_co_eff > 0:
                            co0_b = _strict_co_prior_from_batch(sids, co_prior_map, x.device, x.dtype)
                            co_proxy = 0.5 * (Q_mean_in.abs() + Q_mean_out.abs())
                            Lco = F.smooth_l1_loss(
                                torch.log(co_proxy + 1e-6),
                                torch.log(co0_b.abs() + 1e-6),
                                beta=0.35,
                            )
                        else:
                            Lco = zero

                        Lco_sup, n_co_sup = _co_supervised_loss_if_enabled(
                            batch,
                            aux["CO_head_L_min"].view(-1),
                            lambda_co_supervised_eff,
                        )
                        L_cvp, n_cvp = _cvp_loss_if_enabled(
                            batch,
                            wk["Pv"].view(-1),
                            lambda_cvp_eff,
                        )
                        log_tau_pred = torch.log(wk["tau"].view(-1).clamp_min(1e-4))
                        L_tau_dia, n_tau_dia, tau_obs_epoch_mean, tau_pred_epoch_mean = _tau_loss_if_enabled(
                            batch,
                            log_tau_pred,
                            lambda_tau_dia_eff,
                        )
                        tau_obs_log_for_morph = batch["tau_obs_log"].to(
                            device=y_hat_bt.device,
                            dtype=y_hat_bt.dtype,
                        ).view(-1)
                        L_tau_morph, n_tau_morph, n_tau_morph_segments, tau_morph_epoch_mean = (
                            _tau_morphology_loss_if_enabled(
                                batch,
                                y_hat_bt,
                                y_bt,
                                mask,
                                dt_bt,
                                source_T=source_T,
                                target_T=T,
                                target_log_tau=tau_obs_log_for_morph,
                                lambda_tau_morph_eff=lambda_tau_morph_eff,
                                beta=tau_morph_beta,
                                cfg=tau_morph_cfg,
                                precomputed_segments=tau_morph_segments_by_key,
                            )
                        )
                        if lambda_tau_morph_internal_eff > 0.0:
                            physics_bt = v9c._to_bt2d(fields["abp_physics"])
                            physics_bt, _, _ = v9c._align_bt(physics_bt, y_bt, mask)
                            internal_ref = log_tau_pred.detach() if tau_morph_internal_detach_ref else log_tau_pred
                            (
                                L_tau_morph_internal,
                                n_tau_morph_internal,
                                n_tau_morph_internal_segments,
                                tau_morph_internal_epoch_mean,
                            ) = _tau_morphology_loss_if_enabled(
                                batch,
                                physics_bt,
                                y_bt,
                                mask,
                                dt_bt,
                                source_T=source_T,
                                target_T=T,
                                target_log_tau=internal_ref,
                                lambda_tau_morph_eff=lambda_tau_morph_internal_eff,
                                beta=tau_morph_beta,
                                cfg=tau_morph_cfg,
                                precomputed_segments=tau_morph_segments_by_key,
                            )

                        phys_loss = (
                            w_couple * Lcouple
                            + w_wk * Lwk
                            + w_mass * Lmass
                            + w_pde * Lpde
                            + w_rtot * L_rtot
                            + w_pc_periodic * L_pc_periodic
                            + w_flow_smooth * L_smooth_total
                            + w_latent_smooth * L_latent_smooth
                            + w_pressure_aux * L_pressure_aux
                        )
                        param_loss = (
                            w_prior * L_prior
                            + w_subject_param_consistency * L_subject_param
                            + lambda_co_eff * Lco
                        )
                        wk_epoch_rows.append({
                            "R1": wk["R1"].detach().float().cpu(),
                            "R2": wk["R2"].detach().float().cpu(),
                            "C": wk["C"].detach().float().cpu(),
                            "tau": wk["tau"].detach().float().cpu(),
                            "Pv": wk["Pv"].detach().float().cpu(),
                            "R_total": wk["R_total"].detach().float().cpu(),
                        })
                        qscale_epoch_values.append(aux["Qscale"].detach().float().cpu().view(-1))
                        param_run["R1"] += wk["R1"].detach().mean().item()
                        param_run["R2"] += wk["R2"].detach().mean().item()
                        param_run["C"] += wk["C"].detach().mean().item()
                        param_run["tau"] += wk["tau"].detach().mean().item()
                        param_run["Pv"] += wk["Pv"].detach().mean().item()
                        param_batches += 1

                    aux_loss = (
                        lambda_nibp_eff * L_nibp
                        + lambda_co_supervised_eff * Lco_sup
                        + lambda_cvp_eff * L_cvp
                        + lambda_tau_dia_eff * L_tau_dia
                        + lambda_tau_morph_eff * L_tau_morph
                        + lambda_tau_morph_internal_eff * L_tau_morph_internal
                        + lambda_pv_anchor_eff * L_pv_anchor
                        + lambda_qscale_gauge_eff * L_qscale_gauge
                    )
                    total_loss = (
                        rec_loss
                        + lambda_distill_eff * L_distill
                        + lambda_phys_eff * phys_loss
                        + lambda_param_eff * param_loss
                        + aux_loss
                    )

            if use_grad_scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(opt)
                grad_norm = nn.utils.clip_grad_norm_(v9c._parameters_with_grad(model), grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(v9c._parameters_with_grad(model), grad_clip)
                opt.step()
                optimizer_stepped = True

            if optimizer_stepped:
                scheduler.step()
                if ema:
                    ema.update(model)

            run["loss"] += float(total_loss.item())
            run["wave"] += float(rec_loss.item())
            run["phase"] += float(phase_loss.item())
            run["distill"] += float(L_distill.item())
            run["phys"] += float(phys_loss.item())
            run["param"] += float(param_loss.item())
            run["aux"] += float(aux_loss.item())
            run["nibp_aux"] += float(L_nibp.item())
            run["co_sup"] += float(Lco_sup.item())
            run["cvp_aux"] += float(L_cvp.item())
            run["tau_dia"] += float(L_tau_dia.item())
            run["tau_morph"] += float(L_tau_morph.item())
            run["tau_morph_internal"] += float(L_tau_morph_internal.item())
            run["wk"] += float(Lwk.item())
            run["rtot"] += float(L_rtot.item())
            run["pc_periodic"] += float(L_pc_periodic.item())
            run["couple"] += float(Lcouple.item())
            run["pde"] += float(Lpde.item())
            run["mass"] += float(Lmass.item())
            run["flow_smooth"] += float(L_smooth_total.item())
            run["latent_smooth"] += float(L_latent_smooth.item())
            run["prior"] += float(L_prior.item())
            run["pressure_aux"] += float(L_pressure_aux.item())
            run["pv_anchor"] += float(L_pv_anchor.item())
            run["qscale_gauge"] += float(L_qscale_gauge.item())
            run["abp_stats"] += float(loss_parts["abp_stats"].item())
            run["beat"] += float(loss_parts["beat"].item())
            run["abs"] += float(loss_parts["abs"].item())
            run["corr"] += float(loss_parts["corr"].item())
            run["der"] += float(loss_parts["der"].item())
            run["wave_w"] += 1.0 if stage != "phase" else 0.0
            run["phys_w"] += float(lambda_phys_eff)
            run["param_w"] += float(lambda_param_eff)
            run["distill_w"] += float(lambda_distill_eff)
            run["nibp_w"] += float(lambda_nibp_eff)
            run["co_sup_w"] += float(lambda_co_supervised_eff)
            run["cvp_w"] += float(lambda_cvp_eff)
            run["tau_dia_w"] += float(lambda_tau_dia_eff)
            run["tau_morph_w"] += float(lambda_tau_morph_eff)
            run["tau_morph_internal_w"] += float(lambda_tau_morph_internal_eff)
            run["pv_anchor_w"] += float(lambda_pv_anchor_eff)
            run["qscale_gauge_w"] += float(lambda_qscale_gauge_eff)
            run["nibp_n"] += int(n_nibp.detach().cpu())
            run["co_sup_n"] += int(n_co_sup.detach().cpu())
            run["cvp_n"] += int(n_cvp.detach().cpu())
            run["tau_dia_n"] += int(n_tau_dia.detach().cpu())
            run["tau_morph_n"] += int(n_tau_morph.detach().cpu())
            run["tau_morph_segments"] += int(n_tau_morph_segments.detach().cpu())
            run["tau_morph_internal_n"] += int(n_tau_morph_internal.detach().cpu())
            run["tau_morph_internal_segments"] += int(n_tau_morph_internal_segments.detach().cpu())
            if int(n_tau_dia.detach().cpu()) > 0:
                n_tau_int = int(n_tau_dia.detach().cpu())
                run["tau_obs_mean"] += float(tau_obs_epoch_mean.detach().cpu()) * n_tau_int
                run["tau_pred_mean"] += float(tau_pred_epoch_mean.detach().cpu()) * n_tau_int
            if int(n_tau_morph.detach().cpu()) > 0:
                n_morph_int = int(n_tau_morph.detach().cpu())
                run["tau_morph_mean"] += float(tau_morph_epoch_mean.detach().cpu()) * n_morph_int
            if int(n_tau_morph_internal.detach().cpu()) > 0:
                n_morph_internal_int = int(n_tau_morph_internal.detach().cpu())
                run["tau_morph_internal_mean"] += (
                    float(tau_morph_internal_epoch_mean.detach().cpu()) * n_morph_internal_int
                )
            run["grad_norm"] += float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            run["direct_alpha"] += float(alpha)
            n_batches += 1

        COUNT_KEYS = {
            "nibp_n", "co_sup_n", "cvp_n", "tau_dia_n",
            "tau_morph_n", "tau_morph_segments",
            "tau_morph_internal_n", "tau_morph_internal_segments",
        }
        tau_obs_sum = run["tau_obs_mean"]
        tau_pred_sum = run["tau_pred_mean"]
        tau_morph_sum = run["tau_morph_mean"]
        tau_morph_internal_sum = run["tau_morph_internal_mean"]
        for k in run:
            if k not in COUNT_KEYS:
                run[k] /= max(1, n_batches)
        if int(run["tau_dia_n"]) > 0:
            run["tau_obs_mean"] = tau_obs_sum / float(run["tau_dia_n"])
            run["tau_pred_mean"] = tau_pred_sum / float(run["tau_dia_n"])
        else:
            run["tau_obs_mean"] = float("nan")
            run["tau_pred_mean"] = float("nan")
        if int(run["tau_morph_n"]) > 0:
            run["tau_morph_mean"] = tau_morph_sum / float(run["tau_morph_n"])
        else:
            run["tau_morph_mean"] = float("nan")
        if int(run["tau_morph_internal_n"]) > 0:
            run["tau_morph_internal_mean"] = tau_morph_internal_sum / float(run["tau_morph_internal_n"])
        else:
            run["tau_morph_internal_mean"] = float("nan")

        if run["nibp_w"] > 0.0 and int(run["nibp_n"]) == 0:
            raise RuntimeError(
                "NIBP supervision is enabled, but this epoch used zero valid NIBP labels. "
                "Check dataset auxiliary keys, label masks, and sampling coverage."
            )
        if run["co_sup_w"] > 0.0 and int(run["co_sup_n"]) == 0:
            raise RuntimeError(
                "CO supervision is enabled, but this epoch used zero valid CO labels. "
                "Check dataset auxiliary keys, label masks, and sampling coverage."
            )
        if run["cvp_w"] > 0.0 and int(run["cvp_n"]) == 0:
            raise RuntimeError(
                "CVP supervision is enabled, but this epoch used zero valid CVP labels. "
                "Check dataset auxiliary keys, label masks, and sampling coverage."
            )
        if run["tau_dia_w"] > 0.0 and int(run["tau_dia_n"]) == 0:
            raise RuntimeError(
                "Tau supervision is enabled, but this epoch used zero valid tau labels. "
                "Check tau-target emission, tau_valid masks, and sampling coverage."
            )
        if run["tau_morph_w"] > 0.0 and int(run["tau_morph_n"]) == 0:
            raise RuntimeError(
                "Tau morphology A is enabled, but this epoch used zero valid morphology windows. "
                "Check morphology segment keys, validity masks, and sampling coverage."
            )
        if run["tau_morph_internal_w"] > 0.0 and int(run["tau_morph_internal_n"]) == 0:
            raise RuntimeError(
                "Tau morphology B is enabled, but this epoch used zero valid internal morphology windows. "
                "Check internal morphology extraction, validity masks, and sampling coverage."
            )

        wk_epoch_summary = v9c._wk_summary_from_rows(wk_epoch_rows)
        for k in METRIC_KEYS:
            history[k].append(run[k])
        history["wk_epoch_summary"].append(wk_epoch_summary)
        history["stage"].append(stage)

        base_model = v9c._unwrap_compiled_model(model)
        if param_batches > 0:
            for k in param_run:
                param_run[k] /= param_batches
            history["R1"].append(param_run["R1"])
            history["R2"].append(param_run["R2"])
            history["C"].append(param_run["C"])
            history["tau"].append(param_run["tau"])
            history["Pv"].append(param_run["Pv"])
        else:
            history["R1"].append(float(base_model.R1.detach().cpu()))
            history["R2"].append(float(base_model.R2.detach().cpu()))
            history["C"].append(float(base_model.Cwk.detach().cpu()))
            history["tau"].append(float(base_model.tau.detach().cpu()))
            history["Pv"].append(float(base_model.Pv.detach().cpu()))
        history["Pext"].append(float(base_model.Pext.detach().cpu()))
        history["beta"].append(float(base_model.beta.detach().cpu()))
        history["c_p"].append(float(base_model.c_p.detach().cpu()))
        history["C_f"].append(float(base_model.C_f.detach().cpu()))
        history["excess_gain"].append(float(base_model.excess_gain.detach().cpu()))
        history["abp_residual_scale"].append(float(base_model.abp_residual_scale.detach().cpu()))
        if qscale_epoch_values:
            qscale_summary = v9c._tensor_summary_1d(torch.cat(qscale_epoch_values, dim=0))
            history["Qscale_mean"].append(qscale_summary["mean"])
            history["Qscale_p05"].append(qscale_summary["p05"])
            history["Qscale_p50"].append(qscale_summary["p50"])
            history["Qscale_p95"].append(qscale_summary["p95"])
        else:
            history["Qscale_mean"].append(float("nan"))
            history["Qscale_p05"].append(float("nan"))
            history["Qscale_p50"].append(float("nan"))
            history["Qscale_p95"].append(float("nan"))
        history["wk_delta_scale"].append(float(base_model._wk_delta_scale.detach().cpu()))
        history["lr"].append(opt.param_groups[0]["lr"])

        print(
            f"Ep {epoch:2d} | stage={stage} alpha={run['direct_alpha']:.3f} "
            f"| loss={run['loss']:.4f} wave={run['wave']:.4f} "
            f"distill={run['distill']:.4f}/{run['distill_w']:.3f} "
            f"phys={run['phys']:.4f}/{run['phys_w']:.3f} "
            f"param={run['param']:.4f}/{run['param_w']:.3f} aux={run['aux']:.4f} "
            f"| rho1={history['R1'][-1]:.3f} tau={history['tau'][-1]:.3f} "
            f"C_gauge={history['C'][-1]:.1f} "
            f"Pv={history['Pv'][-1]:.2f} wk_delta={history['wk_delta_scale'][-1]:.3f} "
            f"| pde={run['pde']:.3f} wk={run['wk']:.3f} rtot={run['rtot']:.3f} "
            f"mass={run['mass']:.3f} | "
            f"NIBP={run['nibp_aux']:.4f}/{run['nibp_w']:.3f}/n_epoch={int(run['nibp_n'])} "
            f"COsup={run['co_sup']:.4f}/{run['co_sup_w']:.3f}/n_epoch={int(run['co_sup_n'])} "
            f"CVP={run['cvp_aux']:.4f}/{run['cvp_w']:.3f}/n_epoch={int(run['cvp_n'])} "
            f"Tau={run['tau_dia']:.4f}/{run['tau_dia_w']:.3f}/n_epoch={int(run['tau_dia_n'])} "
            f"obs={run['tau_obs_mean']:.3f} pred={run['tau_pred_mean']:.3f} "
            f"MorphA={run['tau_morph']:.4f}/{run['tau_morph_w']:.3f}/"
            f"n={int(run['tau_morph_n'])}/seg={int(run['tau_morph_segments'])}/"
            f"tau={run['tau_morph_mean']:.3f} "
            f"MorphB={run['tau_morph_internal']:.4f}/{run['tau_morph_internal_w']:.3f}/"
            f"n={int(run['tau_morph_internal_n'])}/seg={int(run['tau_morph_internal_segments'])}/"
            f"tau={run['tau_morph_internal_mean']:.3f} "
            f"PvA={run['pv_anchor']:.4f}/{run['pv_anchor_w']:.3f} "
            f"Qg={run['qscale_gauge']:.4f}/{run['qscale_gauge_w']:.3f} "
            f"| Uscale50={history['Qscale_p50'][-1]:.3f} "
            f"| trainable={v9c._count_trainable_parameters(model)} "
            f"grad={run['grad_norm']:.2f} lr={opt.param_groups[0]['lr']:.2e}"
        )

        run_val = (
            eval_dataloader is not None
            and (epoch % int(eval_every_n_epochs) == 0 or epoch == int(n_epochs))
        )
        if run_val:
            val_metrics = evaluate_epoch_v6(
                model,
                eval_dataloader,
                device,
                operator_time_points=operator_time_points,
                max_steps=eval_max_steps,
                use_amp=use_amp,
                ema=ema,
                non_blocking=non_blocking,
                use_nibp_input=bool(use_nibp_input),
                require_nibp_input_valid=bool(require_nibp_input_valid),
            )
            val_dict = val_metrics.to_dict()
            for k in VAL_METRIC_KEYS:
                history[k].append(val_dict[k])
            print(f"       | {val_metrics.summary_line()}")

            current_score = val_dict.get(best_model_metric, float("nan"))
            best_metric_eligible = True
            if best_model_metric in {"val_tau_log_mae", "val_tau_mae"}:
                best_metric_eligible = lambda_tau_dia_eff > 0.0
            if best_metric_eligible and np.isfinite(current_score):
                improved = (
                    current_score < best_val_score
                    if best_model_metric in lower_is_better
                    else current_score > best_val_score
                )
                if improved:
                    best_val_score = current_score
                    best_epoch = epoch
                    best_model_state = {
                        k: v.detach().cpu().clone()
                        for k, v in v9c.export_model_state_dict(model).items()
                    }
                    best_ema_state = (
                        {k: v.detach().cpu().clone() for k, v in v9c.export_ema_state_dict(ema).items()}
                        if ema is not None
                        else None
                    )
                    print(f"       | ** new best {best_model_metric}={current_score:.4f} at epoch {epoch} **")
        else:
            for k in VAL_METRIC_KEYS:
                history[k].append(float("nan"))

        if epoch_end_callback is not None:
            epoch_end_callback(
                epoch=epoch,
                model=model,
                ema=ema,
                history=history,
                metrics=run,
                wk_epoch_summary=wk_epoch_summary,
            )

    history["best_epoch"] = best_epoch
    history["best_val_score"] = best_val_score
    history["best_model_metric"] = best_model_metric
    history["best_model_state"] = best_model_state
    history["best_ema_state"] = best_ema_state
    if best_epoch > 0:
        print(f"\n[training_v9_tau] Best {best_model_metric}={best_val_score:.4f} at epoch {best_epoch}/{n_epochs}")

    return model, ema, history


# =============================================================
# v6 validation and visualization
# =============================================================


@torch.no_grad()
def evaluate_epoch_v6(
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    *,
    operator_time_points: int | None = None,
    max_steps: int | None = None,
    use_amp: bool = False,
    ema: v9c.ModelEMA | None = None,
    tau_soft_extrema: float = 2.0,
    non_blocking: bool = False,
    use_nibp_input: bool = True,
    require_nibp_input_valid: bool = False,
) -> v9c.ValidationMetrics:
    was_training = model.training
    if ema is not None:
        ema.apply(model)

    model.eval()
    use_cuda_amp = use_amp and device.type == "cuda"
    amp_dtype = v9c.resolve_amp_dtype(device) if use_cuda_amp else torch.float32

    try:
        sum_ae = 0.0
        sum_se = 0.0
        sum_corr = 0.0
        sum_sbp_ae = 0.0
        sum_dbp_ae = 0.0
        sum_map_ae = 0.0
        sum_nibp_sbp_ae = 0.0
        sum_nibp_dbp_ae = 0.0
        sum_nibp_map_ae = 0.0
        sum_co_ae = 0.0
        sum_cvp_ae = 0.0
        sum_tau_log_ae = 0.0
        sum_tau_ae = 0.0
        sum_loss = 0.0  # sum of eligible per-window reconstruction MSE values
        n_valid_points = 0
        n_windows = 0
        n_stat_windows = 0
        n_nibp = 0
        n_co = 0
        n_cvp = 0
        n_tau = 0

        eval_iter = islice(eval_dataloader, max_steps) if max_steps else eval_dataloader
        for batch in eval_iter:
            x = batch["input"].to(device, non_blocking=non_blocking)
            y = batch["output"].to(device, non_blocking=non_blocking)
            mask = batch["output_mask"].to(device, non_blocking=non_blocking)
            dt_in = batch["dt"]

            B, source_T = x.shape[0], x.shape[1]
            raw_dt_b = v9c._broadcast_dt(dt_in, B, source_T, x.device, x.dtype)
            raw_dt_b = v9c._calibrate_dt(raw_dt_b)

            target_T = int(operator_time_points) if operator_time_points is not None else source_T
            x, y, mask, dt_b = v9c._resample_to_operator_grid(x, y, mask, raw_dt_b, target_T)
            nibp_features = (
                _nibp_features_from_batch(
                    batch,
                    device=x.device,
                    dtype=x.dtype,
                    require_all_valid=bool(require_nibp_input_valid),
                )
                if use_nibp_input
                else None
            )

            with torch.amp.autocast("cuda", enabled=use_cuda_amp, dtype=amp_dtype):
                y_hat, fields, aux = model(x, dt_seconds=dt_b, nibp_features=nibp_features)

            y_hat_bt = v9c._to_bt2d(y_hat)
            y_bt = v9c._to_bt2d(y)
            y_hat_bt, y_bt, mask = v9c._align_bt(y_hat_bt, y_bt, mask)
            mask_bool = mask.bool()

            for b in range(B):
                m = mask_bool[b]
                n_valid = int(m.sum().item())
                if n_valid < 6:
                    continue

                pred = y_hat_bt[b][m]
                true = y_bt[b][m]
                ae = (pred - true).abs()
                se = (pred - true).pow(2)
                se_sum = float(se.sum().item())
                sum_ae += float(ae.sum().item())
                sum_se += se_sum
                sum_loss += se_sum / n_valid
                n_valid_points += n_valid

                pm = pred - pred.mean()
                tm = true - true.mean()
                numer = (pm * tm).sum()
                denom = torch.sqrt(pm.pow(2).sum() * tm.pow(2).sum() + 1e-12)
                sum_corr += float((numer / denom).clamp(-1, 1).item())

                n_windows += 1

            valid_count = mask_bool.float().sum(dim=1)
            valid_window = valid_count >= 6.0
            pred_sbp_batch = v9c._masked_soft_extrema(y_hat_bt, mask_bool, tau=tau_soft_extrema, sign=+1.0)
            true_sbp_batch = v9c._masked_soft_extrema(y_bt, mask_bool, tau=tau_soft_extrema, sign=+1.0)
            pred_dbp_batch = v9c._masked_soft_extrema(y_hat_bt, mask_bool, tau=tau_soft_extrema, sign=-1.0)
            true_dbp_batch = v9c._masked_soft_extrema(y_bt, mask_bool, tau=tau_soft_extrema, sign=-1.0)
            w_batch = mask_bool.float()
            denom_batch = w_batch.sum(dim=1).clamp_min(1.0)
            pred_map_batch = (y_hat_bt * w_batch).sum(dim=1) / denom_batch
            true_map_batch = (y_bt * w_batch).sum(dim=1) / denom_batch
            n_batch_stat = int(valid_window.sum().item())
            if n_batch_stat > 0:
                sum_sbp_ae += float((pred_sbp_batch[valid_window] - true_sbp_batch[valid_window]).abs().sum().item())
                sum_dbp_ae += float((pred_dbp_batch[valid_window] - true_dbp_batch[valid_window]).abs().sum().item())
                sum_map_ae += float((pred_map_batch[valid_window] - true_map_batch[valid_window]).abs().sum().item())
                n_stat_windows += n_batch_stat
            if all(key in batch for key in ("nibp_sbp_label", "nibp_dbp_label", "nibp_map_label", "nibp_valid")):
                nibp_sbp = batch["nibp_sbp_label"].to(device=device, dtype=y_hat_bt.dtype).view(-1)
                nibp_dbp = batch["nibp_dbp_label"].to(device=device, dtype=y_hat_bt.dtype).view(-1)
                nibp_map = batch["nibp_map_label"].to(device=device, dtype=y_hat_bt.dtype).view(-1)
                nibp_valid = batch["nibp_valid"].to(device=device).view(-1).bool()
                valid_nibp = (
                    nibp_valid
                    & torch.isfinite(nibp_sbp)
                    & torch.isfinite(nibp_dbp)
                    & torch.isfinite(nibp_map)
                    & valid_window
                )
                n_batch_nibp = int(valid_nibp.sum().item())
                if n_batch_nibp > 0:
                    sum_nibp_sbp_ae += float((pred_sbp_batch[valid_nibp] - nibp_sbp[valid_nibp]).abs().sum().item())
                    sum_nibp_dbp_ae += float((pred_dbp_batch[valid_nibp] - nibp_dbp[valid_nibp]).abs().sum().item())
                    sum_nibp_map_ae += float((pred_map_batch[valid_nibp] - nibp_map[valid_nibp]).abs().sum().item())
                    n_nibp += n_batch_nibp

            if all(key in batch for key in ("co_ref_l_min", "co_valid")):
                co_ref = batch["co_ref_l_min"].to(device=device, dtype=y_hat_bt.dtype).view(-1)
                co_valid = batch["co_valid"].to(device=device).view(-1).bool()
                co_pred = aux["CO_head_L_min"].view(-1)
                valid_co = co_valid & torch.isfinite(co_ref) & torch.isfinite(co_pred)
                n_batch_co = int(valid_co.sum().item())
                if n_batch_co > 0:
                    sum_co_ae += float((co_pred[valid_co] - co_ref[valid_co]).abs().sum().item())
                    n_co += n_batch_co

            if all(key in batch for key in ("cvp_label", "cvp_valid")):
                cvp_ref = batch["cvp_label"].to(device=device, dtype=y_hat_bt.dtype).view(-1)
                cvp_valid = batch["cvp_valid"].to(device=device).view(-1).bool()
                cvp_pred = aux["wk"]["Pv"].view(-1)
                valid_cvp = cvp_valid & torch.isfinite(cvp_ref) & torch.isfinite(cvp_pred)
                n_batch_cvp = int(valid_cvp.sum().item())
                if n_batch_cvp > 0:
                    sum_cvp_ae += float((cvp_pred[valid_cvp] - cvp_ref[valid_cvp]).abs().sum().item())
                    n_cvp += n_batch_cvp

            if all(key in batch for key in ("tau_obs_log", "tau_valid")):
                tau_obs_log = batch["tau_obs_log"].to(device=device, dtype=y_hat_bt.dtype).view(-1)
                tau_valid = batch["tau_valid"].to(device=device).view(-1).bool()
                log_tau_pred = torch.log(aux["wk"]["tau"].view(-1).clamp_min(1e-4))
                valid_tau = tau_valid & torch.isfinite(tau_obs_log) & torch.isfinite(log_tau_pred)
                n_batch_tau = int(valid_tau.sum().item())
                if n_batch_tau > 0:
                    tau_obs = torch.exp(tau_obs_log[valid_tau])
                    tau_pred = torch.exp(log_tau_pred[valid_tau])
                    sum_tau_log_ae += float((log_tau_pred[valid_tau] - tau_obs_log[valid_tau]).abs().sum().item())
                    sum_tau_ae += float((tau_pred - tau_obs).abs().sum().item())
                    n_tau += n_batch_tau

    finally:
        if ema is not None:
            ema.restore(model)
        model.train(was_training)

    if n_windows == 0 or n_valid_points == 0:
        return v9c.ValidationMetrics()

    mae = sum_ae / n_valid_points
    rmse = math.sqrt(sum_se / n_valid_points)
    mean_corr = sum_corr / n_windows
    val_loss = sum_loss / n_windows
    sbp_mae = sum_sbp_ae / n_stat_windows if n_stat_windows > 0 else float("nan")
    dbp_mae = sum_dbp_ae / n_stat_windows if n_stat_windows > 0 else float("nan")
    map_mae = sum_map_ae / n_stat_windows if n_stat_windows > 0 else float("nan")
    nibp_sbp_mae = sum_nibp_sbp_ae / n_nibp if n_nibp > 0 else float("nan")
    nibp_dbp_mae = sum_nibp_dbp_ae / n_nibp if n_nibp > 0 else float("nan")
    nibp_map_mae = sum_nibp_map_ae / n_nibp if n_nibp > 0 else float("nan")
    co_mae = sum_co_ae / n_co if n_co > 0 else float("nan")
    cvp_mae = sum_cvp_ae / n_cvp if n_cvp > 0 else float("nan")
    tau_log_mae = sum_tau_log_ae / n_tau if n_tau > 0 else float("nan")
    tau_mae = sum_tau_ae / n_tau if n_tau > 0 else float("nan")

    return v9c.ValidationMetrics(
        mae_mmHg=mae,
        rmse_mmHg=rmse,
        corr=mean_corr,
        sbp_mae_mmHg=sbp_mae,
        dbp_mae_mmHg=dbp_mae,
        map_mae_mmHg=map_mae,
        nibp_sbp_mae_mmHg=nibp_sbp_mae,
        nibp_dbp_mae_mmHg=nibp_dbp_mae,
        nibp_map_mae_mmHg=nibp_map_mae,
        co_mae_l_min=co_mae,
        cvp_mae_mmHg=cvp_mae,
        tau_log_mae=tau_log_mae,
        tau_mae_sec=tau_mae,
        nibp_n=n_nibp,
        co_n=n_co,
        cvp_n=n_cvp,
        tau_n=n_tau,
        n_windows=n_windows,
        val_loss=val_loss,
    )


@torch.no_grad()
def visualize_results_v6(
    model,
    history,
    batch,
    device="cpu",
    outdir="figs_v6",
    ema=None,
    operator_time_points=None,
    use_nibp_input: bool = True,
    require_nibp_input_valid: bool = False,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    if ema is not None:
        ema.apply(model)
    was_training = model.training
    model.eval()

    try:
        x = batch["input"].to(device)
        y = batch["output"].to(device)
        mask = batch["output_mask"].to(device)
        dt = batch["dt"]

        B, T = x.size(0), x.size(1)
        raw_dt_b = v9c._broadcast_dt(dt, B, T, x.device, x.dtype)
        raw_dt_b = v9c._calibrate_dt(raw_dt_b)
        if operator_time_points is not None:
            x, y, mask, dt_b = v9c._resample_to_operator_grid(x, y, mask, raw_dt_b, int(operator_time_points))
            v9c._assert_operator_grid_dt_consistency(raw_dt_b, dt_b, T, int(operator_time_points))
        else:
            dt_b = raw_dt_b

        nibp_features = (
            _nibp_features_from_batch(
                batch,
                device=x.device,
                dtype=x.dtype,
                require_all_valid=bool(require_nibp_input_valid),
            )
            if use_nibp_input
            else None
        )
        y_hat, fields, aux = model(x, dt_seconds=dt_b, nibp_features=nibp_features)

        dt_bt = dt_b.squeeze(-1)
        tb = torch.cumsum(dt_bt[0], dim=0).detach().cpu().numpy()
        tb = tb - tb[0]
        y_true = v9c._to_bt2d(y)[0, :].detach().cpu().numpy()
        y_pred = y_hat[0, :, 0].detach().cpu().numpy()
        y_direct = fields["abp_direct"][0, :].detach().cpu().numpy()
        y_phys = fields["abp_physics"][0, :].detach().cpu().numpy()

        fig1, ax = plt.subplots(figsize=(11, 4))
        ax.plot(tb, y_true, label="True ABP")
        ax.plot(tb, y_pred, label="Pred ABP", alpha=0.9)
        ax.plot(tb, y_direct, "--", label="Direct", alpha=0.7)
        ax.plot(tb, y_phys, ":", label="Physics", alpha=0.7)
        ax.set_title("ABP: true vs prediction")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (mmHg)")
        ax.legend()
        plt.tight_layout()
        fig1.savefig(os.path.join(outdir, "abp_comparison.png"), dpi=300)

        fig2, ax = plt.subplots(figsize=(11, 3))
        ax.plot(tb, aux["U_in"][0, :].detach().cpu().numpy(), label="U(x=0)")
        ax.plot(tb, aux["U_in_bc"][0, :].detach().cpu().numpy(), "--", label="BC head", alpha=0.8)
        ax.set_title("Inlet Normalized Drive")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("U = Q/C (mmHg/s)")
        ax.legend()
        plt.tight_layout()
        fig2.savefig(os.path.join(outdir, "normalized_drive.png"), dpi=300)

        ep = np.arange(1, len(history["loss"]) + 1)
        fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(ep, history["loss"], label="Total")
        axes[0, 0].plot(ep, history["wave"], label="Wave")
        axes[0, 0].plot(ep, history["distill"], label="Distill")
        axes[0, 0].plot(ep, history["phys"], label="Phys")
        axes[0, 0].plot(ep, history["param"], label="Param")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].legend()
        axes[0, 1].plot(ep, history["direct_alpha"], label="Direct alpha")
        axes[0, 1].set_title("Blend Curriculum")
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[1, 0].plot(ep, history["tau"], label="tau")
        axes[1, 0].plot(ep, history["R2"], label="R2 gauge (=tau)")
        axes[1, 0].set_title("Normalized WK Coordinates")
        axes[1, 0].legend()
        axes[1, 1].plot(ep, history["lr"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_yscale("log")
        plt.tight_layout()
        fig3.savefig(os.path.join(outdir, "training_curves.png"), dpi=300)
        plt.close("all")
    finally:
        if ema is not None:
            ema.restore(model)
        model.train(was_training)


TAU_CHECKPOINT_ARCHITECTURE = "training_v9_tau_slow_wk_routed"
TAU_CHECKPOINT_VARIANT = "training_v9_tauA_slow_wk_routed_morphology_external"


def _tau_checkpoint_metadata(
    *,
    route_slow_wk_rollout: bool,
    lambda_pv_anchor: float,
    pv_anchor_mmHg: float,
    pv_anchor_sigma_mmHg: float,
    lambda_qscale_gauge: float,
    lambda_tau_morph: float,
    lambda_tau_morph_internal: float,
    tau_morph_start_epoch: int,
    tau_morph_full_epoch: int,
    tau_morph_beta: float,
    tau_morph_internal_detach_ref: bool,
) -> dict:
    return {
        "architecture": TAU_CHECKPOINT_ARCHITECTURE,
        "training_variant": TAU_CHECKPOINT_VARIANT,
        "slow_state_controls": {
            "route_slow_wk_rollout": bool(route_slow_wk_rollout),
            "lambda_pv_anchor": float(lambda_pv_anchor),
            "pv_anchor_mmHg": float(pv_anchor_mmHg),
            "pv_anchor_sigma_mmHg": float(pv_anchor_sigma_mmHg),
            "lambda_qscale_gauge": float(lambda_qscale_gauge),
        },
        "tau_morphology_loss": {
            "A_blend_to_tau_obs_log": float(lambda_tau_morph),
            "B_physics_to_wk_tau_log": float(lambda_tau_morph_internal),
            "start_epoch": int(tau_morph_start_epoch),
            "full_epoch": int(tau_morph_full_epoch),
            "smooth_l1_beta": float(tau_morph_beta),
            "internal_detach_ref": bool(tau_morph_internal_detach_ref),
        },
    }


# =============================================================
# Entry point
# =============================================================


if __name__ == "__main__":
    task = os.getenv("PINN_TASK", "").strip()
    if task:
        raise RuntimeError(
            f"PINN_TASK={task!r} is set, but training_v9_tauA.py only runs training. "
            "Unset PINN_TASK before launch."
        )

    os.environ.setdefault("PINN_NUM_WORKERS", "0")
    os.environ.setdefault("PINN_PREFETCH_FACTOR", "1")

    TRAIN_MODE = v9c.resolve_train_mode()
    DEVICE = v9c.get_device()
    print("Using device:", DEVICE)
    print(f"[training_v9_tau] Training mode: {TRAIN_MODE}")

    DL_CFG = v9c.resolve_dataloader_config(DEVICE)
    NUM_WORKERS = DL_CFG["num_workers"]
    PREFETCH_FACTOR = DL_CFG["prefetch_factor"]
    PIN_MEMORY = DL_CFG["pin_memory"]
    NON_BLOCKING = DL_CFG["non_blocking"]
    PERSISTENT_WORKERS = DL_CFG["persistent_workers"] and NUM_WORKERS > 0
    BUFFER_SIZE = DL_CFG["shuffle_buffer"]
    EVAL_BUFFER_SIZE = DL_CFG["eval_buffer"]
    print(f"[training_v9_tau] Unified memory: {DL_CFG['unified_memory']}")

    dt_units = os.getenv("PINN_DT_UNITS", "").strip().lower()
    if dt_units not in {"seconds", "second", "sec", "s", "hz", "frequency", "freq", "sample_rate", "sampling_rate"}:
        raise RuntimeError(
            "training_v9_tau requires PINN_DT_UNITS=seconds or PINN_DT_UNITS=hz before launch."
    )
    print(f"[training_v9_tau] PINN_DT_UNITS={dt_units}")
    SKIP_FINAL_VIS = v9c._env_bool("PINN_SKIP_FINAL_VIS", True)
    print(f"[training_v9_tau] PINN_SKIP_FINAL_VIS={int(SKIP_FINAL_VIS)}")

    N_EPOCHS = v9c._env_int("PINN_EPOCHS", 100)
    MAX_STEPS_PER_EPOCH = v9c._env_int("PINN_MAX_STEPS", 9000)
    WARMUP_STEPS = v9c._env_int("PINN_WARMUP_STEPS", 3000)
    PHASE_PRETRAIN_EPOCHS = v9c._env_int("PINN_PHASE_PRETRAIN_EPOCHS", 8)
    WAVEFORM_END_EPOCH = v9c._env_int("PINN_WAVEFORM_END_EPOCH", 40)
    PHYSICS_END_EPOCH = v9c._env_int("PINN_PHYSICS_END_EPOCH", 80)
    BATCH_SIZE = v9c._env_int("PINN_BATCH_SIZE", 16)
    EVAL_MAX_STEPS = v9c._env_int("PINN_EVAL_MAX_STEPS", 500)
    EVAL_EVERY_N_EPOCHS = v9c._env_int("PINN_EVAL_EVERY_N_EPOCHS", 1)
    ARROW_USE_THREADS = v9c._env_bool("PINN_ARROW_USE_THREADS", False)
    USE_AMP = v9c._env_bool("PINN_USE_AMP", True)
    ema_decay = v9c._env_float("PINN_EMA_DECAY", 0.995)
    seeds = v9c._env_int_list("PINN_SEEDS", [0])
    MILESTONE_EPOCHS = sorted(
        {
            ep
            for ep in v9c._env_int_list(
                "PINN_MILESTONE_EPOCHS",
                [1, 8, 20, 40, 55, 80, 100],
            )
            if ep > 0
        }
    )

    lambda_phys = v9c._env_float("PINN_LAMBDA_PHYS", 0.20)
    lambda_param = v9c._env_float("PINN_LAMBDA_PARAM", 0.05)
    lambda_distill = v9c._env_float("PINN_LAMBDA_DISTILL", 2.0)
    lambda_co_env = v9c._env_float("PINN_LAMBDA_CO", 0.0)
    lambda_co_supervised_env = v9c._env_float("PINN_LAMBDA_CO_SUPERVISED", 0.0)
    lambda_nibp = v9c._env_float("PINN_LAMBDA_NIBP", 0.0)
    lambda_cvp_env = v9c._env_float("PINN_LAMBDA_CVP", 0.0)
    if lambda_co_env != 0.0 or lambda_co_supervised_env != 0.0 or lambda_cvp_env != 0.0:
        raise RuntimeError(
            "training_v9_tauA.py does not train CO/CVP losses. "
            "Unset PINN_LAMBDA_CO, PINN_LAMBDA_CO_SUPERVISED, and PINN_LAMBDA_CVP or set them to 0."
        )
    lambda_co = 0.0
    lambda_co_supervised = 0.0
    lambda_cvp = 0.0
    lambda_tau_dia = v9c._env_float("PINN_LAMBDA_TAU_DIA", 0.25)
    if lambda_tau_dia <= 0.0:
        raise RuntimeError("training_v9_tauA.py requires PINN_LAMBDA_TAU_DIA > 0.")
    lambda_tau_morph = v9c._env_float("PINN_LAMBDA_TAU_MORPH", 0.50)
    if lambda_tau_morph <= 0.0:
        raise RuntimeError("training_v9_tauA.py requires PINN_LAMBDA_TAU_MORPH > 0.")
    lambda_tau_morph_internal = v9c._env_float("PINN_LAMBDA_TAU_MORPH_INTERNAL", 0.0)
    if lambda_tau_morph_internal != 0.0:
        raise RuntimeError("training_v9_tauA.py supports A only; set PINN_LAMBDA_TAU_MORPH_INTERNAL=0.")
    tau_morph_beta = v9c._env_float("PINN_TAU_MORPH_BETA", 0.20)
    if tau_morph_beta <= 0.0:
        raise RuntimeError("PINN_TAU_MORPH_BETA must be positive.")
    tau_morph_internal_detach_ref = v9c._env_bool("PINN_TAU_MORPH_INTERNAL_DETACH_REF", True)
    EMIT_TAU_TARGETS = v9c._env_bool("PINN_EMIT_TAU_TARGETS", lambda_tau_dia > 0.0)
    if not EMIT_TAU_TARGETS:
        raise RuntimeError("training_v9_tauA.py requires tau targets; set PINN_EMIT_TAU_TARGETS=1.")

    pde_collocation_time = v9c._env_int("PINN_PDE_COLLOCATION_TIME", 24)
    pde_collocation_x = v9c._env_int("PINN_PDE_COLLOCATION_X", 6)
    group_ramp_epochs = v9c._env_int("PINN_GROUP_RAMP_EPOCHS", 15)
    wk_delta_initial_scale = v9c._env_float("PINN_WK_DELTA_INITIAL_SCALE", 0.02)
    wk_delta_full_epoch = v9c._env_int("PINN_WK_DELTA_FULL_EPOCH", PHYSICS_END_EPOCH)
    physics_warm_end_epoch = v9c._env_int("PINN_PHYSICS_WARM_END_EPOCH", 55)
    direct_alpha_physics_start = v9c._env_float("PINN_DIRECT_ALPHA_PHYSICS_START", 0.95)
    direct_alpha_physics_warm_end = v9c._env_float("PINN_DIRECT_ALPHA_PHYSICS_WARM_END", 0.80)
    direct_alpha_physics_end = v9c._env_float("PINN_DIRECT_ALPHA_PHYSICS_END", 0.60)
    direct_alpha_joint = v9c._env_float("PINN_DIRECT_ALPHA_JOINT", 0.45)
    direct_alpha_joint_ramp_epochs = v9c._env_int("PINN_DIRECT_ALPHA_JOINT_RAMP_EPOCHS", 20)
    use_nibp_input = v9c._env_bool("PINN_USE_NIBP_INPUT", True)
    require_nibp_input_valid = v9c._env_bool("PINN_REQUIRE_NIBP_INPUT_VALID", False)
    nibp_full_epoch = v9c._env_int("PINN_NIBP_FULL_EPOCH", WAVEFORM_END_EPOCH)
    co_supervised_start_epoch = v9c._env_int("PINN_CO_SUPERVISED_START_EPOCH", WAVEFORM_END_EPOCH + 1)
    co_supervised_full_epoch = v9c._env_int("PINN_CO_SUPERVISED_FULL_EPOCH", PHYSICS_END_EPOCH)
    cvp_full_epoch = v9c._env_int("PINN_CVP_FULL_EPOCH", PHYSICS_END_EPOCH)
    tau_dia_start_epoch = v9c._env_int("PINN_TAU_DIA_START_EPOCH", WAVEFORM_END_EPOCH)
    tau_dia_full_epoch = v9c._env_int("PINN_TAU_DIA_FULL_EPOCH", physics_warm_end_epoch)
    tau_morph_start_epoch = v9c._env_int("PINN_TAU_MORPH_START_EPOCH", tau_dia_start_epoch)
    tau_morph_full_epoch = v9c._env_int("PINN_TAU_MORPH_FULL_EPOCH", tau_dia_full_epoch)
    tau_morph_segments_path = os.getenv("PINN_TAU_MORPH_SEGMENTS", "").strip()
    tau_morph_segments_by_key = _load_tau_morphology_segments(tau_morph_segments_path)

    w_abp_stats = v9c._env_float("PINN_W_ABP_STATS", 6.5)
    w_pressure_aux = v9c._env_float("PINN_W_PRESSURE_AUX", 0.25)
    w_prior = v9c._env_float("PINN_W_PRIOR", 0.75)
    route_slow_wk_rollout = v9c._env_bool("PINN_ROUTE_SLOW_WK_ROLLOUT", True)
    lambda_pv_anchor = v9c._env_float("PINN_LAMBDA_PV_ANCHOR", 0.10)
    pv_anchor_mmHg = v9c._env_float("PINN_PV_ANCHOR_MMHG", 5.0)
    pv_anchor_sigma_mmHg = v9c._env_float("PINN_PV_ANCHOR_SIGMA", 3.0)
    lambda_qscale_gauge = v9c._env_float("PINN_LAMBDA_QSCALE_GAUGE", 0.10)
    if lambda_pv_anchor < 0.0:
        raise RuntimeError("PINN_LAMBDA_PV_ANCHOR must be non-negative.")
    if lambda_qscale_gauge < 0.0:
        raise RuntimeError("PINN_LAMBDA_QSCALE_GAUGE must be non-negative.")
    if pv_anchor_sigma_mmHg <= 0.0:
        raise RuntimeError("PINN_PV_ANCHOR_SIGMA must be positive.")
    encoder_temporal_stride = v9c._env_int("PINN_ENCODER_TEMPORAL_STRIDE", 1)
    CHUNK_DURATION = v9c._env_float("PINN_CHUNK_DURATION", 10.0)
    GRID_POINTS = v9c._env_int("PINN_GRID_POINTS", 250)
    eval_split = os.getenv("PINN_EVAL_SPLIT", "validation").strip().lower() or "validation"

    if NUM_WORKERS < 0:
        raise ValueError("PINN_NUM_WORKERS must be non-negative.")
    if NUM_WORKERS > 0 and PREFETCH_FACTOR <= 0:
        raise ValueError("PINN_PREFETCH_FACTOR must be positive when workers are enabled.")
    if pde_collocation_time < 5:
        raise ValueError("PINN_PDE_COLLOCATION_TIME must be at least 5.")
    if pde_collocation_x < 3:
        raise ValueError("PINN_PDE_COLLOCATION_X must be at least 3.")

    train_meta_csv = v9c._resolve_meta_csv("training")
    eval_meta_csv = v9c._resolve_meta_csv(eval_split)
    train_manifest_rows = len(pd.read_csv(train_meta_csv))
    eval_manifest_rows = len(pd.read_csv(eval_meta_csv))
    print(f"[training_v9_tau] Data root: {v9c._data_root()}")
    print(f"[training_v9_tau] Training metadata actually used: {train_meta_csv}")
    print(f"[training_v9_tau] Training rows: {train_manifest_rows}")
    print(f"[training_v9_tau] Eval split: {eval_split}")
    print(f"[training_v9_tau] Eval metadata actually used: {eval_meta_csv}")
    print(f"[training_v9_tau] Eval rows: {eval_manifest_rows}")
    print(
        "[training_v9_tau] DataLoader config: "
        f"batch_size={BATCH_SIZE}, workers={NUM_WORKERS}, "
        f"persistent_workers={PERSISTENT_WORKERS and NUM_WORKERS > 0}, "
        f"prefetch_factor={PREFETCH_FACTOR if NUM_WORKERS > 0 else 'n/a'}, "
        f"pin_memory={PIN_MEMORY and DEVICE.type == 'cuda'}, "
        f"shuffle_buffer={BUFFER_SIZE}, arrow_use_threads={ARROW_USE_THREADS}"
    )

    co_prior_map = {}
    print("[training_v9_tau] CO/CVP supervised losses disabled; use hemodynamic state for downstream CO/CVP.")

    artifact_tag = os.getenv("PINN_ARTIFACT_TAG", "training_v9_tauA").strip() or "training_v9_tauA"
    print(
        f"[training_v9_tau] budget: epochs={N_EPOCHS}, max_steps={MAX_STEPS_PER_EPOCH}, "
        f"warmup={WARMUP_STEPS}, phase_pretrain={PHASE_PRETRAIN_EPOCHS}, "
        f"chunk_duration={CHUNK_DURATION}s, M={GRID_POINTS}, "
        f"encoder_stride={encoder_temporal_stride}, seeds={seeds}"
    )
    print(
        f"[training_v9_tau] loss groups: phys={lambda_phys}, param={lambda_param}, "
        f"distill={lambda_distill}; pde_collocation=({pde_collocation_time}, {pde_collocation_x}); "
        f"group_ramp_epochs={group_ramp_epochs}"
    )
    print(
        f"[training_v9_tau] sparse aux: nibp={lambda_nibp} full_epoch={nibp_full_epoch}; "
            "co_sup=0.0 downstream_only; cvp=0.0 downstream_only; "
            f"tau_dia={lambda_tau_dia} start_epoch={tau_dia_start_epoch} full_epoch={tau_dia_full_epoch}"
    )
    print(
        "[training_v9_tauA] morphology loss: "
        f"A_blend={lambda_tau_morph} start_epoch={tau_morph_start_epoch} "
        f"full_epoch={tau_morph_full_epoch} beta={tau_morph_beta}; "
        f"B_internal={lambda_tau_morph_internal}; "
        f"segments={'online' if tau_morph_segments_by_key is None else tau_morph_segments_path}"
    )
    print(
        "[training_v9_tauA] slow-state controls: "
        f"route_slow_wk_rollout={int(route_slow_wk_rollout)} "
        f"lambda_pv_anchor={lambda_pv_anchor} "
        f"pv_anchor={pv_anchor_mmHg} sigma={pv_anchor_sigma_mmHg} "
        f"lambda_qscale_gauge={lambda_qscale_gauge}"
    )
    print(f"[training_v9_tau] emit tau targets: {EMIT_TAU_TARGETS}")
    print(f"[training_v9_tau] EMA decay: {ema_decay}")
    print(
        "[training_v9_tau] curriculum: "
        f"phase=1-{PHASE_PRETRAIN_EPOCHS}, "
        f"waveform={PHASE_PRETRAIN_EPOCHS + 1}-{WAVEFORM_END_EPOCH}, "
        f"physics_warm={WAVEFORM_END_EPOCH + 1}-{physics_warm_end_epoch}, "
        f"physics_ramp={physics_warm_end_epoch + 1}-{PHYSICS_END_EPOCH}, "
        f"joint={PHYSICS_END_EPOCH + 1}-{N_EPOCHS}; "
        f"alpha={direct_alpha_physics_start}->{direct_alpha_physics_warm_end}"
        f"->{direct_alpha_physics_end}->{direct_alpha_joint}"
    )
    print(
        f"[training_v9_tau] NIBP forward input: enabled={use_nibp_input}, "
        f"require_all_valid={require_nibp_input_valid}"
    )
    print(f"[training_v9_tau] milestone epochs: {MILESTONE_EPOCHS if MILESTONE_EPOCHS else 'none'}")

    all_hist = {}
    checkpoint_metadata = _tau_checkpoint_metadata(
        route_slow_wk_rollout=route_slow_wk_rollout,
        lambda_pv_anchor=lambda_pv_anchor,
        pv_anchor_mmHg=pv_anchor_mmHg,
        pv_anchor_sigma_mmHg=pv_anchor_sigma_mmHg,
        lambda_qscale_gauge=lambda_qscale_gauge,
        lambda_tau_morph=lambda_tau_morph,
        lambda_tau_morph_internal=lambda_tau_morph_internal,
        tau_morph_start_epoch=tau_morph_start_epoch,
        tau_morph_full_epoch=tau_morph_full_epoch,
        tau_morph_beta=tau_morph_beta,
        tau_morph_internal_detach_ref=tau_morph_internal_detach_ref,
    )

    for seed in seeds:
        print(f"\n==================== SEED {seed} ====================")
        v9c.seed_everything(seed, deterministic=False)

        g = torch.Generator()
        g.manual_seed(seed)

        loader_bundle = v9c.build_waveform_data_loaders(
            train_metadata_csv=train_meta_csv,
            eval_metadata_csv=eval_meta_csv,
            chunk_duration=CHUNK_DURATION,
            grid_points=GRID_POINTS,
            train_buffer_size=BUFFER_SIZE,
            eval_buffer_size=EVAL_BUFFER_SIZE,
            emit_tau_targets=EMIT_TAU_TARGETS,
            arrow_use_threads=ARROW_USE_THREADS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            generator=g,
        )
        ds = loader_bundle.train_dataset
        dl = loader_bundle.train_loader
        eval_ds = loader_bundle.eval_dataset
        eval_dl = loader_bundle.eval_loader
        reference_batch = loader_bundle.reference_batch

        try:
            loader_len = len(dl)
            print("len(dl) =", loader_len)
        except TypeError:
            loader_len = None
            print("len(dl) is not defined (IterableDataset)")

        train_steps_per_epoch = MAX_STEPS_PER_EPOCH
        if loader_len is not None:
            train_steps_per_epoch = min(MAX_STEPS_PER_EPOCH, max(1, loader_len))

        total_steps_budget = max(1, N_EPOCHS * train_steps_per_epoch)
        effective_warmup_steps = min(WARMUP_STEPS, total_steps_budget)
        print(f"[training_v9_tau] total_steps = {N_EPOCHS} x {train_steps_per_epoch} = {total_steps_budget}")

        milestone_epochs_for_run = [ep for ep in MILESTONE_EPOCHS if ep <= N_EPOCHS]

        def epoch_end_callback(*, epoch, model, ema, history, metrics, wk_epoch_summary):
            del metrics
            if epoch not in milestone_epochs_for_run:
                return
            os.makedirs("runs", exist_ok=True)
            reference_params = v9c._unwrap_compiled_model(model).get_windkessel_params()
            ckpt_path = f"runs/run_seed_{seed:02d}_{artifact_tag}_ep{epoch:03d}.pt"
            milestone_save = {
                "seed": seed,
                "epoch": epoch,
                "train_mode": TRAIN_MODE,
                "manifest_csv": train_meta_csv,
                "eval_manifest_csv": eval_meta_csv,
                "model_state": v9c.export_model_state_dict(model),
                "history": v9c.checkpoint_history(history),
                **checkpoint_metadata,
                "windkessel_reference_params": reference_params,
                "windkessel_reference_params_note": (
                    "Tau parameterization uses identifiable pressure dynamics; "
                    "C is a unit gauge, not physical arterial compliance."
                ),
                "windkessel_epoch_summary": wk_epoch_summary or {},
            }
            if ema is not None:
                milestone_save["ema_state"] = v9c.export_ema_state_dict(ema)
            torch.save(milestone_save, ckpt_path)
            print(f"[training_v9_tau] Saved milestone checkpoint: {ckpt_path}")

        model, ema, hist = train_pinn_deeponet_v6_tau(
            dl,
            co_prior_map=co_prior_map,
            n_epochs=N_EPOCHS,
            lr=1e-4,
            device=DEVICE,
            nx=16,
            encoder_temporal_stride=encoder_temporal_stride,
            operator_time_points=GRID_POINTS,
            grad_clip=1.0,
            phase_pretrain_epochs=PHASE_PRETRAIN_EPOCHS,
            waveform_end_epoch=WAVEFORM_END_EPOCH,
            physics_end_epoch=PHYSICS_END_EPOCH,
            use_amp=USE_AMP,
            max_steps_per_epoch=train_steps_per_epoch,
            warmup_steps=effective_warmup_steps,
            min_lr_ratio=0.01,
            ema_decay=ema_decay,
            lambda_phys=lambda_phys,
            lambda_param=lambda_param,
            lambda_distill=lambda_distill,
            lambda_co=lambda_co,
            lambda_co_supervised=lambda_co_supervised,
            lambda_nibp=lambda_nibp,
            lambda_cvp=lambda_cvp,
            lambda_tau_dia=lambda_tau_dia,
            lambda_tau_morph=lambda_tau_morph,
            lambda_tau_morph_internal=lambda_tau_morph_internal,
            tau_morph_start_epoch=tau_morph_start_epoch,
            tau_morph_full_epoch=tau_morph_full_epoch,
            tau_morph_beta=tau_morph_beta,
            tau_morph_internal_detach_ref=tau_morph_internal_detach_ref,
            w_beat=0.5,
            w_abp_stats=w_abp_stats,
            w_wk=0.1,
            w_rtot=0.5,
            w_pc_periodic=0.1,
            w_flow_smooth=0.12,
            w_latent_smooth=0.05,
            w_pressure_aux=w_pressure_aux,
            w_subject_param_consistency=0.05,
            w_pde_max=0.05,
            pde_collocation_time=pde_collocation_time,
            pde_collocation_x=pde_collocation_x,
            group_ramp_epochs=group_ramp_epochs,
            w_prior=w_prior,
            wk_delta_initial_scale=wk_delta_initial_scale,
            wk_delta_full_epoch=wk_delta_full_epoch,
            physics_warm_end_epoch=physics_warm_end_epoch,
            direct_alpha_physics_start=direct_alpha_physics_start,
            direct_alpha_physics_warm_end=direct_alpha_physics_warm_end,
            direct_alpha_physics_end=direct_alpha_physics_end,
            direct_alpha_joint=direct_alpha_joint,
            direct_alpha_joint_ramp_epochs=direct_alpha_joint_ramp_epochs,
            route_slow_wk_rollout=route_slow_wk_rollout,
            lambda_pv_anchor=lambda_pv_anchor,
            pv_anchor_mmHg=pv_anchor_mmHg,
            pv_anchor_sigma_mmHg=pv_anchor_sigma_mmHg,
            lambda_qscale_gauge=lambda_qscale_gauge,
            use_nibp_input=use_nibp_input,
            require_nibp_input_valid=require_nibp_input_valid,
            nibp_full_epoch=nibp_full_epoch,
            co_supervised_start_epoch=co_supervised_start_epoch,
            co_supervised_full_epoch=co_supervised_full_epoch,
            cvp_full_epoch=cvp_full_epoch,
            tau_dia_start_epoch=tau_dia_start_epoch,
            tau_dia_full_epoch=tau_dia_full_epoch,
            epoch_end_callback=epoch_end_callback,
            eval_dataloader=eval_dl,
            eval_max_steps=EVAL_MAX_STEPS,
            eval_every_n_epochs=EVAL_EVERY_N_EPOCHS,
            non_blocking=NON_BLOCKING,
            tau_morph_segments_by_key=tau_morph_segments_by_key,
        )

        all_hist[seed] = hist

        os.makedirs("runs", exist_ok=True)
        reference_params = v9c._unwrap_compiled_model(model).get_windkessel_params()
        wk_epoch_summary = hist["wk_epoch_summary"][-1] if hist.get("wk_epoch_summary") else {}
        save_dict = {
            "seed": seed,
            "train_mode": TRAIN_MODE,
            "manifest_csv": train_meta_csv,
            "eval_manifest_csv": eval_meta_csv,
            "model_state": v9c.export_model_state_dict(model),
            "history": v9c.checkpoint_history(hist),
            **checkpoint_metadata,
            "windkessel_reference_params": reference_params,
            "windkessel_reference_params_note": (
                "Tau parameterization uses identifiable pressure dynamics: "
                "C is fixed to unit gauge, R2 is tau in the normalized system, "
                "and R1 is rho1=R1_physical*C. Physical compliance must come "
                "from a downstream C_cal calibrator."
            ),
            "windkessel_epoch_summary": wk_epoch_summary,
        }
        if ema is not None:
            save_dict["ema_state"] = v9c.export_ema_state_dict(ema)
        run_path = f"runs/run_seed_{seed:02d}_{artifact_tag}.pt"
        torch.save(save_dict, run_path)
        print(f"[training_v9_tau] Saved final checkpoint: {run_path}")

        best_state = hist.get("best_model_state")
        if best_state is not None:
            best_save = {
                "seed": seed,
                "train_mode": TRAIN_MODE,
                "manifest_csv": train_meta_csv,
                "eval_manifest_csv": eval_meta_csv,
                "model_state": best_state,
                "best_epoch": hist["best_epoch"],
                "best_val_score": hist["best_val_score"],
                "best_model_metric": hist["best_model_metric"],
                "history": v9c.checkpoint_history(hist),
                **checkpoint_metadata,
                "windkessel_reference_params_note": (
                    "Tau parameterization uses identifiable pressure dynamics; "
                    "C is a unit gauge, not physical arterial compliance."
                ),
            }
            best_ema = hist.get("best_ema_state")
            if best_ema is not None:
                best_save["ema_state"] = best_ema
            best_path = f"runs/run_seed_{seed:02d}_{artifact_tag}_best.pt"
            torch.save(best_save, best_path)
            print(
                f"[training_v9_tau] Saved best checkpoint: {best_path} "
                f"(epoch {hist['best_epoch']}, "
                f"{hist['best_model_metric']}={hist['best_val_score']:.4f})"
            )

        final_fig_dir = f"figs_seed_{artifact_tag}_{seed:02d}"
        if SKIP_FINAL_VIS:
            print(f"[training_v9_tau] Skipping final visualization: {final_fig_dir}", flush=True)
        else:
            print(f"[training_v9_tau] Starting final visualization: {final_fig_dir}", flush=True)
            visualize_results_v6(
                model,
                hist,
                reference_batch,
                device=DEVICE,
                outdir=final_fig_dir,
                ema=ema,
                operator_time_points=GRID_POINTS,
                use_nibp_input=use_nibp_input,
                require_nibp_input_valid=require_nibp_input_valid,
            )
            print(f"[training_v9_tau] Finished final visualization: {final_fig_dir}", flush=True)

        del eval_dl, eval_ds, dl, ds
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
