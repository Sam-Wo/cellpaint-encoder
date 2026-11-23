"""Hand-written DINO training loop (DESIGN.md 3.3). No Lightning.

Per optimiser step ``it`` (global, indexes every schedule):

1. set lr / weight decay from the per-iteration schedules;
2. for each of ``accum_steps`` micro-batches ``m = it * accum + j``: augment on the device with a
   generator seeded by ``(seed, m)``, teacher forward on the 2 global crops, student forward on
   all crops, loss / accum, backward; teacher outputs are summed for the center;
3. gradient clipping (DINO per-parameter by default), cancel last-layer gradients during the first
   ``freeze_last_layer_epochs``, optimiser step;
4. EMA teacher update with ``momentum[it]`` and the DINO center update, **once per step**;
5. logging every ``log_every_steps``; checkpoints every ``ckpt_every_minutes`` /
   ``ckpt_every_steps``, at each epoch end and on SIGTERM.

Precision: ``bf16`` autocast with fp32 master weights and no GradScaler; ``fp16`` uses a
GradScaler; ``fp32`` disables autocast (CPU smoke runs).
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ..config import Config, config_hash, save_config, to_dict
from ..data.augment_gpu import MultiCropAugment
from ..data.sampler import BalancedBatchSampler
from ..data.shards import TileDataset, collate_tiles, load_index, load_stats
from ..models.dino_head import DINOHead, MultiCropWrapper
from ..models.dino_loss import DINOLoss, collapse_monitors
from ..models.vit import build_vit, load_imagenet_dino
from .checkpoint import StopFlag, find_resume, git_sha, load_checkpoint, save_checkpoint
from .logging import RunLogger
from .schedules import build_schedules
from .seed import get_rng_state, make_worker_init, micro_batch_seed, seed_everything, set_rng_state


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def default_run_dir(cfg: Config) -> Path:
    return Path(cfg.run.out_dir) / f"{cfg.run.name}-{config_hash(cfg)}"


def build_models(cfg: Config, device: torch.device):
    student = MultiCropWrapper(
        build_vit(cfg.model),
        DINOHead(
            cfg.model.embed_dim,
            cfg.dino.out_dim,
            cfg.dino.hidden_dim,
            cfg.dino.bottleneck_dim,
            norm_last_layer=cfg.dino.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(
        build_vit(cfg.model, drop_path=0.0),
        DINOHead(
            cfg.model.embed_dim,
            cfg.dino.out_dim,
            cfg.dino.hidden_dim,
            cfg.dino.bottleneck_dim,
            norm_last_layer=cfg.dino.norm_last_layer,
        ),
    )
    info = {}
    if cfg.model.init == "imagenet_dino":
        info = load_imagenet_dino(student.backbone, cfg.model.init_weights)
    elif cfg.model.init != "random":
        raise ValueError(f"unknown model.init {cfg.model.init!r}")
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student.to(device), teacher.to(device), info


def param_groups(model: torch.nn.Module):
    reg, noreg = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (noreg if n.endswith(".bias") or p.ndim == 1 else reg).append(p)
    return [{"params": reg}, {"params": noreg, "weight_decay": 0.0}]


def clip_gradients(params, clip: float, mode: str) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([g.detach().norm(2) for g in grads])
    total = norms.norm(2)
    if clip > 0:
        if mode == "per_param":  # DINO utils.clip_gradients: each parameter separately
            coefs = (clip / (norms + 1e-6)).clamp(max=1.0)
            for g, c in zip(grads, coefs):
                g.mul_(c)
        elif mode == "global":
            torch.nn.utils.clip_grad_norm_(params, clip)
        else:
            raise ValueError(f"unknown clip_mode {mode!r}")
    return float(total)


@torch.no_grad()
def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, m: float) -> None:
    """teacher = m * teacher + (1 - m) * student, parameters only."""
    tp = [p for p in teacher.parameters()]
    sp = [p.detach() for p in student.parameters()]
    torch._foreach_mul_(tp, m)
    torch._foreach_add_(tp, sp, alpha=1 - m)


def _channel_stats(cfg: Config) -> tuple[list[float], list[float]]:
    if cfg.data.channel_mean and cfg.data.channel_std:
        return list(cfg.data.channel_mean), list(cfg.data.channel_std)
    st = load_stats(cfg.data.shards_dir)
    return st["channel_mean"], st["channel_std"]


class Trainer:
    def __init__(self, cfg: Config, run_dir: str | Path | None = None, install_signals: bool = True):
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir else default_run_dir(cfg)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop = StopFlag(install=install_signals)
        r = cfg.run
        if r.threads:
            torch.set_num_threads(r.threads)
        self.device = resolve_device(r.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = r.cudnn_benchmark and not r.deterministic
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if r.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.index = load_index(cfg.data.shards_dir)
        o = cfg.optim
        if o.batch_size % o.accum_steps:
            raise ValueError("optim.batch_size must be divisible by optim.accum_steps")
        self.micro = o.batch_size // o.accum_steps
        epoch_size = r.epoch_size or len(self.index)
        self.iters_per_epoch = max(1, epoch_size // o.batch_size)
        self.total_steps = o.epochs * self.iters_per_epoch
        self.sched = build_schedules(cfg, self.iters_per_epoch)
        mean, std = _channel_stats(cfg)
        self.channel_mean, self.channel_std = mean, std
        d = cfg.dino
        self.aug = MultiCropAugment(
            d.aug, d.global_crop_size, d.local_crop_size, d.n_local_crops, d.scale_jitter, mean, std
        )
        self.n_crops = 2 + d.n_local_crops

        seed_everything(r.seed)
        self.student, self.teacher, self.init_info = build_models(cfg, self.device)
        self.loss_fn = DINOLoss(d.out_dim, self.n_crops, d.student_temp, d.center_momentum).to(self.device)
        self.opt = torch.optim.AdamW(param_groups(self.student), lr=0.0, betas=tuple(o.betas))
        self.use_amp = r.precision in ("bf16", "fp16")
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(r.precision)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=r.precision == "fp16")
        self.student_fwd = torch.compile(self.student) if cfg.model.compile else self.student
        self.teacher_fwd = torch.compile(self.teacher) if cfg.model.compile else self.teacher
        self.step = 0
        self.resumed_from = None

    # ----------------------------------------------------------------------------------- state
    def state_dict(self, complete: bool = True) -> dict:
        return {
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "center": self.loss_fn.center.detach().cpu().clone(),
            "step": self.step,
            "epoch": self.step // self.iters_per_epoch,
            "sampler": {"micro_batch": self.step * self.cfg.optim.accum_steps},
            "rng": get_rng_state(),
            "config": to_dict(self.cfg),
            "config_hash": config_hash(self.cfg),
            "git_sha": git_sha(Path(__file__).resolve().parents[3]),
            "channel_mean": self.channel_mean,
            "channel_std": self.channel_std,
            "complete": complete,
        }

    def load_state(self, st: dict) -> None:
        if st["config_hash"] != config_hash(self.cfg):
            raise ValueError(f"checkpoint config hash {st['config_hash']} != current {config_hash(self.cfg)}")
        self.student.load_state_dict(st["student"])
        self.teacher.load_state_dict(st["teacher"])
        self.opt.load_state_dict(st["optimizer"])
        self.scaler.load_state_dict(st["scaler"])
        self.loss_fn.center.copy_(st["center"].to(self.device))
        self.step = int(st["step"])
        # RNG states are restored only after the DataLoader iterator exists: creating it draws a
        # base seed from the global torch RNG, which the uninterrupted run did not do at this step.
        self._pending_rng = st["rng"]

    def maybe_resume(self, resume: str | None) -> None:
        if not resume:
            return
        if resume == "auto":
            p, st = find_resume(self.run_dir)
        else:
            p, st = Path(resume), load_checkpoint(Path(resume))
        if st is None:
            print(f"no checkpoint in {self.run_dir}: starting from scratch")
            return
        self.load_state(st)
        self.resumed_from = str(p)
        print(f"resumed from {p.name} at step {self.step}")

    def checkpoint(self, epoch_tag: int | None = None) -> Path:
        return save_checkpoint(self.state_dict(), self.run_dir, self.step, self.cfg.run.keep_last, epoch_tag)

    # ----------------------------------------------------------------------------------- data
    def loader(self):
        cfg = self.cfg
        sampler = BalancedBatchSampler(
            self.index,
            self.micro,
            cfg.run.seed,
            self.iters_per_epoch * cfg.optim.accum_steps,
            sources=list(cfg.data.train_sources) if cfg.data.balance == "source" else None,
            dmso_cap=cfg.data.dmso_cap,
            balance=cfg.data.balance,
            start=self.step * cfg.optim.accum_steps,
            stop=self.total_steps * cfg.optim.accum_steps,
        )
        ds = TileDataset(self.index, cfg.data.shards_dir, cfg.data.tile_cache_mb)
        nw = cfg.data.num_workers
        kw = dict(
            num_workers=nw,
            collate_fn=collate_tiles,
            pin_memory=cfg.data.pin_memory and self.device.type == "cuda",
        )
        if nw > 0:
            kw.update(
                persistent_workers=True,
                prefetch_factor=cfg.data.prefetch_factor,
                worker_init_fn=make_worker_init(cfg.run.seed, self.step // self.iters_per_epoch),
            )
        return torch.utils.data.DataLoader(ds, batch_sampler=sampler, **kw), sampler

    # ----------------------------------------------------------------------------------- train
    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def train(self, resume: str | None = "auto", dev_eval=None) -> dict:
        cfg, o, r = self.cfg, self.cfg.optim, self.cfg.run
        save_config(cfg, self.run_dir / "config.yaml")
        self.maybe_resume(resume)
        logger = RunLogger(self.run_dir, cfg.log, self.step if self.resumed_from else None, to_dict(cfg))
        (self.run_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "config_hash": config_hash(cfg),
                    "iters_per_epoch": self.iters_per_epoch,
                    "total_steps": self.total_steps,
                    "micro_batch": self.micro,
                    "accum_steps": o.accum_steps,
                    "effective_batch": o.batch_size,
                    "n_tiles": len(self.index),
                    "device": str(self.device),
                    "init": self.init_info,
                    "resumed_from": self.resumed_from,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        loader, _ = self.loader()
        it_loader = iter(loader)
        if getattr(self, "_pending_rng", None) is not None:
            set_rng_state(self._pending_rng)
            self._pending_rng = None
        t_start = time.time()
        last_ckpt = time.time()
        t_last_log, data_wait, n_since = time.time(), 0.0, 0
        freeze_steps = int(round(cfg.dino.freeze_last_layer_epochs * self.iters_per_epoch))
        last_layer = set(id(p) for p in self.student.head.last_layer_params())
        params = [p for p in self.student.parameters() if p.requires_grad]
        reason = "schedule_done"
        steps_this_run = 0
        while self.step < self.total_steps:
            it = self.step
            epoch = it // self.iters_per_epoch
            for i, g in enumerate(self.opt.param_groups):
                g["lr"] = float(self.sched["lr"][it])
                if i == 0:
                    g["weight_decay"] = float(self.sched["wd"][it])
            temp = float(self.sched["teacher_temp"][it])
            loss_sum, t_sum, t_cnt = 0.0, None, 0
            for j in range(o.accum_steps):
                m = it * o.accum_steps + j
                t0 = time.time()
                tiles = next(it_loader)
                data_wait += time.time() - t0
                tiles = tiles.to(self.device, non_blocking=True)
                gen = torch.Generator(device=self.device).manual_seed(micro_batch_seed(r.seed, m))
                crops = self.aug(tiles, gen)
                with self._autocast():
                    with torch.no_grad():
                        t_out = self.teacher_fwd(crops[:2])
                    s_out, s_feat = self.student_fwd(crops, return_backbone=True)
                    loss = self.loss_fn(s_out, t_out, temp) / o.accum_steps
                if not math.isfinite(float(loss.detach())):
                    raise FloatingPointError(f"non-finite loss at step {it}")
                self.scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
                ts = t_out.float().sum(0)
                t_sum = ts if t_sum is None else t_sum + ts
                t_cnt += t_out.shape[0]
            self.scaler.unscale_(self.opt)
            gnorm = clip_gradients(params, o.clip_grad, o.clip_mode)
            if it < freeze_steps:
                for p in params:
                    if id(p) in last_layer:
                        p.grad = None
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            ema_update(self.teacher, self.student, float(self.sched["momentum"][it]))
            self.loss_fn.update_center(t_sum, t_cnt)
            self.step += 1
            steps_this_run += 1
            n_since += 1

            if self.step % r.log_every_steps == 0 or self.step == self.total_steps:
                dt = time.time() - t_last_log
                mon = collapse_monitors(
                    s_out.detach(),
                    t_out.detach(),
                    self.loss_fn.center,
                    temp,
                    cfg.dino.student_temp,
                    s_feat.detach()[: t_out.shape[0]],
                )
                row = {
                    "epoch": epoch,
                    "loss": loss_sum,
                    "lr": float(self.sched["lr"][it]),
                    "wd": float(self.sched["wd"][it]),
                    "momentum": float(self.sched["momentum"][it]),
                    "teacher_temp": temp,
                    "grad_norm": gnorm,
                    "samples_per_s": n_since * o.batch_size / max(dt, 1e-9),
                    "data_wait_frac": data_wait / max(dt, 1e-9),
                    "effective_batch": o.batch_size,
                    **mon,
                }
                if self.device.type == "cuda":
                    row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                logger.log(self.step, row)
                t_last_log, data_wait, n_since = time.time(), 0.0, 0

            epoch_end = self.step % self.iters_per_epoch == 0
            ep_done = self.step // self.iters_per_epoch
            if (
                epoch_end
                and dev_eval is not None
                and cfg.eval.dev_every_epochs
                and ep_done % cfg.eval.dev_every_epochs == 0
            ):
                dm = dev_eval(self)
                logger.log(self.step, {f"dev/{k}": v for k, v in dm.items()})
            stop_now = None
            if self.stop:
                stop_now = "signal"
            elif r.max_steps and steps_this_run >= r.max_steps:
                stop_now = "max_steps"
            elif r.max_minutes and (time.time() - t_start) / 60 >= r.max_minutes:
                stop_now = "max_minutes"
            time_ckpt = r.ckpt_every_minutes and (time.time() - last_ckpt) / 60 >= r.ckpt_every_minutes
            step_ckpt = r.ckpt_every_steps and self.step % r.ckpt_every_steps == 0
            if epoch_end or time_ckpt or step_ckpt or stop_now or self.step == self.total_steps:
                keep = (
                    ep_done
                    if (epoch_end and r.keep_every_epochs and ep_done % r.keep_every_epochs == 0)
                    else None
                )
                self.checkpoint(keep)
                last_ckpt = time.time()
            if stop_now:
                reason = stop_now
                break
        logger.close()
        if self.step == self.total_steps:
            torch.save(
                {
                    "backbone": self.teacher.backbone.state_dict(),
                    "config": to_dict(cfg),
                    "channel_mean": self.channel_mean,
                    "channel_std": self.channel_std,
                },
                self.run_dir / "teacher_backbone.pt",
            )
        out = {
            "stopped": reason,
            "step": self.step,
            "total_steps": self.total_steps,
            "seconds": time.time() - t_start,
            "run_dir": str(self.run_dir),
        }
        (self.run_dir / "last_exit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out


def params_digest(model: torch.nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_equal(a: dict, b: dict) -> dict:
    """Bitwise comparison of two trainer state dicts (student, teacher, optimiser, center)."""
    res = {}
    for k in ("student", "teacher"):
        res[k] = all(torch.equal(a[k][n], b[k][n]) for n in a[k])
    res["center"] = torch.equal(a["center"], b["center"])
    oa, ob = a["optimizer"]["state"], b["optimizer"]["state"]
    res["optimizer"] = oa.keys() == ob.keys() and all(
        torch.equal(oa[i][n], ob[i][n]) for i in oa for n in oa[i] if isinstance(oa[i][n], torch.Tensor)
    )
    res["step"] = a["step"] == b["step"]
    return res


__all__ = ["Trainer", "default_run_dir", "params_digest", "state_equal", "np"]
