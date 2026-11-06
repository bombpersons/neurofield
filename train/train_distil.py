"""Train the dream creature network (dream.py) on distilled targets (dream_targets.py).

    python train/train_distil.py --data data/dream_targets --out runs/distil --steps 30

CLIP alone could not teach the small net (train_dream.py: its parts never came to match
their phrases), so CLIP draws each crop's creature first, optimised per crop, and the
net learns those drawings: L1 on the layer (premultiplied) and alpha, and LPIPS on the
layer laid over the level's frame and over grey (where the frame cannot hide it). The
targets start from the net's own input noise, so for a given input there is one target
to learn, not a spread to blur together.

The phrases' CLIP text embeddings are saved in the checkpoint (`text`, with `vocab`),
as in train_dream.py, so the engine needs a table, not CLIP.
"""

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPTokenizer

from creatures import composite
from dream import Q_PHRASE, VOCAB, DreamNet, fields
from train_dream import CLIP, embed

WARMUP = 500


class Targets(Dataset):
    """The crops of some target shards: per item parts [MAX_PARTS, Q_PARAMS], the crop
    [C_PARAMS], the level frame [3, n, n], the colour scheme [PALETTE, 3] and the target
    layer [4, n, n] (the last kept step)."""

    def __init__(self, files):
        cols = {k: [] for k in ("q", "cam", "frame", "palette", "target")}
        for f in files:
            d = np.load(f)
            for k in ("q", "cam", "frame", "palette"):
                cols[k].append(d[k])
            cols["target"].append(d["targets"][:, -1])
        self.cols = {k: np.concatenate(v) for k, v in cols.items()}

    def __len__(self):
        return len(self.cols["q"])

    def __getitem__(self, i):
        c = self.cols
        return (
            torch.from_numpy(c["q"][i]),
            torch.from_numpy(c["cam"][i]),
            torch.from_numpy(c["frame"][i]).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(c["palette"][i]),
            torch.from_numpy(c["target"][i]).permute(2, 0, 1).float() / 255.0,
        )


class Trainer:
    def __init__(self, data, out, batch=16, lr=3e-4, workers=2, device=None, lpips_weight=0.5, ema_decay=0.999):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.device == "cuda"
        files = sorted(Path(data).glob("**/targets_*.npz"))
        assert len(files) >= 2, f"need at least two target shards (the last one is held out), found {len(files)}"
        vocab = json.loads(next(Path(data).glob("**/vocab.json")).read_text())
        assert vocab == VOCAB, "the targets were made with another vocabulary"
        self.train_set, self.val_set = Targets(files[:-1]), Targets(files[-1:])
        clip = CLIPModel.from_pretrained(CLIP).to(self.device).eval()
        tok = CLIPTokenizer.from_pretrained(CLIP)
        with torch.no_grad():
            self.text = torch.cat([
                embed(clip.get_text_features(**tok(VOCAB[i : i + 256], padding=True, return_tensors="pt").to(self.device)))
                for i in range(0, len(VOCAB), 256)
            ])
        del clip
        self.model = DreamNet(text_dim=self.text.shape[1]).to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval().requires_grad_(False)
        self.lr, self.lpips_weight, self.ema_decay = lr, lpips_weight, ema_decay
        self.step = 0
        ckpt = self.out / "ckpt.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.ema.load_state_dict(state["ema"])
            self.opt.load_state_dict(state["opt"])
            self.step = state["step"]
            print(f"resumed at step {self.step}")
        self.loader = DataLoader(self.train_set, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                                 persistent_workers=workers > 0)
        self.batches = (b for _ in itertools.count() for b in self.loader)
        self.watch = next(iter(DataLoader(self.val_set, batch_size=6, sampler=np.linspace(0, len(self.val_set) - 1, 6).astype(int))))
        params = sum(p.numel() for p in self.model.parameters())
        print(f"{params / 1e6:.2f}M params, {len(self.train_set)} train / {len(self.val_set)} val crops, {self.device}")

    def forward(self, batch, model=None):
        """(predicted layer, target layer, level frame) for a batch."""
        q, cam, frame, palette, target = (x.to(self.device, non_blocking=True) for x in batch)
        with torch.no_grad():
            feats, weights, _, _ = fields(q, cam, palette, frame.shape[-1])
        # In fp32, as in train_dream.py: without normalisation the net can outgrow fp16.
        pred = (model or self.model)(feats, weights, self.text[q[..., Q_PHRASE].long()], palette)
        return pred.float(), target, frame

    def losses(self, pred, target, frame):
        """(loss, layer L1, LPIPS of the composites) for a batch."""
        pa, ta = pred[:, 3:], target[:, 3:]
        l1 = (pred[:, :3] * pa - target[:, :3] * ta).abs().mean() + (pa - ta).abs().mean()
        grey = torch.full_like(frame, 0.5)
        backs = torch.cat([frame, grey])
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            perceptual = self.lpips(composite(backs, pred.repeat(2, 1, 1, 1)) * 2 - 1,
                                    composite(backs, target.repeat(2, 1, 1, 1)) * 2 - 1).float().mean()
        return l1 + self.lpips_weight * perceptual, l1, perceptual

    def run(self, minutes=None, steps=None, log_every=50, sample_every=1000):
        self.model.train()
        start, done, logs = time.time(), 0, {k: [] for k in ("loss", "l1", "lpips")}
        while (steps is None or done < steps) and (minutes is None or time.time() - start < minutes * 60):
            for g in self.opt.param_groups:
                g["lr"] = self.lr * min(1.0, (self.step + 1) / WARMUP)
            out = self.losses(*self.forward(next(self.batches)))
            self.opt.zero_grad(set_to_none=True)
            out[0].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if torch.isfinite(out[0]):
                self.opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(self.ema.parameters()), list(self.model.parameters()), 1 - self.ema_decay)
            for k, v in zip(logs, out):
                logs[k].append(v.item())
            self.step += 1
            done += 1
            if self.step % log_every == 0:
                self.log(**{f"train_{k}": float(np.nanmean(v[-log_every:])) for k, v in logs.items()}, secs=time.time() - start)
            if self.step % sample_every == 0:
                self.checkpoint()
        self.checkpoint()

    @torch.no_grad()
    def validate(self, batches=16):
        rs = [[x.item() for x in self.losses(*self.forward(b, self.ema))[1:]]
              for b in itertools.islice(DataLoader(self.val_set, batch_size=16), batches)]
        return dict(zip(("val_l1", "val_lpips"), np.mean(rs, 0).tolist()))

    @torch.no_grad()
    def sample(self, path):
        """Rows of the level frame | target over it | the net's over it | target over grey | the net's over grey | the net's alpha."""
        pred, target, frame = self.forward(self.watch, self.ema)
        grey = torch.full_like(frame, 0.5)
        rows = [torch.cat([frame[i], composite(frame, target)[i], composite(frame, pred)[i], composite(grey, target)[i],
                           composite(grey, pred)[i], pred[i, 3:].expand(3, -1, -1)], 2) for i in range(len(pred))]
        grid = torch.cat(rows, 1).clamp(0, 1)
        Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)

    def checkpoint(self):
        self.log(**self.validate())
        self.sample(self.out / f"sample_{self.step:07d}.png")
        self.sample(self.out / "sample_latest.png")
        torch.save({"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                    "step": self.step, "vocab": VOCAB, "text": self.text.cpu()}, self.out / "ckpt.pt")

    def log(self, **kv):
        rec = {"step": self.step, **kv}
        print(json.dumps(rec), flush=True)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lpips", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=1000)
    a = ap.parse_args()
    t = Trainer(a.data, a.out, batch=a.batch, lr=a.lr, workers=a.workers, lpips_weight=a.lpips)
    t.run(minutes=a.minutes, steps=a.steps, log_every=a.log_every, sample_every=a.sample_every)


if __name__ == "__main__":
    main()
