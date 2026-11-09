"""Build the open-set bank once, so that guns, bottles and other objects are reported as
"Not a local weapon" instead of being forced into one of the six classes.

    python build_bank.py --data "H:/BD-LW_public"
    python build_bank.py --data "H:/BD-LW_public" --manifest splits/manifest.csv   (optional: training images only)

--data is the folder that contains Churi/, Dao/, Khor/, Kodal/, Kural/, Shabol/ (sub-folders are searched).
Needs internet the first time, to download the general ImageNet model.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

import model as M

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--manifest", default=None)
ap.add_argument("--per-class", type=int, default=250)
ap.add_argument("--percentile", type=float, default=1.0, help="lower tail of training similarities that is still accepted")
a = ap.parse_args()

root = Path(a.data)
items = []                                          # (path, class, group)
if a.manifest:
    man = pd.read_csv(a.manifest)
    man = man[man.split == "trainval"]
    for r in man.itertuples():
        items.append((root / r.source_file, r.label, getattr(r, "group_id", r.Index)))
else:
    for c in M.CLASSES:
        for p in sorted((root / c).rglob("*")):
            if p.suffix.lower() in EXT:
                items.append((p, c, str(p)))
random.seed(0)
sel = []
for c in M.CLASSES:
    ci = [it for it in items if it[1] == c and it[0].exists()]
    sel += random.sample(ci, min(len(ci), a.per_class))
if not sel:
    raise SystemExit(f"No images found under {root}. It must contain the folders {', '.join(M.CLASSES)}.")
if M.generic()[0] is None:
    raise SystemExit("The general ImageNet model could not be downloaded. Connect to the internet and run again.")

feats, top1, groups = [], [], []
for i, (p, c, g) in enumerate(sel):
    try:
        gp, gf = M.generic_features(M.read_image(p))
    except Exception as e:
        print("skip", p, e); continue
    feats.append(gf); top1.append(int(gp.argmax())); groups.append(str(g))
    if i % 100 == 0:
        print(f"{i}/{len(sel)}")
F_ = np.stack(feats); groups = np.array(groups)

# nearest training photo for every training photo, ignoring photos of the same object group
S = F_ @ F_.T
S[groups[:, None] == groups[None, :]] = -1
nn = S.max(1)
thr = float(np.percentile(nn, a.percentile)) - 0.02

# ImageNet objects that the training photos themselves look like (seen at least twice)
vals, cnt = np.unique(top1, return_counts=True)
allowed = sorted(set(vals[cnt >= 2].tolist()) | M.TOOL_LIKE)

M.BANK.mkdir(exist_ok=True)
np.savez(M.BANK / "generic.npz", feats=F_, threshold=thr, allowed=np.array(allowed))
names = M.generic()[1]
print(f"\nsaved bank/generic.npz: {len(F_)} images, similarity threshold {thr:.3f}")
print("ImageNet objects accepted:", ", ".join(names[i] for i in allowed))
