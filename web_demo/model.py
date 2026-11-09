"""Model loading and prediction (same preprocessing as recipe F in the paper:
letterbox to 384 px with grey padding, ImageNet normalisation, horizontal-flip averaging)."""
import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
WEIGHTS, BANK = ROOT / "weights", ROOT / "bank"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES, PAD = 384, 128
CLASSES = ["Churi", "Dao", "Khor", "Kodal", "Kural", "Shabol"]
ENGLISH = {"Churi": "Knife", "Dao": "Machete", "Khor": "Straight razor",
           "Kodal": "Hoe", "Kural": "Axe", "Shabol": "Iron crowbar"}
# best model in the paper (highest CV macro-F1, recipe F); temperature from calibration
MODELS = {"convnext_tiny": ("ConvNeXt-Tiny", 0.3919)}

HIGH, LOW = 0.90, 0.70   # clear match / lowest probability still accepted as a weapon

# PAPER_MODE = True : the evaluated model of the paper (five fold models, recipe F, Grad-CAM as in Notebook 10)
#                     with only the demo confidence threshold below.
# PAPER_MODE = False: adds the demo-only filter below that reports images without one of the six
#                     weapons as "Not one of the six local weapons". This filter is NOT part of the paper.
PAPER_MODE = True

# Demo threshold. The paper's 95 %-recall threshold for ConvNeXt-Tiny is 0.9997 (Section 4.11,
# Supplementary Table S11), which rejects most photos taken outside the BD-LW capture conditions.
# The demo uses 0.95 instead. On the paper's 112 images without weapons, 15 (13.4 %) reach 0.95
# (results/no_weapon/no_weapon_per_image.csv, ConvNeXt-Tiny, five-fold mean).
# Confidence = mean over the five E1 fold models and two views (original + flip) of softmax(logits / T),
# exactly as in Notebook 7.
PAPER_THRESHOLD = 0.9997   # the paper's 95 %-recall threshold (reference only, not used below)
DEMO_THRESHOLD = 0.95      # threshold used by this demo

# ---------------------------------------------------------------------------------------------
# Open-set check. The BD-LW model only knows six classes, so any photo gets one of them.
# A second, general ImageNet model (downloaded once by timm) tells us what else the photo shows:
#   1. firearms are rejected,
#   2. (with bank) a confident ImageNet object that never appears in the BD-LW training photos is rejected,
#   3. a photo far from every training photo (cosine similarity of general features) is rejected.
#   4. a photo with almost no "tool-like" content (hand, bottle, phone, person, animal) is rejected.
# Rules 1 and 4 work without any dataset. build_bank.py (optional) adds rules 2 and 3.
# ---------------------------------------------------------------------------------------------
GENERIC = "resnet50.a1_in1k"       # general ImageNet-1k model from timm (downloaded once)
FIREARMS = {413, 763, 764, 597, 471}      # assault rifle, revolver, rifle, holster, cannon
OBJECT_CONF = 0.50                 # ImageNet confidence above which an unrelated object is trusted
FIREARM_CONF = 0.20
TOOL_MIN = 0.08                    # minimum ImageNet probability on tool-like objects (no bank needed)
CROP_TOOL_MIN = 0.15               # stricter limit for the second look at our model's Grad-CAM region
# ImageNet classes that BD-LW weapons resemble (hatchet, cleaver, shovel, hammer, hook, ...)
TOOL_LIKE = {596, 499, 623, 792, 587, 784, 726, 777, 600, 488, 677, 783, 473, 512, 813, 618, 693,
             462, 840, 491, 730, 477, 910, 769}
_generic = {}


def generic():
    """General ImageNet model: (net, labels) or (None, None) when it cannot be downloaded."""
    if "net" not in _generic:
        try:
            net = timm.create_model(GENERIC, pretrained=True).eval().to(DEVICE)
            from timm.data import ImageNetInfo
            info = ImageNetInfo()
            _generic["net"], _generic["names"] = net, [info.index_to_description(i).split(",")[0] for i in range(1000)]
        except Exception as e:
            print("General model not available (needs internet once):", e)
            _generic["net"], _generic["names"] = None, None
    return _generic["net"], _generic["names"]


@torch.no_grad()
def generic_features(rgb):
    """ImageNet probabilities and L2-normalised embedding of the letterboxed image (224 px)."""
    net, _ = generic()
    if net is None:
        return None, None
    x = cv2.resize(letterbox(rgb), (224, 224), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None] / 255
    x = ((x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)).to(DEVICE)
    f = net.forward_head(net.forward_features(x), pre_logits=True)
    p = F.softmax(net.get_classifier()(f), 1)[0].cpu().numpy()
    return p, F.normalize(f, dim=1)[0].cpu().numpy()


_bank = {}


def bank():
    if "b" not in _bank:
        f = BANK / "generic.npz"
        if f.exists():
            b = np.load(f)
            _bank["b"] = {"f": b["feats"], "thr": float(b["threshold"]), "allowed": set(b["allowed"].tolist())}
        else:
            _bank["b"] = None
    return _bank["b"]

# ---------------------------------------------------------------------------------------------
# Background removal, exactly as in the paper's preprocessing notebook (Section 3.3.1):
# IS-Net (rembg "isnet-general-use"), mask binarised at 0.5, gaps closed, small holes filled,
# specks removed, weapon placed on uniform grey (128) at its original position.
# The photo is first standardised like the dataset (longer side at most 1,600 px, Lanczos).
# Note: the demo applies the same recipe-F models (trained with the background kept) to the
# background-removed photo, so the switch shows how much the prediction depends on the background.
# ---------------------------------------------------------------------------------------------
BG_MODEL = "isnet-general-use"
BG_GRAY = (128, 128, 128)
MASK_THRESHOLD, MIN_COMPONENT_FRAC, MAX_HOLE_FRAC = 0.5, 0.05, 0.001
FAIL_MIN_FG, FAIL_MAX_FG = 0.005, 0.95
MAX_SIDE = 1600
_bg = {}


def standardize(rgb):
    h, w = rgb.shape[:2]; s = MAX_SIDE / max(h, w)
    if s >= 1:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((round(w * s), round(h * s)), Image.LANCZOS))


def clean_mask(alpha):
    h, w = alpha.shape
    binm = alpha >= int(MASK_THRESHOLD * 255)
    k = max(3, int(round(min(h, w) * 0.005)) | 1)
    binm = cv2.morphologyEx(binm.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    n_bg, bg_lab, bg_stats, _ = cv2.connectedComponentsWithStats((~binm).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([bg_lab[0], bg_lab[-1], bg_lab[:, 0], bg_lab[:, -1]])))
    for lab in range(1, n_bg):
        if lab not in border and bg_stats[lab, cv2.CC_STAT_AREA] < MAX_HOLE_FRAC * h * w:
            binm[bg_lab == lab] = True
    n_fg, fg_lab, fg_stats, _ = cv2.connectedComponentsWithStats(binm.astype(np.uint8), connectivity=8)
    if n_fg > 1:
        areas = fg_stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, a in enumerate(areas) if a >= MIN_COMPONENT_FRAC * areas.max()]
        binm = np.isin(fg_lab, keep)
    zone = cv2.dilate(binm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    soft = np.where(zone, alpha, 0).astype(np.uint8)
    soft[binm] = 255
    return soft, binm


def remove_background(rgb):
    """Returns (image on grey background, status). status is "ok" or "failed" (original kept, as in the paper)."""
    if "s" not in _bg:
        from rembg import new_session            # pip install rembg  (downloads IS-Net once)
        _bg["s"] = new_session(BG_MODEL)
    from rembg import remove
    rgb = standardize(rgb)
    alpha = np.asarray(remove(Image.fromarray(rgb), session=_bg["s"], only_mask=True).convert("L"))
    if alpha.shape != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    soft, binm = clean_mask(alpha)
    fg = float(binm.mean())
    if fg < FAIL_MIN_FG or fg > FAIL_MAX_FG:
        return rgb, "failed"
    a = (soft.astype(np.float32) / 255.0)[..., None]
    bg = np.array(BG_GRAY, dtype=np.float32)[None, None, :]
    return np.clip(rgb.astype(np.float32) * a + bg * (1 - a) + 0.5, 0, 255).astype(np.uint8), "ok"


def to_uri(rgb, max_side=1200):
    im = Image.fromarray(rgb); im.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_cache = {}


def fold_files(name):
    """Five fold checkpoints (weights/<name>_fold0.pt ... _fold4.pt) if present, else the single weights/<name>.pt."""
    folds = [WEIGHTS / f"{name}_fold{k}.pt" for k in range(5)]
    folds = [f for f in folds if f.exists()]
    return folds or ([WEIGHTS / f"{name}.pt"] if (WEIGHTS / f"{name}.pt").exists() else [])


def available():
    return [k for k in MODELS if fold_files(k)]


def load(name):
    """Returns (list of networks, mean, std, temperature). The first network is fold 0 (used for Grad-CAM)."""
    if name in _cache:
        return _cache[name]
    nets = []
    for f in fold_files(name):
        kw = dict(pretrained=False, num_classes=len(CLASSES))
        if name.startswith("vit"):
            kw["img_size"] = RES
        net = timm.create_model(name, **kw)
        sd = torch.load(f, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict({k.replace("module.", ""): v.float() for k, v in sd.items()})
        nets.append(net.eval().to(DEVICE))
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    nj = WEIGHTS / f"{name}_norm.json"
    if nj.exists():
        n = json.loads(nj.read_text()); n = n.get("0", n); mean, std = n["mean"], n["std"]
    _cache[name] = (nets, torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1), MODELS[name][1])
    return _cache[name]


def read_image(file):
    return np.asarray(ImageOps.exif_transpose(Image.open(file)).convert("RGB"))


def letterbox(img):
    h, w = img.shape[:2]; s = RES / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.full((RES, RES, 3), PAD, np.uint8)
    t, l = (RES - nh) // 2, (RES - nw) // 2
    out[t:t + nh, l:l + nw] = small
    return out


@torch.no_grad()
def embed_and_predict(nets, mean, std, temp, rgb):
    """As in the paper (Notebook 7): mean over all fold models and two views (original + flip)
    of softmax(logits / T). Also returns the mean L2-normalised embedding."""
    x = torch.from_numpy(letterbox(rgb)).permute(2, 0, 1).float()[None] / 255
    x = ((x - mean) / std).to(DEVICE)
    ps, fs = [], []
    for net in nets:
        for xi in (x, torch.flip(x, dims=[3])):
            f = net.forward_head(net.forward_features(xi), pre_logits=True)
            ps.append(F.softmax(net.get_classifier()(f) / temp, 1)); fs.append(f)
    p = torch.stack(ps).mean(0)[0].cpu().numpy()
    # class chosen by each fold model (mean of its two views), used to report fold agreement
    embed_and_predict.fold_votes = [int(torch.stack(ps[i:i + 2]).mean(0).argmax()) for i in range(0, len(ps), 2)]
    return p, F.normalize(torch.stack(fs).mean(0), dim=1)[0].cpu().numpy()


def gradcam(net, x):
    """Grad-CAM exactly as in the paper (Notebooks 3 and 10): target layer = last ConvNeXt stage,
    class = the model's prediction on this single (unflipped) input, ReLU of the gradient-weighted
    activations, bilinear upsampling to the 384 px input, division by the maximum."""
    st = {}
    def hook(m, i, o):
        st["a"] = o; o.register_hook(lambda g: st.__setitem__("g", g))
    h = net.stages[-1].register_forward_hook(hook)
    net.zero_grad()
    with torch.enable_grad():
        logits = net(x)
        cls = int(logits.argmax(1)); logits[0, cls].backward()
    h.remove()
    a, g = st["a"].detach(), st["g"].detach()
    cam = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1))[0].float().cpu().numpy()
    cam = cv2.resize(cam, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, cls


def overlay(rgb384, cam=None):
    """Same overlay as the paper figure: 0.55 x input + 0.45 x inferno colour map, on the 384 px
    letterboxed image that the model actually receives. Without a map the model input is shown."""
    img = rgb384.astype(np.float32) / 255
    if cam is not None:
        heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1] / 255
        img = np.clip(0.55 * img + 0.45 * heat, 0, 1)
    out = (img * 255).round().astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(out).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cam_crop(rgb, cam384, thr=0.5, pad=0.15):
    """Crop of the original image around the strongest Grad-CAM region (largest component of cam >= thr)."""
    H, W = rgb.shape[:2]; s = RES / max(H, W)
    nh, nw = round(H * s), round(W * s); t, l = (RES - nh) // 2, (RES - nw) // 2
    m = (cam384 >= thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[i, :4]
    p = pad * max(w, h)
    x0, y0 = max(0, int((x - p - l) / s)), max(0, int((y - p - t) / s))
    x1, y1 = min(W, int((x + w + p - l) / s)), min(H, int((y + h + p - t) / s))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return rgb[y0:y1, x0:x1]


def art(n):
    return ("an " if n[:1].lower() in "aeiou" else "a ") + n


def predict(rgb, name, bg="keep"):
    t0 = time.perf_counter()
    bg_status, shown = "kept", None
    if bg == "remove":
        rgb, st = remove_background(rgb)
        bg_status = "removed" if st == "ok" else "failed"
        shown = to_uri(rgb)
    nets, mean, std, temp = load(name)
    net = nets[0]                                    # fold 0, as in the paper's Grad-CAM figure
    p, _ = embed_and_predict(nets, mean, std, temp, rgb)
    k = int(p.argmax()); pk = float(p[k])
    votes = sum(v == k for v in embed_and_predict.fold_votes)

    lb = letterbox(rgb)
    x = ((torch.from_numpy(lb).permute(2, 0, 1).float()[None] / 255 - mean) / std).to(DEVICE)
    cam, cam_cls = None, None

    reason, sim, seen, tool = None, None, None, None
    gp, gf = (None, None) if PAPER_MODE else generic_features(rgb)
    if gp is not None:
        names = generic()[1]
        g = int(gp.argmax()); gpk = float(gp[g]); seen = f"{names[g]} ({gpk * 100:.0f} %)"
        b = bank()
        allowed = b["allowed"] if b else TOOL_LIKE
        fire = max(FIREARMS, key=lambda i: gp[i])
        tool = float(gp[list(TOOL_LIKE)].sum())
        if tool < TOOL_MIN and pk >= LOW:
            # busy scene: also look at the region our model used (its Grad-CAM hotspot)
            cam, cam_cls = gradcam(net, x)
            crop = cam_crop(rgb, cam)
            if crop is not None:
                cp, _ = generic_features(crop)
                if float(cp[list(TOOL_LIKE)].sum()) >= CROP_TOOL_MIN:
                    tool = float(cp[list(TOOL_LIKE)].sum())
        if gp[fire] >= FIREARM_CONF and fire == g:
            reason = f"The image looks like a firearm ({names[fire]}). Firearms are not among the six local weapons."
        elif tool < TOOL_MIN:
            reason = f"No weapon-like object was found in the image (it looks like {art(names[g])})."
        elif b is not None and gpk >= OBJECT_CONF and g not in allowed:
            reason = f"The image looks like {art(names[g])}, not one of the six local weapons."
        elif b is not None:
            sim = float((b["f"] @ gf).max())
            if sim < b["thr"]:
                reason = "The image is not similar to any photo of the six local weapons in the training data."
    if PAPER_MODE and pk < DEMO_THRESHOLD:
        reason = f"Confidence {pk * 100:.2f} % is below the demo threshold of {DEMO_THRESHOLD * 100:.0f} %."
    if not PAPER_MODE and reason is None and pk < LOW:
        reason = f"The model is not confident enough (highest probability {pk * 100:.1f} %)."

    verdict = "none" if reason else ("clear" if pk >= HIGH else "likely")
    if verdict == "none":
        cam, cam_cls = None, None
    elif cam is None:
        cam, cam_cls = gradcam(net, x)
    marked = overlay(lb, cam)
    order = np.argsort(-p)
    return {"verdict": verdict, "reason": reason, "cls": CLASSES[k], "english": ENGLISH[CLASSES[k]], "p": pk,
            "similarity": sim, "seen": seen, "marked": marked,
            "cam_cls": None if cam_cls is None else CLASSES[cam_cls], "openset": gp is not None, "paper_mode": PAPER_MODE, "bank": bank() is not None,
            "model": MODELS[name][0], "folds": len(nets), "threshold": DEMO_THRESHOLD if PAPER_MODE else LOW,
            "votes": votes, "bg": bg_status, "shown": shown, "input": [int(rgb.shape[1]), int(rgb.shape[0])], "res": RES, "time": round(time.perf_counter() - t0, 2),
            "ranking": [{"cls": CLASSES[i], "english": ENGLISH[CLASSES[i]], "p": float(p[i])} for i in order]}
