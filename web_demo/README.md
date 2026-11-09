# BD-LW Local Weapon Recognition (demo web interface)

Upload a photo and the demo shows which of the six BD-LW classes our model assigns
(Churi, Dao, Khor, Kodal, Kural, Shabol), its confidence, and the Grad-CAM map of the model input.
The background can be kept (as in the paper) or removed before recognition.

## Quick start (commands)

Windows PowerShell, run in the project folder (the one that contains `app.py`):

```
python -m venv venv                      # once: create the virtual environment
venv\Scripts\activate                    # every time: activate it
pip install -r requirements.txt          # once: install Flask, PyTorch, timm, rembg, ...
python -c "from rembg import new_session; new_session('isnet-general-use')"   # once: download IS-Net (~170 MB)
python app.py                            # start the app, then open http://127.0.0.1:5000
```

macOS / Linux: use `source venv/bin/activate` instead of `venv\Scripts\activate`.
Before the first start, copy the five trained models into `weights/` (step 4 below).
To stop the app press Ctrl + C; after changing any file, restart `python app.py` and reload the page with Ctrl + F5.

## What this demo is and is not

- **`PAPER_MODE = True` (default, top of `model.py`)**: the demo runs the evaluated model of the paper:
  the five ConvNeXt-Tiny E1 fold models, recipe F (384 px letterbox, grey padding 128, horizontal-flip averaging)
  and the same Grad-CAM as Notebook 10 (fold 0). Confidence is the mean over the five fold models and two views
  of softmax(logits / T), T = 0.3919, exactly as in Notebook 7.
  Results below the demo threshold of 0.95 (`DEMO_THRESHOLD`) are shown as "Not recognised". The paper's own threshold
  (0.9997, Section 4.11 and Supplementary Table S11) keeps 95 % of the BD-LW test weapons and accepted none
  of the 112 images without weapons, but it rejects most photos taken outside the BD-LW capture conditions.
  At 0.95, 15 of those 112 images (13.4 %) would still be accepted as a weapon, so a "weapon" result is not
  proof that the image shows one of the six weapons. The threshold is a demo setting, not a result of the paper.
- **`PAPER_MODE = False`**: an extra, demo-only filter based on a general ImageNet classifier (ResNet-50) reports
  photos that do not show one of the six weapons as "Not one of the six local weapons". Its thresholds were set
  by hand on a few images. This filter is **not** part of the method or the results reported in the paper.

## Run in PyCharm

1. **Open the folder.** Unzip and use *File → Open* on the `bdlw_web` folder.
2. **Create the interpreter.** *Settings → Project → Python Interpreter → Add Interpreter → Virtualenv* (Python 3.10–3.13).
3. **Install packages.** In the PyCharm terminal (with the virtual environment active):
   ```
   pip install -r requirements.txt
   python -c "from rembg import new_session; new_session('isnet-general-use')"
   ```
   The second command downloads the IS-Net background-removal model once (about 170 MB, needs internet).
   If it cannot be downloaded, save `isnet-general-use.onnx` from
   https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx into `C:\Users\<you>\.u2net\`.
4. **Add the trained models.** From the Notebook 2 output, copy the five checkpoints of
   `results/all/convnext_tiny/F/weights/` into `weights/` and rename them:
   ```
   fold0.pt -> weights/convnext_tiny_fold0.pt
   fold1.pt -> weights/convnext_tiny_fold1.pt
   fold2.pt -> weights/convnext_tiny_fold2.pt
   fold3.pt -> weights/convnext_tiny_fold3.pt
   fold4.pt -> weights/convnext_tiny_fold4.pt
   weights/convnext_tiny_norm.json   (already included)
   ```
   With only one file (`weights/convnext_tiny.pt`) the demo still runs, but the threshold below was derived
   for the five-fold mean, so use all five folds to reproduce the paper.
5. **Start the app.** `python app.py`, then open **http://127.0.0.1:5000** in the browser.

## Optional demo filter (`PAPER_MODE = False`)

The general ImageNet model is downloaded automatically the first time (internet needed once). An image is reported
as not one of the six weapons when it looks like a firearm, when it contains almost nothing tool-like (also checked
on the region marked by our model's Grad-CAM), or when the BD-LW probability is below 70 %.

Optional stricter check: `python build_bank.py --data "<dataset folder>" --manifest splits/manifest.csv`
adds a similarity test against the **training** photos. Always pass `--manifest`, so that test images are not used.

Settings at the top of `model.py`: `DEMO_THRESHOLD` (0.95), `PAPER_MODE`, and for the optional filter `HIGH`, `LOW`, `TOOL_MIN`, `CROP_TOOL_MIN`.

## Background switch (Kept / Removed)

"Removed" runs the same background removal as the paper's preprocessing (IS-Net via rembg, `isnet-general-use`;
mask at 0.5, gaps closed, small holes filled, specks removed, weapon on grey 128) and then classifies the result with
the same recipe-F models, which were trained with the background kept. The switch therefore shows how much a prediction
depends on the background (Section 4.2 of the paper). The IS-Net weights (about 170 MB) are downloaded once on first use.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask server (run this) |
| `model.py` | Model loading, preprocessing (recipe F), background removal, Grad-CAM, optional demo filter |
| `build_bank.py` | Optional bank of training-image embeddings for the demo filter |
| `templates/index.html`, `static/` | Web page, style and script |
