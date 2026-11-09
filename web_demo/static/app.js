const $ = id => document.getElementById(id);
let file = null, aspect = 0;            // aspect = width / height of the uploaded photo
let origUrl = null, origSize = "", bg = "keep", done = false;

const RES = 384;                        // model input size (recipe F)
const size = b => b < 1048576 ? (b / 1024).toFixed(0) + " KB" : (b / 1048576).toFixed(1) + " MB";
const pct = p => (p * 100).toFixed(1) + " %";
const DASH = "–";

function steps(res, colour, back, time) {
  $("s_res").textContent = res; $("votes").textContent = colour;
  $("s_bg").textContent = back; $("time").textContent = time;
}

function verdict(cls, name, sub, conf = "") {
  $("verdict").className = "verdict " + cls;
  $("name").textContent = name; $("sub").textContent = sub; $("conf").textContent = conf;
}

function idle() {
  verdict("idle", "—", file ? "Press Recognize to classify the photo." : "Upload an image and press Recognize.");
  steps(DASH, DASH, DASH, DASH);
  $("marked").hidden = true; $("busy").hidden = true; $("wait").hidden = false;
}

/* panel 1 shows the original photo, or the background-removed image the model received */
function showPhoto(src, note) {
  $("preview").src = src;
  $("dims").textContent = note || origSize;
}

/* background switch: Kept / Removed; re-runs the recognition when a result is already shown */
function setBg(v) {
  if (bg === v) return;
  bg = v;
  $("bgkeep").classList.toggle("on", v === "keep");
  $("bgremove").classList.toggle("on", v === "remove");
  if (file && done) recognize();
}
$("bgkeep").addEventListener("click", () => setBg("keep"));
$("bgremove").addEventListener("click", () => setBg("remove"));

/* The block height is set by the left column. The right column gets the width at which
   both frames have the photo's shape, and each frame wraps its image exactly. */
function layout() {
  const main = document.querySelector("main"), frames = [$("f1"), $("f2")];
  frames.forEach(f => { f.classList.remove("fit"); f.style.width = f.style.height = ""; });
  if (!aspect || innerWidth <= 860) { main.style.gridTemplateColumns = ""; return; }
  const extra = 2 + 20 + 2;                                   // panel border + body padding + frame border
  const h = Math.min(frames[0].clientHeight, frames[1].clientHeight);
  const left = document.querySelector(".left").offsetWidth;
  const maxRight = Math.max(220, main.clientWidth - 28 - 14 - left);
  const right = Math.max(170, Math.min(maxRight, h * aspect + extra));
  main.style.gridTemplateColumns = `var(--left) ${Math.round(right)}px`;
  const w = Math.min(right - extra, h * aspect);
  frames.forEach(f => {
    f.classList.add("fit");
    f.style.width = Math.round(w) + "px";
    f.style.height = Math.round(Math.min(h, w / aspect)) + "px";
  });
}
addEventListener("resize", layout);

/* remove the grey letterbox padding of the 384 px model input, so the map has the photo's shape */
function cropMap(src) {
  return new Promise(res => {
    const im = new Image();
    im.onload = () => {
      const iw = aspect >= 1 ? 1 : aspect, ih = aspect >= 1 ? 1 / aspect : 1;
      const nw = Math.round(RES * iw), nh = Math.round(RES * ih);
      const l = Math.floor((RES - nw) / 2), t = Math.floor((RES - nh) / 2);
      const k = im.naturalWidth / RES, c = document.createElement("canvas");
      c.width = Math.round(nw * k); c.height = Math.round(nh * k);
      c.getContext("2d").drawImage(im, l * k, t * k, nw * k, nh * k, 0, 0, c.width, c.height);
      res(c.toDataURL("image/png"));
    };
    im.src = src;
  });
}

function status(text, ok) {
  $("fstatus").textContent = text ? " · " + text : "";
  $("fstatus").className = "fstatus" + (ok ? " done" : "");
}

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  file = f; done = false;
  origUrl = URL.createObjectURL(f);
  const probe = new Image();
  probe.onload = () => {
    origSize = `${probe.naturalWidth} × ${probe.naturalHeight} px`;
    aspect = probe.naturalWidth / probe.naturalHeight;
    $("dims").textContent = origSize; layout();
  };
  probe.src = origUrl;
  $("preview").src = origUrl; $("preview").hidden = false; $("nohint").hidden = true;
  $("thumb").src = origUrl; $("thumb").hidden = false;
  $("fname").textContent = f.name; $("fsize").textContent = size(f.size); status("Ready", false);
  $("go").disabled = false; $("clear").disabled = false; $("remove").disabled = false;
  idle();
}

function clearAll() {
  file = null; aspect = 0; origUrl = null; origSize = ""; done = false; $("file").value = "";
  $("preview").hidden = true; $("nohint").hidden = false; $("dims").textContent = "";
  $("thumb").hidden = true; $("fname").textContent = "No file selected"; $("fsize").innerHTML = "&nbsp;"; status("", false);
  $("go").disabled = true; $("clear").disabled = true; $("remove").disabled = true;
  idle(); layout();
}

$("file").addEventListener("change", e => setFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", e => { e.preventDefault(); setFile(e.dataTransfer.files[0]); });
$("clear").addEventListener("click", clearAll);
$("remove").addEventListener("click", clearAll);
$("go").addEventListener("click", () => recognize());

async function recognize() {
  if (!file) return;
  $("wait").hidden = true; $("marked").hidden = true; $("busy").hidden = false;
  $("go").disabled = true; $("bgkeep").disabled = $("bgremove").disabled = true;
  status(bg === "remove" ? "Removing background…" : "Analyzing…", false);
  const fd = new FormData(); fd.append("image", file); fd.append("bg", bg);
  try {
    const r = await fetch("/predict", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Request failed.");
    await render(d); status("Completed", true);
  } catch (e) {
    $("busy").hidden = true; $("wait").hidden = false;
    verdict("error", "Error", e.message); steps(DASH, DASH, DASH, DASH);
    status("Failed", false);
  }
  $("go").disabled = false; $("bgkeep").disabled = $("bgremove").disabled = false;
}

async function render(d) {
  done = true;
  if (d.shown) showPhoto(d.shown, d.bg === "removed" ? "background removed" : "removal failed, original used");
  else showPhoto(origUrl);
  $("marked").src = await cropMap(d.marked);
  $("busy").hidden = true; $("marked").hidden = false;
  const time = d.time != null ? d.time.toFixed(2) + " s" : DASH;
  const res = `${d.res || RES} × ${d.res || RES} px`;
  const back = d.bg === "removed" ? "Removed" : d.bg === "failed" ? "Kept*" : "Kept";
  if (d.verdict === "none") {
    verdict("none", "Not recognised", "Not one of the six weapons, or an unclear photo, or below the demo threshold.");
  } else {
    verdict(d.verdict, d.cls, d.english, pct(d.p));
  }
  steps(res, "RGB", back, time);
}

idle();
