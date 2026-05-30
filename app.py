from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for, send_from_directory
from functools import wraps
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import threading
import queue
import shutil
import json
import re
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "farisvision_ai_secret_key_production_2026")

ADMIN_USERNAME = "agent_faris"
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

# ── PORTABILITAS PATH: Menggunakan relative path agar aman saat dideploy ──
BASE_DIR          = Path(__file__).resolve().parent
PROJECT_DIR       = BASE_DIR.parent.parent / "project_computer_vision/deteksi_objek_harian"
DATASET_DIR       = PROJECT_DIR / "dataset"
WEIGHTS_DIR       = PROJECT_DIR / "scripts/runs/runs/grafik_train/weights"
DATA_YAML         = DATASET_DIR / "data.yaml"
ALERT_DIR         = BASE_DIR / "alert_file"
MODEL_UPLOAD_DIR  = BASE_DIR.parent.parent / "project_computer_vision/model_cv"
IP_DATASET_DIR    = BASE_DIR / "ip_dataset"

# Buat direktori otomatis jika belum tersedia
ALERT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
IP_DATASET_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH         = WEIGHTS_DIR / "best.pt"
_active_model_path = MODEL_PATH        

# Inisialisasi Model dengan Fallback Device Aman
model = YOLO(str(MODEL_PATH))
try:
    model.to("mps")
except Exception:
    model.to("cpu")

CLASS_COLORS = {
    "Handphone": (255, 100, 50),
    "Jam":       (50, 200, 255),
    "Orang":     (50, 220, 100),
}
CLASSES = ["Handphone", "Jam", "Orang"]  

_train_queue  = queue.Queue()
_train_running = False
_last_screenshot = datetime.min

def save_alert(img):
    global _last_screenshot
    now = datetime.now()
    if (now - _last_screenshot).total_seconds() < 5:
        return None
    _last_screenshot = now
    filename = now.strftime("alert_%Y%m%d_%H%M%S.jpg")
    cv2.imwrite(str(ALERT_DIR / filename), img)
    return filename

def draw_detections(img, results):
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pad = 10
            x1, y1, x2, y2 = x1+pad, y1+pad, x2-pad, y2-pad

            color = CLASS_COLORS.get(label, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            conf_r = round(conf * 20) / 20
            text   = f"{label} {conf_r:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(img, text, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            detections.append({"label": label, "confidence": round(conf*100, 1)})

    alarm_classes = get_alarm_classes()
    alarm = any(d["label"] in alarm_classes for d in detections)
    return img, detections, alarm

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        
        # ── PERBAIKAN: Hanya mencocokkan username saja tanpa verifikasi password ──
        if username == ADMIN_USERNAME:
            session["logged_in"] = True
            session["username"]  = username
            return redirect(url_for("index"))
        error = "Username tidak terdaftar."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar"}), 400
    img_bytes = request.files["image"].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Format tidak valid"}), 400

    results = model.predict(img, conf=0.85, verbose=False)
    img, detections, alarm = draw_detections(img, results)
    if alarm:
        save_alert(img)

    _, buf = cv2.imencode(".jpg", img)
    return jsonify({"image": base64.b64encode(buf).decode(), "detections": detections, "alarm": alarm})

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Tidak ada frame"}), 400

    img_bytes = base64.b64decode(data["frame"].split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Frame tidak valid"}), 400

    results = model.predict(img, conf=0.85, verbose=False)
    img, detections, alarm = draw_detections(img, results)
    if alarm:
        save_alert(img)

    _, buf = cv2.imencode(".jpg", img)
    return jsonify({"image": base64.b64encode(buf).decode(), "detections": detections, "alarm": alarm})

@app.route("/train")
@login_required
def train_page():
    return render_template("train.html")

LABEL_ALERT_DIR = BASE_DIR / "label_alert"
COLORS_FILE     = LABEL_ALERT_DIR / "label_colors.json"
ICONS_FILE      = LABEL_ALERT_DIR / "label_icons.json"
ALERTS_FILE     = LABEL_ALERT_DIR / "label_alerts.json"

DEFAULT_COLORS  = ["#f97316", "#38bdf8", "#4ade80"]
DEFAULT_ICONS   = ["📱", "⌚", "👤"]
DEFAULT_ALERTS  = [True, False, False]  

def read_classes():
    import yaml
    if not DATA_YAML.exists():
        return CLASSES
    data = yaml.safe_load(DATA_YAML.read_text())
    return data.get("names", CLASSES)

def read_colors():
    if COLORS_FILE.exists():
        return json.load(COLORS_FILE.open())
    return DEFAULT_COLORS[:len(read_classes())]

def read_icons():
    if ICONS_FILE.exists():
        return json.load(ICONS_FILE.open())
    return DEFAULT_ICONS[:len(read_classes())]

def read_alerts():
    if ALERTS_FILE.exists():
        return json.load(ALERTS_FILE.open())
    return DEFAULT_ALERTS[:len(read_classes())]

def get_alarm_classes():
    classes = read_classes()
    alerts  = read_alerts()
    while len(alerts) < len(classes):
        alerts.append(False)
    return {cls for cls, on in zip(classes, alerts) if on}

def write_classes(names):
    import yaml
    data = yaml.safe_load(DATA_YAML.read_text()) if DATA_YAML.exists() else {"train": "../train/images", "val": "../valid/images", "nc": len(names)}
    data["names"] = names
    data["nc"]    = len(names)
    DATA_YAML.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

def write_colors(colors):
    LABEL_ALERT_DIR.mkdir(exist_ok=True, parents=True)
    json.dump(colors, COLORS_FILE.open("w"), ensure_ascii=False)

def write_icons(icons):
    LABEL_ALERT_DIR.mkdir(exist_ok=True, parents=True)
    json.dump(icons, ICONS_FILE.open("w"), ensure_ascii=False)

def write_alerts(alerts):
    LABEL_ALERT_DIR.mkdir(exist_ok=True, parents=True)
    json.dump(alerts, ALERTS_FILE.open("w"))

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

@app.route("/train/classes")
def get_classes():
    classes = read_classes()
    colors  = read_colors()
    icons   = read_icons()
    alerts  = read_alerts()
    while len(colors) < len(classes): colors.append("#a78bfa")
    while len(icons)  < len(classes): icons.append("🏷")
    while len(alerts) < len(classes): alerts.append(False)
    return jsonify({"classes": classes, "colors": colors, "icons": icons, "alerts": alerts})

@app.route("/train/add-class", methods=["POST"])
def add_class():
    data       = request.get_json() or {}
    name       = data.get("name", "").strip()
    color      = data.get("color", "#a78bfa")
    icon       = data.get("icon", "🏷")
    alert_on   = bool(data.get("alert", False))

    if not name:
        return jsonify({"error": "Nama kelas kosong"}), 400

    classes = read_classes()
    if name in classes:
        return jsonify({"error": f"Kelas '{name}' sudah ada"}), 409

    classes.append(name)
    write_classes(classes)

    colors = read_colors()
    while len(colors) < len(classes) - 1: colors.append("#a78bfa")
    colors.append(color)
    write_colors(colors)

    icons = read_icons()
    while len(icons) < len(classes) - 1: icons.append("🏷")
    icons.append(icon)
    write_icons(icons)

    alerts = read_alerts()
    while len(alerts) < len(classes) - 1: alerts.append(False)
    alerts.append(alert_on)
    write_alerts(alerts)

    CLASS_COLORS[name] = hex_to_bgr(color)
    return jsonify({"ok": True, "classes": classes, "colors": colors, "icons": icons, "alerts": alerts})

@app.route("/flow")
@login_required
def flow_page():
    return render_template("flow.html")

@app.route("/flow/metrics")
def flow_metrics():
    results_csv = PROJECT_DIR / "scripts/runs/runs/grafik_train/results.csv"
    if not results_csv.exists():
        return jsonify({"error": "Data tidak ditemukan"}), 404
    import csv
    rows = []
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch":      int(row["epoch"]),
                "precision":  round(float(row["metrics/precision(B)"]) * 100, 1),
                "recall":     round(float(row["metrics/recall(B)"]) * 100, 1),
                "mAP50":      round(float(row["metrics/mAP50(B)"]) * 100, 1),
                "mAP50_95":   round(float(row["metrics/mAP50-95(B)"]) * 100, 1),
                "train_loss": round(float(row["train/box_loss"]), 4),
                "val_loss":   round(float(row["val/box_loss"]), 4),
            })
    return jsonify(rows)

@app.route("/train/dataset-info")
def dataset_info():
    counts = {}
    for split in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split / "images"
        counts[split] = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) if img_dir.exists() else 0

    dynamic_classes = read_classes()
    class_counts = {c: 0 for c in dynamic_classes}
    label_dir = DATASET_DIR / "train" / "labels"
    if label_dir.exists():
        for lf in label_dir.glob("*.txt"):
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(dynamic_classes):
                        class_counts[dynamic_classes[cls_id]] += 1

    models = []
    active_str = str(_active_model_path)

    best_grafik = WEIGHTS_DIR / "best.pt"
    if best_grafik.exists():
        p = str(best_grafik)
        models.append({
            "name": best_grafik.name,
            "path": p,
            "active": p == active_str,
            "size_mb": round(best_grafik.stat().st_size / 1e6, 1),
            "modified": datetime.fromtimestamp(best_grafik.stat().st_mtime).strftime("%d %b %Y %H:%M"),
            "group": "Grafik Train"
        })

    best_new = MODEL_UPLOAD_DIR / "best_new.pt"
    if best_new.exists():
        p = str(best_new)
        models.append({
            "name": best_new.name,
            "path": p,
            "active": p == active_str,
            "size_mb": round(best_new.stat().st_size / 1e6, 1),
            "modified": datetime.fromtimestamp(best_new.stat().st_mtime).strftime("%d %b %Y %H:%M"),
            "group": "Upload Train"
        })
    return jsonify({"splits": counts, "classes": class_counts, "models": models})

@app.route("/train/upload", methods=["POST"])
def upload_sample():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar"}), 400

    annotations = json.loads(request.form.get("annotations", "[]"))
    file      = request.files["image"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem      = f"custom_{timestamp}"
    
    img_dir = DATASET_DIR / "train" / "images"
    lbl_dir = DATASET_DIR / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    img_path  = img_dir / f"{stem}.jpg"
    lbl_path  = lbl_dir / f"{stem}.txt"

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Format gambar tidak valid"}), 400

    cv2.imwrite(str(img_path), img)
    h, w = img.shape[:2]
    lines = []
    for ann in annotations:
        cls_id   = ann["class_id"]
        ann_type = ann.get("type", "bbox")

        if ann_type == "polygon":
            pts    = ann["points"]
            coords = " ".join(f"{p['x']/w:.6f} {p['y']/h:.6f}" for p in pts)
            lines.append(f"{cls_id} {coords}")
        else:
            x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    lbl_path.write_text("\n".join(lines))
    return jsonify({"ok": True, "saved": stem})

@app.route("/train/start", methods=["POST"])
def start_training():
    global _train_running
    if _train_running:
        return jsonify({"error": "Training sedang berjalan"}), 409

    cfg      = request.get_json() or {}
    epochs   = int(cfg.get("epochs", 30))
    model_sz = cfg.get("model_size", "yolov8n")
    imgsz    = int(cfg.get("imgsz", 640))
    batch    = int(cfg.get("batch", 8))

    def run_training_safe():
        global _train_running, model
        _train_running = True

        best_new_path = MODEL_UPLOAD_DIR / "best_new.pt"
        if best_new_path.exists():
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
            backup_path = MODEL_UPLOAD_DIR / f"best_{timestamp}.pt"
            shutil.copy(str(best_new_path), str(backup_path))
            _train_queue.put({"type": "log", "msg": f"💾 Backup model lama berhasil"})
            base_model = str(best_new_path)
        else:
            base_model = f"{model_sz}.pt"

        _train_queue.put({"type": "start", "msg": f"🚀 Mulai training | base: {Path(base_model).name} | {epochs} epochs", "total_epochs": epochs})

        train_run_dir = PROJECT_DIR / "scripts/runs/web_train"
        run_dir       = PROJECT_DIR / "scripts/runs"

        try:
            def on_train_epoch_end(trainer):
                _train_queue.put({
                    "type": "epoch",
                    "current": trainer.epoch + 1,
                    "total": trainer.epochs,
                    "msg": f"Epoch {trainer.epoch + 1}/{trainer.epochs} completed."
                })

            train_model = YOLO(base_model)
            train_model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            device_target = "mps" if cv2.ocl.haveOpenCL() else "cpu"

            train_model.train(
                data=str(DATA_YAML),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device_target,
                project=str(run_dir),
                name='web_train',
                exist_ok=True,
                verbose=False
            )

            new_best = train_run_dir / "weights/best.pt"
            if new_best.exists():
                shutil.copy(str(new_best), str(best_new_path))
                _train_queue.put({"type": "done", "msg": f"✅ Training selesai! Model diperbarui → model_cv/best_new.pt"})
            else:
                _train_queue.put({"type": "done", "msg": "✅ Training selesai! (weights tidak ditemukan)"})

        except Exception as e:
            _train_queue.put({"type": "error", "msg": f"❌ Training gagal: {str(e)}"})
        finally:
            _train_running = False

    threading.Thread(target=run_training_safe, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/train/log-stream")
def log_stream():
    def generate():
        yield "data: {\"type\":\"connected\",\"msg\":\"Terhubung ke log stream\"}\n\n"
        while True:
            try:
                item = _train_queue.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: {\"type\":\"ping\"}\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.route("/train/switch-model", methods=["POST"])
def switch_model():
    global model, _active_model_path
    data = request.get_json() or {}
    path = data.get("path")
    if not path or not Path(path).exists():
        return jsonify({"error": "Model tidak ditemukan"}), 404
    try:
        model = YOLO(path)
        try:
            model.to("mps")
        except Exception:
            model.to("cpu")
        _active_model_path = Path(path)
        return jsonify({"ok": True, "msg": f"Model diganti ke {Path(path).name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train/status")
def train_status():
    return jsonify({"running": _train_running})

@app.route("/model/info")
def model_info():
    classes = read_classes()
    icons   = read_icons()
    alerts  = read_alerts()
    while len(icons)  < len(classes): icons.append("🏷")
    while len(alerts) < len(classes): alerts.append(False)

    model_name = _active_model_path.name if _active_model_path.exists() else "best.pt"
    is_upload  = str(_active_model_path).startswith(str(MODEL_UPLOAD_DIR))

    return jsonify({
        "model_name":    model_name,
        "model_path":    str(_active_model_path),
        "is_upload":     is_upload,
        "classes":       classes,
        "icons":         icons,
        "alerts":        alerts,
    })

@app.route("/evaluate")
@login_required
def evaluate_page():
    return render_template("evaluate.html")

@app.route("/evaluate/metrics")
def evaluate_metrics():
    import csv
    results = []
    run_dirs = [
        {"key": "grafik_train", "label": "Model Utama (grafik_train)",
         "csv": PROJECT_DIR / "scripts/runs/runs/grafik_train/results.csv",
         "weights": WEIGHTS_DIR / "best.pt"},
        {"key": "web_train", "label": "Upload Train (web_train)",
         "csv": PROJECT_DIR / "scripts/runs/web_train/results.csv",
         "weights": MODEL_UPLOAD_DIR / "best_new.pt"},
    ]

    for run in run_dirs:
        csv_path = run["csv"]
        if not csv_path.exists():
            continue
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "epoch":      int(row["epoch"]),
                    "precision":  round(float(row["metrics/precision(B)"]) * 100, 1),
                    "recall":     round(float(row["metrics/recall(B)"]) * 100, 1),
                    "mAP50":      round(float(row["metrics/mAP50(B)"]) * 100, 1),
                    "mAP50_95":   round(float(row["metrics/mAP50-95(B)"]) * 100, 1),
                    "train_loss": round(float(row["train/box_loss"]), 4),
                    "val_loss":   round(float(row["val/box_loss"]), 4),
                })
        if not rows:
            continue
        best = rows[-1]
        wt = run["weights"]
        size_mb = round(wt.stat().st_size / 1e6, 1) if wt.exists() else None
        results.append({
            "key":       run["key"],
            "label":     run["label"],
            "epochs":    len(rows),
            "best":      best,
            "rows":      rows,
            "model_path": str(wt) if wt.exists() else None,
            "size_mb":   size_mb,
            "active":    str(wt) == str(_active_model_path),
        })
    return jsonify(results)

@app.route("/dataset/capture", methods=["POST"])
@login_required
def capture_dataset():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Tidak ada frame"}), 400
    try:
        img_bytes = base64.b64decode(data["frame"].split(",")[1])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Frame tidak valid"}), 400
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
        filename = f"capture_{ts}.jpg"
        cv2.imwrite(str(IP_DATASET_DIR / filename), img)
        count = len(list(IP_DATASET_DIR.glob("*.jpg")))
        return jsonify({"ok": True, "filename": filename, "count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/dataset/list")
@login_required
def dataset_list():
    files = sorted(IP_DATASET_DIR.glob("*.jpg"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return jsonify({
        "count": len(files),
        "files": [f.name for f in files[:30]],
        "path":  str(IP_DATASET_DIR)
    })

@app.route("/dataset/image/<filename>")
@login_required
def dataset_image(filename):
    safe_filename = secure_filename(filename)
    return send_from_directory(str(IP_DATASET_DIR), safe_filename)

@app.route("/dataset/delete/<filename>", methods=["DELETE"])
@login_required
def delete_capture(filename):
    safe_filename = secure_filename(filename)
    target = IP_DATASET_DIR / safe_filename
    if target.exists() and target.is_file():
        target.unlink()
    count = len(list(IP_DATASET_DIR.glob("*.jpg")))
    return jsonify({"ok": True, "count": count})

@app.route("/network-info")
@login_required
def network_info():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return jsonify({"ip": ip, "port": 5002, "url": f"http://{ip}:5002"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)