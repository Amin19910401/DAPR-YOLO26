# -*- coding: utf-8 -*-
"""
Training pipeline for DAPR-YOLO26 on the two-class EAGLE benchmark.

This script trains DAPR-YOLO26 (3-head: P2, P3, P4) from scratch on the
EAGLE dataset, reports test metrics (mAP@0.5, mAP@0.5:0.95), per-class
metrics (AP, Precision, Recall, F1), parameter count, GFLOPs, and FPS,
and generates HTML reports, confusion matrices, and PR curves.
"""

import os

os.environ['TORCH_HOME'] = 'D:/torch_cache'
os.environ['ULTRALYTICS_DATA_DIR'] = 'D:/Ultralytics_Data'

import sys
import json
import time
import random
import shutil
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import yaml
import torch
from ultralytics import YOLO
from scipy import stats


# =================================================================================
# Configuration
# =================================================================================

RANDOM_SEED = 42
NEW_SEEDS = [RANDOM_SEED]

SELECTED_CLASSES = [0, 1]
SELECTED_CLASS_NAMES = {0: "car", 1: "truck"}
NUM_CLASSES = len(SELECTED_CLASSES)

IMAGE_SIZES_CONFIG = [
    {
        'imgsz': 640,
        'batch': 4,
        'workers': 4,
        'lr0': 0.01,
        'mosaic': 0.8,
        'cache': 'disk',
        'patience': 15,
    },
]

PROJECT_ROOT_DIR = "D:/YOLO26_Training_Runs"
RESULTS_DIR = "D:/Training_Results_2classes"
DATASET_BASE = r"D:/Doustali/Dataset"

ARCHITECTURES = [
    {
        'name': 'DAPR-YOLO26s-3head',
       'yaml': r"D:/DAPR-YOLO26/configs/dapr_yolo26s_3head.yaml",
        'project_name': 'DAPR-YOLO26s-3head',
        'root_dir': 'D:/eagle',
        'color': '#85c1e9',
    },
]


# =================================================================================
# Utilities
# =================================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def print_header(text, char="=", length=70):
    print("\n" + char * length)
    print(f" {text}")
    print(char * length)


def print_success(text):
    print(f"   [OK] {text}")


def print_warning(text):
    print(f"   [WARN] {text}")


def print_error(text):
    print(f"   [ERROR] {text}")


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def safe_metric_attr(obj, path, default=0.0):
    try:
        cur = obj
        for p in path.split("."):
            cur = getattr(cur, p)
        return float(cur)
    except Exception:
        return default


def get_class_ap_array(metrics, num_classes):
    ap_values = [0.0] * num_classes
    try:
        maps = getattr(metrics.box, "maps", None)
        if maps is not None:
            maps = np.array(maps).reshape(-1)
            for i in range(min(num_classes, len(maps))):
                ap_values[i] = float(maps[i] if maps[i] is not None else 0.0)
            return ap_values
    except Exception:
        pass
    return ap_values


def check_gpu_or_exit():
    print_header("Hardware check")
    if not torch.cuda.is_available():
        print_error("CUDA is not available.")
        sys.exit(1)
    print_success(f"GPU: {torch.cuda.get_device_name(0)}")
    print_success(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print_success(f"CUDA: {torch.version.cuda}")
    return 'cuda'


def check_files():
    print_header("Required files check")
    all_ok = True

    data_yaml = os.path.join(DATASET_BASE, "data.yaml")
    if not os.path.exists(data_yaml):
        print_error(f"data.yaml not found: {data_yaml}")
        all_ok = False
    else:
        print_success(f"data.yaml: {data_yaml}")

    for arch in ARCHITECTURES:
        if not os.path.exists(arch['yaml']):
            print_error(f"YAML not found: {arch['yaml']}")
            all_ok = False
        else:
            print_success(f"{os.path.basename(arch['yaml'])}")

    print("\n   Selected classes:")
    for idx, cls_id in enumerate(SELECTED_CLASSES):
        print(f"      {idx}: {SELECTED_CLASS_NAMES[cls_id]} (ID: {cls_id})")

    return all_ok


# =================================================================================
# Hyperparameters
# =================================================================================

def save_hyperparameters(results_dir):
    hyperparams = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_name': 'DAPR-YOLO26_2Classes_Training',
        'seed': RANDOM_SEED,
        'classes': {
            'selected_ids': SELECTED_CLASSES,
            'selected_names': SELECTED_CLASS_NAMES,
            'num_classes': NUM_CLASSES,
        },
        'image_sizes': IMAGE_SIZES_CONFIG,
        'architectures': [
            {
                'name': a['name'],
                'yaml_file': os.path.basename(a['yaml']),
                'project_name': a['project_name'],
            }
            for a in ARCHITECTURES
        ],
        'training_params': {
            'epochs': 100,
            'patience': 15,
            'save_period': 10,
            'seed': RANDOM_SEED,
        },
        'augmentation': {
            'hsv_h': 0.01, 'hsv_s': 0.6, 'hsv_v': 0.4,
            'translate': 0.08, 'scale': 0.5, 'fliplr': 0.5,
            'mosaic': 0.8, 'degrees': 3.0, 'flipud': 0.1,
            'mixup': 0.15, 'shear': 2.0, 'copy_paste': 0.08,
        },
        'loss_weights': {'box': 7.5, 'cls': 0.5, 'dfl': 1.5},
        'optimizer': {'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005},
    }

    hyperparams_path = os.path.join(results_dir, 'hyperparameters.json')
    save_json(hyperparams_path, hyperparams)
    print_success(f"Hyperparameters saved: {hyperparams_path}")
    return hyperparams


# =================================================================================
# Measurement utilities
# =================================================================================

def get_model_params_and_flops(model_path, imgsz=640):
    print("\nComputing Params and GFLOPs...")
    model = YOLO(model_path)
    model.model.to('cuda')
    model.model.eval()
    total_params = sum(p.numel() for p in model.model.parameters())
    total_params_millions = total_params / 1_000_000

    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to('cuda')
        flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
        print_success(f"Parameters: {total_params_millions:.2f}M, GFLOPs: {gflops:.2f}")
        return total_params_millions, gflops
    except ImportError:
        gflops = (total_params * 2 * imgsz * imgsz) / 1e9
        print_success(f"Parameters: {total_params_millions:.2f}M, GFLOPs: {gflops:.2f} (approx)")
        return total_params_millions, gflops


def measure_fps_on_real_images(model_path, data_yaml, imgsz=640, num_samples=30):
    print("\nMeasuring FPS on real images...")
    model = YOLO(model_path)

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    test_dir = os.path.join(os.path.dirname(data_yaml), data['test'])

    if not os.path.exists(test_dir):
        return 0.0, 0.0

    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        return 0.0, 0.0
    if len(images) > num_samples:
        images = random.sample(images, num_samples)

    sample_images = []
    for img_name in images[:num_samples]:
        img = cv2.imread(os.path.join(test_dir, img_name))
        if img is not None:
            sample_images.append(img)

    if not sample_images:
        return 0.0, 0.0

    for _ in range(3):
        _ = model.predict(sample_images, imgsz=imgsz, device='cuda', verbose=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model.predict(sample_images, imgsz=imgsz, device='cuda', verbose=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    fps = (len(sample_images) / avg_time) * 1000
    fps_std = fps * (std_time / avg_time)

    print_success(f"FPS: {fps:.1f} +/- {fps_std:.1f}")
    return fps, fps_std


# =================================================================================
# Training
# =================================================================================

def train_model_for_size(arch_config, imgsz_config, results_dir, seed=42):
    imgsz = imgsz_config['imgsz']
    batch = imgsz_config['batch']
    workers = imgsz_config.get('workers', 4)
    lr0 = imgsz_config.get('lr0', 0.01)
    mosaic = imgsz_config.get('mosaic', 0.8)
    cache = imgsz_config.get('cache', False)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_name = f"{arch_config['project_name']}_{imgsz}_seed{seed}_{timestamp}"
    project_root_dir = arch_config['root_dir']

    print_header(f"Training: {arch_config['name']} (imgsz={imgsz})")
    print(f"   Save path: {project_root_dir}/{project_name}")
    print(f"   Input size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Learning rate: {lr0}")
    print(f"   Classes: {', '.join(SELECTED_CLASS_NAMES.values())}")

    start_time = datetime.now()

    try:
        model = YOLO(arch_config['yaml'])
        model.to('cuda')

        model.train(
            data=os.path.join(DATASET_BASE, "data.yaml"),
            epochs=100,
            patience=imgsz_config.get('patience', 15),
            batch=batch,
            imgsz=imgsz,
            device='cuda',
            workers=workers,
            save_period=10,
            plots=True,
            save=True,
            exist_ok=True,
            project=project_root_dir,
            name=project_name,
            seed=seed,
            deterministic=False,
            cache=cache,
            lr0=lr0,
            classes=SELECTED_CLASSES,
            optimizer='auto',
            verbose=True,
            pretrained=False,
            amp=True,
            hsv_h=0.01, hsv_s=0.6, hsv_v=0.4,
            translate=0.08, scale=0.5, fliplr=0.5,
            mosaic=mosaic, degrees=3.0, flipud=0.1,
            mixup=0.15, shear=2.0, copy_paste=0.08,
            box=7.5, cls=0.5, dfl=1.5,
        )

        best_model_path = f"{project_root_dir}/{project_name}/weights/best.pt"
        if not os.path.exists(best_model_path):
            return None

        test_results = model.val(
            data=os.path.join(DATASET_BASE, "data.yaml"),
            split='test',
            batch=batch,
            imgsz=imgsz,
            device='cuda',
            conf=0.25,
            iou=0.6,
            classes=SELECTED_CLASSES,
            plots=True,
            save_json=True,
            verbose=True,
        )

        test_map50 = safe_metric_attr(test_results, "box.map50", 0.0)
        test_map = safe_metric_attr(test_results, "box.map", 0.0)
        test_precision = safe_metric_attr(test_results, "box.mp", 0.0)
        test_recall = safe_metric_attr(test_results, "box.mr", 0.0)
        per_class_ap = get_class_ap_array(test_results, NUM_CLASSES)

        per_class_precision = [0.0] * NUM_CLASSES
        per_class_recall = [0.0] * NUM_CLASSES
        per_class_f1 = [0.0] * NUM_CLASSES

        try:
            if hasattr(test_results.box, 'p') and test_results.box.p is not None:
                p_vals = test_results.box.p
                r_vals = test_results.box.r
                f1_vals = test_results.box.f1
                for i in range(min(NUM_CLASSES, len(p_vals))):
                    per_class_precision[i] = float(p_vals[i] if p_vals[i] is not None else 0.0)
                    per_class_recall[i] = float(r_vals[i] if r_vals[i] is not None else 0.0)
                    per_class_f1[i] = float(f1_vals[i] if f1_vals[i] is not None else 0.0)
        except Exception as e:
            print_warning(f"Error extracting per-class metrics: {e}")

        params_millions, gflops = get_model_params_and_flops(best_model_path, imgsz)
        fps_real, fps_real_std = measure_fps_on_real_images(
            best_model_path, os.path.join(DATASET_BASE, "data.yaml"), imgsz
        )

        duration = str(datetime.now() - start_time).split('.')[0]

        result = {
            'architecture': arch_config['name'],
            'imgsz': imgsz,
            'project_name': project_name,
            'project_path': f"{project_root_dir}/{project_name}",
            'color': arch_config['color'],
            'test_map50': round(test_map50, 4),
            'test_map': round(test_map, 4),
            'test_precision': round(test_precision, 4),
            'test_recall': round(test_recall, 4),
            'params_millions': round(params_millions, 2),
            'gflops': round(gflops, 2),
            'fps_real': round(fps_real, 1),
            'fps_real_std': round(fps_real_std, 1),
            'per_class_ap': [round(float(x), 4) for x in per_class_ap],
            'per_class_precision': [round(float(x), 4) for x in per_class_precision],
            'per_class_recall': [round(float(x), 4) for x in per_class_recall],
            'per_class_f1': [round(float(x), 4) for x in per_class_f1],
            'duration': duration,
            'success': True,
            'classes': SELECTED_CLASSES,
            'class_names': SELECTED_CLASS_NAMES,
        }

        print(f"\n   Results (imgsz={imgsz}):")
        print(f"      mAP@0.5:      {test_map50:.4f}")
        print(f"      mAP@0.5:0.95: {test_map:.4f}")
        print(f"      FPS:          {fps_real:.1f}")

        print(f"\n   Per-class AP (mAP@0.5:0.95):")
        for i, (cls_id, cls_name) in enumerate(SELECTED_CLASS_NAMES.items()):
            ap = per_class_ap[i] if i < len(per_class_ap) else 0
            print(f"      {cls_name}: {ap:.4f}")

        print(f"\n   Per-class Precision / Recall / F1:")
        for i, (cls_id, cls_name) in enumerate(SELECTED_CLASS_NAMES.items()):
            prec = per_class_precision[i] if i < len(per_class_precision) else 0
            rec = per_class_recall[i] if i < len(per_class_recall) else 0
            f1 = per_class_f1[i] if i < len(per_class_f1) else 0
            print(f"      {cls_name}: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")

        json_path = os.path.join(results_dir, f'results_{project_name}.json')
        save_json(json_path, result)

        return result

    except Exception as e:
        print_error(f"Training error (imgsz={imgsz}): {e}")
        traceback.print_exc()
        return None


# =================================================================================
# Report generation
# =================================================================================

def generate_confusion_matrix(project_path, output_dir, model_name, imgsz):
    print(f"\nCopying Confusion Matrix for {model_name} (imgsz={imgsz})...")
    candidates = [
        f"{project_path}/confusion_matrix_normalized.png",
        f"{project_path}/confusion_matrix.png",
    ]
    for src in candidates:
        if os.path.exists(src):
            dst = os.path.join(
                output_dir,
                f'confusion_matrix_{model_name.replace(" ", "_")}_{imgsz}.png'
            )
            shutil.copy2(src, dst)
            print_success(f"Confusion Matrix copied")
            return
    print_warning(f"Confusion Matrix not found")


def generate_pr_curve(project_path, output_dir, model_name, imgsz):
    print(f"\nCopying PR Curve for {model_name} (imgsz={imgsz})...")
    candidates = [
        f"{project_path}/PR_curve.png",
        f"{project_path}/pr_curve.png",
    ]
    for src in candidates:
        if os.path.exists(src):
            dst = os.path.join(
                output_dir,
                f'pr_curve_{model_name.replace(" ", "_")}_{imgsz}.png'
            )
            shutil.copy2(src, dst)
            print_success(f"PR Curve copied")
            return
    print_warning(f"PR Curve not found")


def generate_sample_detections(project_path, data_yaml, output_dir, model_name, imgsz, num_samples=6):
    print(f"\nGenerating sample detections for {model_name} (imgsz={imgsz})...")
    try:
        model_path = os.path.join(project_path, 'weights/best.pt')
        if not os.path.exists(model_path):
            print_warning(f"best.pt not found at {model_path}")
            return
        model = YOLO(model_path)

        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        test_dir = os.path.join(os.path.dirname(data_yaml), data['test'])

        if not os.path.exists(test_dir):
            print_warning(f"Test directory not found: {test_dir}")
            return

        images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print_warning("No images in test directory")
            return
        if len(images) > num_samples:
            images = random.sample(images, num_samples)

        sample_dir = os.path.join(
            output_dir,
            f'sample_detections_{model_name.replace(" ", "_")}_{imgsz}'
        )
        os.makedirs(sample_dir, exist_ok=True)

        saved_count = 0
        for img_name in images:
            img_path = os.path.join(test_dir, img_name)
            results = model.predict(
                img_path, conf=0.50, device='cuda',
                classes=SELECTED_CLASSES, verbose=False
            )
            if results and len(results) > 0:
                for r in results:
                    if r.plot() is not None:
                        cv2.imwrite(
                            os.path.join(sample_dir, f'detection_{img_name}'),
                            r.plot()
                        )
                        saved_count += 1
                        break

        if saved_count > 0:
            print_success(f"{saved_count} sample images saved to {sample_dir}")
        else:
            print_warning("No images saved")

    except Exception as e:
        print_warning(f"Error generating sample detections: {e}")


# =================================================================================
# HTML reports
# =================================================================================

def create_individual_report(result, output_dir):
    f1 = 2 * (result['test_precision'] * result['test_recall']) / (
        result['test_precision'] + result['test_recall']
    ) if (result['test_precision'] + result['test_recall']) > 0 else 0
    class_list = "<br>".join([f"&bull; {name}" for name in SELECTED_CLASS_NAMES.values()])

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Report {result['architecture']} - imgsz={result['imgsz']}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; }}
h1, h2 {{ color: #2c3e50; border-bottom: 2px solid {result['color']}; padding-bottom: 10px; }}
.metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 8px; min-width: 150px; }}
.metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
.metric-label {{ font-size: 12px; color: #7f8c8d; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
th {{ background: #3498db; color: white; }}
</style></head>
<body>
<div class="container">
    <h1>{result['architecture']} (imgsz={result['imgsz']})</h1>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Training duration:</strong> {result['duration']}</p>
    <p><strong>Classes:</strong><br>{class_list}</p>

    <h2>Model statistics</h2>
    <div>
        <div class="metric"><div class="metric-value">{result['params_millions']:.2f}M</div><div class="metric-label">Parameters</div></div>
        <div class="metric"><div class="metric-value">{result['gflops']:.2f}</div><div class="metric-label">GFLOPs</div></div>
        <div class="metric"><div class="metric-value">{result['fps_real']:.1f} &plusmn; {result['fps_real_std']:.1f}</div><div class="metric-label">FPS</div></div>
    </div>

    <h2>Overall metrics</h2>
    <div>
        <div class="metric"><div class="metric-value">{result['test_map50']:.4f}</div><div class="metric-label">mAP@0.5</div></div>
        <div class="metric"><div class="metric-value">{result['test_map']:.4f}</div><div class="metric-label">mAP@0.5:0.95</div></div>
        <div class="metric"><div class="metric-value">{result['test_precision']:.4f}</div><div class="metric-label">Precision</div></div>
        <div class="metric"><div class="metric-value">{result['test_recall']:.4f}</div><div class="metric-label">Recall</div></div>
        <div class="metric"><div class="metric-value">{f1:.4f}</div><div class="metric-label">F1-Score</div></div>
    </div>

    <h2>Per-class metrics</h2>
    <table>
        <tr>
            <th>Class</th>
            <th>AP (mAP@0.5:0.95)</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
        </tr>'''

    for i, (cls_id, cls_name) in enumerate(SELECTED_CLASS_NAMES.items()):
        ap = result['per_class_ap'][i] if i < len(result['per_class_ap']) else 0
        prec = result['per_class_precision'][i] if i < len(result['per_class_precision']) else 0
        rec = result['per_class_recall'][i] if i < len(result['per_class_recall']) else 0
        f1_cls = result['per_class_f1'][i] if i < len(result['per_class_f1']) else 0
        html_content += f'''
        <tr>
            <td style="font-weight:bold;">{cls_name}</td>
            <td>{ap:.4f}</td>
            <td>{prec:.4f}</td>
            <td>{rec:.4f}</td>
            <td>{f1_cls:.4f}</td>
        </tr>'''

    html_content += '''
    </table>
</div></body></html>'''

    report_path = os.path.join(output_dir, f'report_{result["project_name"]}.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print_success(f"HTML report saved: {report_path}")


def create_comparison_report(all_results, output_dir):
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Comparison Report - 2 Classes (car, truck)</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1400px; margin: auto; background: white; padding: 20px; border-radius: 10px; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 30px; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
th {{ background: #3498db; color: white; font-weight: bold; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
</style></head>
<body>
<div class="container">
    <h1>Comparison Report - 2 Classes (car, truck)</h1>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Full comparison</h2>
    <table>
        <tr>
            <th>Architecture</th>
            <th>imgsz</th>
            <th>mAP@0.5</th>
            <th>mAP@0.5:0.95</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Params (M)</th>
            <th>GFLOPs</th>
            <th>FPS</th>
            <th>Duration</th>
        </tr>'''

    for r in all_results:
        html_content += f'''
        <tr>
            <td>{r['architecture']}</td>
            <td><strong>{r['imgsz']}</strong></td>
            <td>{r['test_map50']:.4f}</td>
            <td>{r['test_map']:.4f}</td>
            <td>{r['test_precision']:.4f}</td>
            <td>{r['test_recall']:.4f}</td>
            <td>{r['params_millions']:.2f}</td>
            <td>{r['gflops']:.2f}</td>
            <td>{r['fps_real']:.1f}</td>
            <td>{r['duration']}</td>
        </tr>'''

    html_content += '''
    </table>
</div></body></html>'''

    report_path = os.path.join(output_dir, 'comparison_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print_success(f"Comparison report saved: {report_path}")


def generate_per_class_ap_plot(all_results, output_dir, imgsz):
    print(f"\nGenerating per-class AP plot (imgsz={imgsz})...")
    results_this_size = [r for r in all_results if r['imgsz'] == imgsz]
    if not results_this_size:
        return

    fig, axes = plt.subplots(1, len(results_this_size), figsize=(6 * len(results_this_size), 6))
    if len(results_this_size) == 1:
        axes = [axes]

    fig.suptitle(
        f'Per-Class AP (mAP@0.5:0.95) - imgsz={imgsz}',
        fontsize=14, fontweight='bold'
    )

    for idx, r in enumerate(results_this_size):
        ax = axes[idx]
        aps = r['per_class_ap']
        class_names = list(SELECTED_CLASS_NAMES.values())
        plot_names = class_names[:len(aps)]

        colors = ['#27ae60' if ap > 0.7 else '#f39c12' if ap > 0.5 else '#e74c3c' for ap in aps]
        bars = ax.bar(plot_names, aps, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(r['architecture'], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('AP (mAP@0.5:0.95)')
        ax.tick_params(axis='x', rotation=15, labelsize=10)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, aps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9
            )

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f'per_class_ap_comparison_{imgsz}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_success(f"Plot saved: {chart_path}")


def save_per_class_metrics_to_txt(all_results, output_dir):
    txt_path = os.path.join(output_dir, 'per_class_metrics.txt')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Per-class metrics report (AP, Precision, Recall, F1)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Classes: car (0), truck (1)\n")
        f.write("=" * 80 + "\n\n")

        for r in all_results:
            f.write(f"Architecture: {r['architecture']}\n")
            f.write(f"   Image size: {r['imgsz']}\n")
            f.write(f"   mAP@0.5: {r['test_map50']:.4f}   |   mAP@0.5:0.95: {r['test_map']:.4f}\n\n")

            f.write("   " + "-" * 60 + "\n")
            f.write("   {:>10} | {:>10} | {:>10} | {:>10} | {:>10}\n".format(
                "Class", "AP", "Precision", "Recall", "F1"
            ))
            f.write("   " + "-" * 60 + "\n")

            for i, (cls_id, cls_name) in enumerate(SELECTED_CLASS_NAMES.items()):
                ap = r['per_class_ap'][i] if i < len(r['per_class_ap']) else 0.0
                prec = r['per_class_precision'][i] if i < len(r['per_class_precision']) else 0.0
                rec = r['per_class_recall'][i] if i < len(r['per_class_recall']) else 0.0
                f1 = r['per_class_f1'][i] if i < len(r['per_class_f1']) else 0.0
                f.write("   {:>10} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f}\n".format(
                    cls_name, ap, prec, rec, f1
                ))

            f.write("\n" + "-" * 80 + "\n\n")

        if all_results:
            best = max(all_results, key=lambda x: x['test_map'])
            f.write("\nBest model by mAP@0.5:0.95:\n")
            f.write(f"   Architecture: {best['architecture']}\n")
            f.write(f"   Image size: {best['imgsz']}\n")
            f.write(f"   mAP@0.5:0.95: {best['test_map']:.4f}\n")

    print_success(f"Per-class metrics report saved: {txt_path}")


def generate_latex_table(all_results, output_dir):
    latex = r"""\begin{table}[htbp]
\centering
\caption{Performance comparison on the EAGLE benchmark.}
\label{tab:comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{imgsz} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} & \textbf{Precision} & \textbf{Recall} & \textbf{FPS} \\
\hline
"""

    for r in all_results:
        latex += f"{r['architecture']} & {r['imgsz']} & "
        latex += f"{r['test_map50']:.4f} & "
        latex += f"{r['test_map']:.4f} & "
        latex += f"{r['test_precision']:.4f} & "
        latex += f"{r['test_recall']:.4f} & "
        latex += f"{r['fps_real']:.1f} \\\\\n\\hline\n"

    latex += r"""\end{tabular}
}
\end{table}"""

    tex_path = os.path.join(output_dir, 'comparison_table.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print_success(f"LaTeX table saved: {tex_path}")


# =================================================================================
# Main
# =================================================================================

def main():
    print_header("DAPR-YOLO26 Training on EAGLE (2 classes: car, truck)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(RANDOM_SEED)
    check_gpu_or_exit()

    if not check_files():
        print_error("Required files are missing.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    hyperparams = save_hyperparameters(RESULTS_DIR)

    all_results = []

    for imgsz_config in IMAGE_SIZES_CONFIG:
        imgsz = imgsz_config['imgsz']
        print_header(f"Processing image size: {imgsz}", "-")

        print("\nArchitectures to be trained (in order):")
        for idx, arch in enumerate(ARCHITECTURES, 1):
            print(f"   {idx}. {arch['name']}")
            print(f"      YAML: {os.path.basename(arch['yaml'])}")
            print(f"      Exists: {os.path.exists(arch['yaml'])}")

        for arch in ARCHITECTURES:
            for seed in NEW_SEEDS:
                result = train_model_for_size(arch, imgsz_config, RESULTS_DIR, seed=seed)

                if result:
                    all_results.append(result)
                    project_path = result['project_path']
                    generate_confusion_matrix(project_path, RESULTS_DIR, arch['name'], imgsz)
                    generate_pr_curve(project_path, RESULTS_DIR, arch['name'], imgsz)
                    generate_sample_detections(
                        project_path,
                        os.path.join(DATASET_BASE, "data.yaml"),
                        RESULTS_DIR, arch['name'], imgsz
                    )
                    create_individual_report(result, RESULTS_DIR)

    if all_results:
        create_comparison_report(all_results, RESULTS_DIR)
        for imgsz in [640]:
            generate_per_class_ap_plot(all_results, RESULTS_DIR, imgsz)
        generate_latex_table(all_results, RESULTS_DIR)
        save_per_class_metrics_to_txt(all_results, RESULTS_DIR)

        df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, 'all_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print_success(f"Results saved: {csv_path}")

        hyperparams['completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hyperparams['total_models_trained'] = len(all_results)
        hyperparams['best_results'] = {
            'best_map50': max([r['test_map50'] for r in all_results]),
            'best_map': max([r['test_map'] for r in all_results]),
        }
        save_json(os.path.join(RESULTS_DIR, 'hyperparameters.json'), hyperparams)

    print_header("All training runs completed.")


if __name__ == '__main__':
    main()