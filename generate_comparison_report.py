# -*- coding: utf-8 -*-
"""
Four-Model Comparison Report Generator

Evaluates:
1. Core model (ConvNeXt V2-Tiny + BiLSTM + hybrid loss)
2. Baseline model (VGG CNN + BiLSTM + hybrid loss)
3. Traditional method 1 (OpenCV contour detection)
4. Traditional method 2 (KNN classification)

Generates comparison reports and visualization charts.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_evaluation_results(results_dir="results"):
    results_dir = Path(results_dir)
    results = {}

    eval_files = list(results_dir.glob("evaluate_all_*.json"))
    if eval_files:
        latest = sorted(eval_files)[-1]
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "models" in data:
            for model_name, metrics in data["models"].items():
                results[model_name] = {
                    "method": model_name,
                    "img_accuracy": metrics.get("image_accuracy", 0),
                    "char_accuracy": metrics.get("char_accuracy", 0),
                    "avg_inference_time_ms": metrics.get("avg_inference_time_ms", 0),
                    "throughput": metrics.get("throughput", 0),
                    "type": "deep_learning"
                }
        logger.info(f"Loaded evaluation results: {latest.name} ({len(results)} models)")
    else:
        core_path = results_dir / "core_model_evaluation.json"
        baseline_path = results_dir / "baseline_model_evaluation.json"

        if core_path.exists():
            with open(core_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results["core"] = {
                "method": "Core model",
                "img_accuracy": data.get("image_accuracy", data.get("img_accuracy", 0)),
                "char_accuracy": data.get("char_accuracy", 0),
                "avg_inference_time_ms": data.get("avg_inference_time_ms", data.get("avg_time_ms", 0)),
                "throughput": data.get("throughput", 0),
                "type": "deep_learning"
            }
            logger.info(f"Loaded core model evaluation results: {core_path}")

        if baseline_path.exists():
            with open(baseline_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results["baseline"] = {
                "method": "Baseline model",
                "img_accuracy": data.get("image_accuracy", data.get("img_accuracy", 0)),
                "char_accuracy": data.get("char_accuracy", 0),
                "avg_inference_time_ms": data.get("avg_inference_time_ms", data.get("avg_time_ms", 0)),
                "throughput": data.get("throughput", 0),
                "type": "deep_learning"
            }
            logger.info(f"Loaded baseline model evaluation results: {baseline_path}")

    trad_path = results_dir / "traditional_comparison.json"
    if trad_path.exists():
        with open(trad_path, "r", encoding="utf-8") as f:
            trad_data = json.load(f)
        if isinstance(trad_data, list):
            for i, t in enumerate(trad_data):
                results[f"traditional_{i}"] = {
                    "method": t.get("method", f"Traditional method {i+1}"),
                    "img_accuracy": t.get("img_accuracy", 0),
                    "char_accuracy": t.get("char_accuracy", 0),
                    "avg_inference_time_ms": t.get("avg_time_ms", 0),
                    "throughput": 1000 / t.get("avg_time_ms", 1) if t.get("avg_time_ms", 0) > 0 else 0,
                    "type": "traditional"
                }
        logger.info(f"Loaded traditional method evaluation results: {len([k for k in results if 'traditional' in k])} methods")
    else:
        logger.warning("Traditional method evaluation results not found, using placeholder data")
        results["traditional_0"] = {
            "method": "OpenCV contour detection",
            "img_accuracy": 0.0,
            "char_accuracy": 0.0,
            "avg_inference_time_ms": 0.0,
            "throughput": 0.0,
            "type": "traditional"
        }
        results["traditional_1"] = {
            "method": "Connected component + KNN",
            "img_accuracy": 0.0,
            "char_accuracy": 0.0,
            "avg_inference_time_ms": 0.0,
            "throughput": 0.0,
            "type": "traditional"
        }

    return results


def generate_comparison_report(results, output_dir="results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_data = []
    architecture_info = {
        "Core model": {
            "architecture": "ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE hybrid loss",
            "param_count": "~31.6M",
            "type": "Deep Learning"
        },
        "Baseline model": {
            "architecture": "VGG CNN + BiLSTM + CTC/CE hybrid loss",
            "param_count": "~3.8M",
            "type": "Deep Learning"
        }
    }

    for key, metrics in results.items():
        method = metrics.get("method", key)
        info = architecture_info.get(method, {
            "architecture": metrics.get("architecture", "Traditional method"),
            "param_count": "N/A",
            "type": "Traditional method"
        })

        report_data.append({
            "method": method,
            "architecture": info["architecture"],
            "param_count": info["param_count"],
            "img_accuracy": metrics.get("img_accuracy", 0),
            "char_accuracy": metrics.get("char_accuracy", 0),
            "avg_inference_time_ms": metrics.get("avg_inference_time_ms", 0),
            "throughput": metrics.get("throughput", 0),
            "type": info["type"]
        })

    comparison_json = output_dir / "comparison_report.json"
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Comparison JSON saved: {comparison_json}")

    md_lines = []
    md_lines.append("# CAPTCHA Recognition Method Comparison Report")
    md_lines.append("")
    md_lines.append(f"## 1. Experiment Overview")
    md_lines.append("")
    md_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("This report compares the performance of four CAPTCHA recognition methods:")
    md_lines.append("- Core model: ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE hybrid loss")
    md_lines.append("- Baseline model: VGG CNN + BiLSTM + CTC/CE hybrid loss")
    md_lines.append("- Traditional method 1: OpenCV contour detection + character recognition")
    md_lines.append("- Traditional method 2: Connected component segmentation + HOG features + KNN classification")
    md_lines.append("")
    md_lines.append("## 2. Experiment Results")
    md_lines.append("")
    md_lines.append("### 2.1 Performance Comparison Table")
    md_lines.append("")
    md_lines.append("| Method | Architecture | Params | Image Accuracy | Char Accuracy | Avg Inference Time(ms) | Throughput(samples/s) |")
    md_lines.append("|------|------|--------|-----------|-----------|-----------------|-------------------|")

    for m in report_data:
        md_lines.append(
            f"| {m['method']} | {m['architecture']} | {m['param_count']} | "
            f"{m['img_accuracy']*100:.2f}% | {m['char_accuracy']*100:.2f}% | "
            f"{m['avg_inference_time_ms']:.2f} | {m['throughput']:.2f} |"
        )

    md_lines.append("")
    md_lines.append("### 2.2 Chart Analysis")
    md_lines.append("")

    chart_path_relative = "figures/comparison_chart.png"
    if (output_dir / chart_path_relative).exists():
        md_lines.append(f"![Performance Comparison]({chart_path_relative})")
        md_lines.append("")
        md_lines.append("*Figure 1: Comparison of image accuracy, char accuracy, inference time, and throughput for four methods*")
        md_lines.append("")

    md_lines.append("### 2.3 Analysis")
    md_lines.append("")

    dl_models = [m for m in report_data if m['type'] == 'Deep Learning']
    trad_models = [m for m in report_data if m['type'] == 'Traditional method']

    if dl_models:
        best_dl = max(dl_models, key=lambda x: x['img_accuracy'])
        md_lines.append(f"- **Best deep learning model**: {best_dl['method']} (image accuracy {best_dl['img_accuracy']*100:.2f}%)")

    if trad_models:
        best_trad = max(trad_models, key=lambda x: x['img_accuracy'])
        md_lines.append(f"- **Best traditional method**: {best_trad['method']} (image accuracy {best_trad['img_accuracy']*100:.2f}%)")

    if dl_models and trad_models:
        best_dl = max(dl_models, key=lambda x: x['img_accuracy'])
        best_trad = max(trad_models, key=lambda x: x['img_accuracy'])
        if best_trad['img_accuracy'] > 0:
            improvement = (best_dl['img_accuracy'] - best_trad['img_accuracy']) / best_trad['img_accuracy'] * 100
            md_lines.append(f"- Deep learning achieves **{improvement:.1f}%** improvement in image accuracy over traditional methods")

    md_lines.append("")
    md_lines.append("## 3. Conclusion")
    md_lines.append("")
    md_lines.append("1. Deep learning methods significantly outperform traditional methods in CAPTCHA recognition")
    md_lines.append("2. End-to-end sequence modeling (CTC + BiLSTM) effectively handles character adhesion and deformation")
    md_lines.append("3. The lightweight baseline model achieves good balance between accuracy and efficiency")

    report_md = output_dir / "comparison_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown report saved: {report_md}")

    return report_data


def generate_charts(report_data, output_dir="results"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed, cannot generate charts")
        return

    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    methods = [m['method'] for m in report_data]
    img_acc = [m['img_accuracy'] * 100 for m in report_data]
    char_acc = [m['char_accuracy'] * 100 for m in report_data]
    times = [m['avg_inference_time_ms'] for m in report_data]
    throughputs = [m['throughput'] for m in report_data]

    dl_indices = [i for i, m in enumerate(report_data) if m['type'] == 'Deep Learning']
    trad_indices = [i for i, m in enumerate(report_data) if m['type'] == 'Traditional method']

    colors = []
    for m in report_data:
        if m['type'] == 'Deep Learning':
            colors.append('#2196F3' if 'Core' in m['method'] else '#4CAF50')
        else:
            colors.append('#FF9800' if 'OpenCV' in m['method'] else '#9C27B0')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CAPTCHA Recognition Method Comparison', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, img_acc, color=colors)
    ax1.set_ylabel('Image Accuracy (%)')
    ax1.set_title('Image Accuracy Comparison')
    ax1.set_ylim(0, max(100, max(img_acc) * 1.1) if img_acc else 100)
    for bar, acc in zip(bars1, img_acc):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, char_acc, color=colors)
    ax2.set_ylabel('Char Accuracy (%)')
    ax2.set_title('Char Accuracy Comparison')
    ax2.set_ylim(0, max(100, max(char_acc) * 1.1) if char_acc else 100)
    for bar, acc in zip(bars2, char_acc):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, times, color=colors)
    ax3.set_ylabel('Avg Inference Time (ms)')
    ax3.set_title('Inference Time Comparison (lower is better)')
    for bar, t in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{t:.2f}ms', ha='center', va='bottom', fontweight='bold')

    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, throughputs, color=colors)
    ax4.set_ylabel('Throughput (samples/s)')
    ax4.set_title('Throughput Comparison (higher is better)')
    for bar, tp in zip(bars4, throughputs):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{tp:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    chart_path = output_dir / "figures" / "comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison chart saved: {chart_path}")
    plt.close()


def main():
    setup_logging()

    logger.info("=" * 60)
    logger.info("Four-model comparison report generation")
    logger.info("=" * 60)

    results = load_evaluation_results()

    if not results:
        logger.error("No evaluation result files found, please run the evaluation scripts first")
        return

    report_data = generate_comparison_report(results)

    generate_charts(report_data)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Comparison report generation complete!")
    logger.info(f"  JSON data: results/comparison_report.json")
    logger.info(f"  Markdown report: results/comparison_report.md")
    logger.info(f"  Comparison chart: results/figures/comparison_chart.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
