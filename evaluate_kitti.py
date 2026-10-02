"""Held-out SemanticKITTI evaluation with the same mapping and planar-range buckets."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from config import CLASSES, IGNORE_LABEL, KITTI_DATASET_ROOT, NUM_CLASSES, RANGE_BUCKETS
from data.kitti import KittiDataError, KittiSequenceDataset
from data.semantic_kitti import map_to_prototype
from data.sources import load_kitti_model_artifact


def _summary(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(NUM_CLASSES), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(NUM_CLASSES), where=cm.sum(1) > 0)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(NUM_CLASSES), where=union > 0)
    per_class = {}
    for i, name in enumerate(CLASSES):
        support, predicted, class_union = int(cm[i].sum()), int(cm[:, i].sum()), int(union[i])
        per_class[name] = {"precision": float(precision[i]) if predicted else None,
                           "recall": float(recall[i]) if support else None,
                           "iou": float(iou[i]) if class_union else None, "support": support}
    return {"accuracy": float(tp.sum() / cm.sum()) if cm.sum() else None, "miou": float(iou[union > 0].mean()) if np.any(union > 0) else None,
            "per_class": per_class,
            "confusion_matrix": cm.tolist()}


def evaluate_kitti(dataset_root, sequence="00", model_path="artifacts/kitti/model/model.npz", max_frames=0,
                   use_artifact_test_split=True):
    model, metadata = load_kitti_model_artifact(model_path)
    ds = KittiSequenceDataset(dataset_root, sequence, require_labels=True)
    split_name = "all_frames"
    if use_artifact_test_split:
        entries = metadata.get("split", {}).get("partitions", {}).get("test", [])
        indices = [int(index) for seq, index in entries if str(seq).zfill(2) == ds.sequence]
        if not indices:
            raise KittiDataError(f"Model artifact has no held-out test frames for sequence {ds.sequence}; use --all-frames only for an explicit diagnostic run")
        split_name = "artifact_test"
    else:
        indices = list(range(len(ds)))
    if max_frames:
        indices = indices[:max_frames]
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    buckets = {pair: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for pair in RANGE_BUCKETS}
    times, evaluated, ignored = [], 0, 0
    for index in indices:
        frame = ds.get_frame(index, require_labels=True)
        truth = map_to_prototype(frame.semantic_labels); valid = truth != IGNORE_LABEL
        ignored += int((~valid).sum())
        t0 = time.perf_counter(); pred, _ = model.predict(frame.points); times.append(time.perf_counter() - t0)
        np.add.at(cm, (truth[valid], pred[valid]), 1); evaluated += int(valid.sum())
        ranges = np.hypot(frame.points[:, 0], frame.points[:, 1])
        for lo, hi in RANGE_BUCKETS:
            mask = valid & (ranges >= lo) & (ranges < hi)
            np.add.at(buckets[(lo, hi)], (truth[mask], pred[mask]), 1)
    return {"sequence": str(sequence).zfill(2), "split": split_name, "frames": len(indices), "points_evaluated": evaluated, "ignored_points": ignored,
            "overall": _summary(cm), "range_buckets": {f"{lo}-{hi}m": {"point_count": int(matrix.sum()), **_summary(matrix)} for (lo, hi), matrix in buckets.items()},
            "inference_latency_ms": {"mean": float(np.mean(times)*1000), "median": float(np.median(times)*1000), "p95": float(np.percentile(times,95)*1000)}}


def print_report(report):
    print("KITTI EVALUATION\n----------------\n" + f"Sequence: {report['sequence']}\nSplit: {report['split']}\nFrames: {report['frames']}\nPoints evaluated: {report['points_evaluated']}\nIgnored: {report['ignored_points']}")
    print(f"\nAccuracy: {report['overall']['accuracy']*100:.2f}%\nmIoU: {report['overall']['miou']*100:.2f}%\n\nClass                         IoU\n--------------------------------------------")
    for name, values in report["overall"]["per_class"].items():
        iou = "N/A" if values["iou"] is None else f"{values['iou'] * 100:6.2f}"
        print(f"{name:<29} {iou}")
    print("\nRange metrics:")
    for name, values in report["range_buckets"].items():
        miou = "N/A" if values["miou"] is None else f"{values['miou']*100:.2f}%"
        acc = "N/A" if values["accuracy"] is None else f"{values['accuracy'] * 100:6.2f}%"
        print(f"  {name:<9} points={values['point_count']:<8} acc={acc:<7} mIoU={miou}")
    latency = report["inference_latency_ms"]
    print(f"\nInference ms: mean={latency['mean']:.2f} median={latency['median']:.2f} p95={latency['p95']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=KITTI_DATASET_ROOT, required=not bool(KITTI_DATASET_ROOT)); parser.add_argument("--sequence", default="00")
    parser.add_argument("--model-path", default="artifacts/kitti/model/model.npz"); parser.add_argument("--max-frames", type=int, default=0); parser.add_argument("--output")
    parser.add_argument("--all-frames", action="store_true", help="Diagnostic only: bypass the saved held-out test split.")
    args = parser.parse_args()
    try:
        report = evaluate_kitti(args.dataset_root, args.sequence, args.model_path, args.max_frames,
                                use_artifact_test_split=not args.all_frames); print_report(report)
        if args.output: Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    except KittiDataError as exc:
        print(f"EVALUATION FAILED: {exc}"); raise SystemExit(2)


if __name__ == "__main__": main()
