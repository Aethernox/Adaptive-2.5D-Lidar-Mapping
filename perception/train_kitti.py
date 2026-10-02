"""Streaming, class-weighted SemanticKITTI training for the NumPy point MLP."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

from config import (CLASS_MAPPING_VERSION, CLASSES, FEATURE_NORMALIZATION_VERSION, IGNORE_LABEL,
                    KITTI_DATASET_ROOT, NUM_CLASSES, RNG_SEED, SEMANTICKITTI_MAPPING_DOCUMENTATION)
from data.kitti import KittiDataError, KittiSequenceDataset
from data.semantic_kitti import map_to_prototype
from data.splits import deterministic_frame_split, save_split
from data.validation import validate_dataset
from perception.features import N_FEATURES, extract_features, feature_configuration
from perception.model import PointSegModel


def _parse_sequences(value: str) -> list[str]:
    result = [v.strip().zfill(2) for v in value.split(",") if v.strip()]
    if not result:
        raise ValueError("At least one sequence is required")
    return result


def _partitions(datasets: dict[str, KittiSequenceDataset], max_frames: int) -> dict[str, list[tuple[str, int]]]:
    sequences = list(datasets)
    if len(sequences) == 1:
        ds = datasets[sequences[0]]
        n = min(len(ds), max_frames) if max_frames else len(ds)
        split = deterministic_frame_split(n)
        return {name: [(sequences[0], i) for i in indices] for name, indices in split.items()}
    # Sequence partitions avoid spatial/temporal leakage when sufficient data exists.
    output = {"train": [], "validation": [], "test": []}
    for pos, sequence in enumerate(sequences):
        target = "test" if pos == len(sequences) - 1 else "validation" if pos == len(sequences) - 2 else "train"
        n = min(len(datasets[sequence]), max_frames) if max_frames else len(datasets[sequence])
        output[target].extend((sequence, i) for i in range(n))
    if not output["train"]:
        raise ValueError("Need at least three sequences, or one sequence with at least three frames")
    return output


def _sample_valid(frame, points_per_frame: int, rng: np.random.RandomState):
    labels = map_to_prototype(frame.semantic_labels)
    valid = np.flatnonzero(labels != IGNORE_LABEL)
    if points_per_frame > 0 and valid.size > points_per_frame:
        valid = rng.choice(valid, size=points_per_frame, replace=False)
    return frame.points[valid], labels[valid].astype(np.int64)


def _class_weights(datasets, entries, points_per_frame, seed):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    rng = np.random.RandomState(seed)
    for seq, index in entries:
        _, labels = _sample_valid(datasets[seq].get_frame(index, require_labels=True), points_per_frame, rng)
        counts += np.bincount(labels, minlength=NUM_CLASSES)
    # Inverse-frequency weights normalized over present classes. Missing class
    # receives zero weight and is reported; it cannot be learned from this split.
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return counts, weights


def _metrics(model, datasets, entries, points_per_frame, batch_size, seed):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    losses, times, n_points = [], [], 0
    rng = np.random.RandomState(seed)
    for seq, index in entries:
        pts, labels = _sample_valid(datasets[seq].get_frame(index, require_labels=True), points_per_frame, rng)
        if not len(labels):
            continue
        feats = extract_features(pts)
        for start in range(0, len(labels), batch_size):
            x, y = feats[start:start + batch_size], labels[start:start + batch_size]
            t0 = time.perf_counter(); probs = model.forward_probs(x); times.append(time.perf_counter() - t0)
            pred = probs.argmax(axis=1)
            losses.append(float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))))
            np.add.at(cm, (y, pred), 1); n_points += len(y)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(NUM_CLASSES), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(NUM_CLASSES), where=cm.sum(axis=1) > 0)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    iou = np.divide(tp, union, out=np.zeros(NUM_CLASSES), where=union > 0)
    present = union > 0
    per_class = {}
    for i, name in enumerate(CLASSES):
        support, predicted = int(cm[i].sum()), int(cm[:, i].sum())
        per_class[name] = {"precision": float(precision[i]) if predicted else None,
                           "recall": float(recall[i]) if support else None,
                           "iou": float(iou[i]) if union[i] else None, "support": support}
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "overall_accuracy": float(tp.sum() / max(n_points, 1)),
        "macro_accuracy": float(np.mean(recall[cm.sum(axis=1) > 0])) if np.any(cm.sum(axis=1) > 0) else None,
        "mean_iou": float(np.mean(iou[present])) if np.any(present) else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(), "points": int(n_points),
        "inference_latency_ms": {"mean": float(np.mean(times) * 1000) if times else None,
                                 "median": float(np.median(times) * 1000) if times else None,
                                 "p95": float(np.percentile(times, 95) * 1000) if times else None},
    }


def train_kitti(dataset_root, sequences=("00",), epochs=20, batch_size=512, learning_rate=0.05,
                max_frames=0, points_per_frame=20000, seed=RNG_SEED,
                checkpoint="artifacts/kitti/model/model.npz") -> tuple[PointSegModel, dict]:
    sequences = [str(s).zfill(2) for s in sequences]
    # The explicit validation pass is intentionally before model construction.
    for sequence in sequences:
        validate_dataset(dataset_root, sequence, max_frames=max_frames)
    datasets = {sequence: KittiSequenceDataset(dataset_root, sequence, require_labels=True) for sequence in sequences}
    parts = _partitions(datasets, max_frames)
    counts, weights = _class_weights(datasets, parts["train"], points_per_frame, seed)
    if not counts.sum():
        raise KittiDataError("Training split has no mapped SemanticKITTI points")
    model, rng = PointSegModel(seed=seed), np.random.RandomState(seed)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(parts["train"]); rng.shuffle(order)
        losses = []
        for seq, index in order:
            points, labels = _sample_valid(datasets[seq].get_frame(index, require_labels=True), points_per_frame, rng)
            if not len(labels):
                continue
            feats = extract_features(points)
            perm = rng.permutation(len(labels))
            for start in range(0, len(labels), batch_size):
                batch = perm[start:start + batch_size]
                loss, _ = model.train_step(feats[batch], labels[batch], lr=learning_rate, class_weights=weights)
                losses.append(loss)
        validation = _metrics(model, datasets, parts["validation"], points_per_frame, batch_size, seed + epoch)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "validation": validation}
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} val_acc={validation['overall_accuracy']:.4f} val_mIoU={validation['mean_iou']:.4f}")
    test = _metrics(model, datasets, parts["test"], points_per_frame, batch_size, seed + 999)
    checkpoint = Path(checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True); model.save(checkpoint)
    split_meta = {"sequences": sequences, "max_frames": max_frames, "partitions": {k: [[s, i] for s, i in v] for k, v in parts.items()}}
    save_split(checkpoint.parent.parent / "splits" / "splits.json", split_meta)
    artifact = {
        "artifact_version": 1, "model_path": str(checkpoint), "num_classes": NUM_CLASSES, "n_features": N_FEATURES,
        "feature_normalization_version": FEATURE_NORMALIZATION_VERSION, "feature_configuration": feature_configuration(),
        "class_mapping_version": CLASS_MAPPING_VERSION, "class_mapping": SEMANTICKITTI_MAPPING_DOCUMENTATION,
        "class_names": CLASSES, "class_counts": counts.tolist(),
        "class_weights": weights.tolist(), "training": {"epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "points_per_frame": points_per_frame, "seed": seed, "elapsed_s": time.perf_counter() - started},
        "split": split_meta, "history": history, "test": test,
    }
    (checkpoint.parent / "metadata.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (checkpoint.parent.parent / "metrics").mkdir(parents=True, exist_ok=True)
    (checkpoint.parent.parent / "metrics" / "metrics.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return model, artifact


def main():
    parser = argparse.ArgumentParser(description="Train the real-data SemanticKITTI NumPy segmentation model.")
    parser.add_argument("--dataset-root", default=KITTI_DATASET_ROOT, required=not bool(KITTI_DATASET_ROOT))
    parser.add_argument("--sequences", default="00")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.05); parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--points-per-frame", type=int, default=20000); parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--checkpoint", default="artifacts/kitti/model/model.npz"); parser.add_argument("--device", default="cpu",
                        help="NumPy backend is CPU-only; retained for CLI compatibility.")
    args = parser.parse_args()
    if args.device.lower() not in {"cpu", "auto"}:
        print("Note: the verified NumPy backend runs on CPU; --device is accepted but GPU is not required.")
    try:
        _, artifact = train_kitti(args.dataset_root, _parse_sequences(args.sequences), args.epochs, args.batch_size,
                                  args.learning_rate, args.max_frames, args.points_per_frame, args.seed, args.checkpoint)
        print(f"Saved KITTI artifact: {artifact['model_path']}")
    except (KittiDataError, ValueError) as exc:
        print(f"TRAINING FAILED: {exc}"); raise SystemExit(2)


if __name__ == "__main__":
    main()
