"""
Range-Bucketed Semantic & Mapping Evaluation Engine
Matches Section 16: Reporting metrics separated by range tiers (0-10m, 10-25m, 25-50m, 50-100m).
"""

from typing import Dict, List, Tuple
import numpy as np


class RangeBucketedEvaluator:
    """
    Evaluates semantic segmentation and 2.5D elevation accuracy across distance tiers.
    """

    def __init__(
        self,
        num_classes: int = 20,
        range_bins: List[Tuple[float, float, str]] = [
            (0.0, 10.0, "0-10m (Tier 0)"),
            (10.0, 25.0, "10-25m (Tier 1)"),
            (25.0, 50.0, "25-50m (Tier 2)"),
            (50.0, 100.0, "50-100m (Tier 3)")
        ]
    ):
        self.num_classes = num_classes
        self.range_bins = range_bins
        self.reset()

    def reset(self):
        # Confusion matrices per range tier: (num_classes, num_classes)
        self.confusion_matrices = {
            name: np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            for _, _, name in self.range_bins
        }
        self.global_confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_points = 0

    def update(
        self,
        points: np.ndarray,
        predicted_labels: np.ndarray,
        ground_truth_labels: np.ndarray
    ):
        """
        Accumulate predictions and ground truths into range-bucketed confusion matrices.
        """
        if points is None or len(points) == 0 or ground_truth_labels is None:
            return

        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        preds = predicted_labels
        gts = ground_truth_labels
        self.total_points += len(points)

        for r_min, r_max, name in self.range_bins:
            mask = (r >= r_min) & (r < r_max)
            if not np.any(mask):
                continue
                
            p_sub = preds[mask]
            g_sub = gts[mask]
            
            # Filter ignore index 0 for confusion matrix computation
            valid = (g_sub > 0) & (g_sub < self.num_classes) & (p_sub < self.num_classes)
            if np.any(valid):
                p_v = p_sub[valid]
                g_v = g_sub[valid]
                
                indices = g_v * self.num_classes + p_v
                cm_flat = np.bincount(indices, minlength=self.num_classes**2)
                cm = cm_flat.reshape((self.num_classes, self.num_classes))
                
                self.confusion_matrices[name] += cm
                self.global_confusion += cm

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mIoU, precision, recall, and point accuracy per range bucket.
        """
        results = {}

        for _, _, name in self.range_bins:
            cm = self.confusion_matrices[name]
            tp = np.diag(cm)
            fp = np.sum(cm, axis=0) - tp
            fn = np.sum(cm, axis=1) - tp
            
            denominator = tp + fp + fn
            valid_classes = denominator > 0
            
            iou = np.zeros(self.num_classes, dtype=np.float64)
            iou[valid_classes] = tp[valid_classes] / denominator[valid_classes]
            
            miou = float(np.mean(iou[valid_classes])) if np.any(valid_classes) else 0.0
            total_corr = float(np.sum(tp))
            total_pts = float(np.sum(cm))
            acc = (total_corr / max(total_pts, 1.0)) * 100.0

            results[name] = {
                "mIoU": float(miou * 100.0),
                "point_accuracy_pct": float(acc),
                "evaluated_points": int(total_pts)
            }

        # Global metrics
        cm_g = self.global_confusion
        tp_g = np.diag(cm_g)
        denom_g = np.sum(cm_g, axis=0) + np.sum(cm_g, axis=1) - tp_g
        valid_g = denom_g > 0
        iou_g = np.zeros(self.num_classes, dtype=np.float64)
        iou_g[valid_g] = tp_g[valid_g] / denom_g[valid_g]
        
        results["Overall"] = {
            "mIoU": float(np.mean(iou_g[valid_g]) * 100.0) if np.any(valid_g) else 0.0,
            "point_accuracy_pct": float((np.sum(tp_g) / max(np.sum(cm_g), 1.0)) * 100.0),
            "evaluated_points": int(np.sum(cm_g))
        }

        return results
