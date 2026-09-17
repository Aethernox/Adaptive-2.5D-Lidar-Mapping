"""
Train the point-wise segmentation MLP on synthetic frames.

Stands in for "train on SemanticKITTI (+ nuScenes-lidarseg fine-tune)" from
the reference architecture: here the "dataset" is a batch of simulated
sweeps (with perfect ground-truth labels from the world generator) sampled
across many (t, ego-pose) combinations so the model sees terrain/objects at
a wide range of distances.

Run: python -m perception.train
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RNG_SEED
from sim.lidar import LidarSimulator
from perception.features import extract_features
from perception.model import PointSegModel

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.npz")


def build_dataset(n_frames=60, t_start=0.0, t_step=1.5, seed=RNG_SEED):
    sim = LidarSimulator(seed=seed)
    feats_list, labels_list = [], []
    for i in range(n_frames):
        t = t_start + i * t_step
        frame = sim.sweep(t)
        feats_list.append(extract_features(frame["points"]))
        labels_list.append(frame["true_labels"])
    return np.concatenate(feats_list), np.concatenate(labels_list)


def train(epochs=25, batch_size=512, lr=0.08, verbose=True):
    t0 = time.time()
    X, y = build_dataset(n_frames=60, t_start=0.0, t_step=1.5, seed=RNG_SEED)
    Xte, yte = build_dataset(n_frames=15, t_start=500.0, t_step=1.7, seed=RNG_SEED + 1)
    if verbose:
        print(f"Train points: {len(X)}  Test points: {len(Xte)}  "
              f"(data gen {time.time()-t0:.1f}s)")

    model = PointSegModel(seed=0)
    rng = np.random.RandomState(1)
    n = len(X)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        losses, accs = [], []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss, acc = model.train_step(X[idx], y[idx], lr=lr)
            losses.append(loss); accs.append(acc)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            test_probs = model.forward_probs(Xte)
            test_acc = (test_probs.argmax(axis=1) == yte).mean()
            print(f"epoch {epoch:2d}  loss={np.mean(losses):.4f}  "
                  f"train_acc={np.mean(accs):.3f}  test_acc={test_acc:.3f}")

    model.save(WEIGHTS_PATH)
    if verbose:
        print(f"Saved weights to {WEIGHTS_PATH}  (total time {time.time()-t0:.1f}s)")
    return model


if __name__ == "__main__":
    train()
