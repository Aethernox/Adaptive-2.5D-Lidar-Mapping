"""
Point-wise semantic segmentation model.

Architecture note: the reference design calls for a PointPillars/Cylinder3D-
style sparse-conv backbone (PointNet++ is explicitly rejected in the
reference doc for not scaling to full sweeps). That is the right choice for
a production system, but it needs GPU deep-learning frameworks (torch +
spconv/torchsparse) which are not available in this sandbox (no network
access to install packages). To keep the *interface* identical to what a
real backbone would expose (per-point class logits from per-point/pillar
features) while staying runnable anywhere with just numpy, this prototype
implements a small fully-connected network (5 -> 64 -> 32 -> 6) trained
with plain minibatch gradient descent / manual backprop.

Swapping this for a real torch PointPillars/Cylinder3D model later only
requires preserving `PointSegModel.predict(points) -> (class_ids, confidence)`
— nothing else in the pipeline needs to change (see SOFTWARE_ARCHITECTURE.md).
"""
import numpy as np
from config import NUM_CLASSES
from perception.features import extract_features, N_FEATURES

HIDDEN1, HIDDEN2 = 64, 32


def _he_init(fan_in, fan_out, rng):
    return rng.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(2.0 / fan_in)


class PointSegModel:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = _he_init(N_FEATURES, HIDDEN1, rng); self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        self.W2 = _he_init(HIDDEN1, HIDDEN2, rng); self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        self.W3 = _he_init(HIDDEN2, NUM_CLASSES, rng); self.b3 = np.zeros(NUM_CLASSES, dtype=np.float32)

    # ---- forward ----
    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        logits = a2 @ self.W3 + self.b3
        return dict(X=X, z1=z1, a1=a1, z2=z2, a2=a2, logits=logits)

    @staticmethod
    def _softmax(logits):
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return e / e.sum(axis=1, keepdims=True)

    def forward_probs(self, feats):
        return self._softmax(self._forward(feats)["logits"])

    def predict(self, points):
        """points: (N,4) raw x,y,z,intensity in ego frame.
        Returns (class_ids (N,) int, confidence (N,) float in [0,1])."""
        if len(points) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        feats = extract_features(points)
        probs = self.forward_probs(feats)
        cls = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        return cls, conf

    # ---- training (manual backprop, cross-entropy loss) ----
    def train_step(self, feats, labels, lr=0.05, l2=1e-4):
        N = feats.shape[0]
        cache = self._forward(feats)
        probs = self._softmax(cache["logits"])
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(N), labels] = 1.0

        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

        dlogits = (probs - y_onehot) / N
        dW3 = cache["a2"].T @ dlogits + l2 * self.W3
        db3 = dlogits.sum(axis=0)
        da2 = dlogits @ self.W3.T
        dz2 = da2 * (cache["z2"] > 0)
        dW2 = cache["a1"].T @ dz2 + l2 * self.W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (cache["z1"] > 0)
        dW1 = cache["X"].T @ dz1 + l2 * self.W1
        db1 = dz1.sum(axis=0)

        for p, dp in [(self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2),
                      (self.W3, dW3), (self.b3, db3)]:
            p -= lr * dp
        acc = (probs.argmax(axis=1) == labels).mean()
        return loss, acc

    # ---- persistence ----
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls.__new__(cls)
        m.W1, m.b1, m.W2, m.b2, m.W3, m.b3 = d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"]
        return m
