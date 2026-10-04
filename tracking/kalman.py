"""
3D Constant-Velocity Kalman Filter for LiDAR Object Tracking
State vector: [x, y, z, vx, vy, vz, length, width, height, heading]
"""

import numpy as np


class KalmanBoxTracker:
    """
    Kalman filter tracking a 3D bounding box with constant velocity motion model.
    """
    count = 0

    def __init__(self, bbox_3d: np.ndarray, class_id: int, dt: float = 0.10):
        """
        bbox_3d: [x, y, z, length, width, height, heading]
        """
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_id = class_id
        self.dt = dt

        # State: [x, y, z, vx, vy, vz, l, w, h, heading] (10D)
        self.x = np.zeros((10, 1), dtype=np.float64)
        self.x[0:3, 0] = bbox_3d[0:3] # x, y, z
        self.x[6:9, 0] = bbox_3d[3:6] # l, w, h
        self.x[9, 0] = bbox_3d[6]     # heading

        # State transition matrix F
        self.F = np.eye(10, dtype=np.float64)
        self.F[0, 3] = dt # x += vx * dt
        self.F[1, 4] = dt # y += vy * dt
        self.F[2, 5] = dt # z += vz * dt

        # Measurement matrix H: measures [x, y, z, l, w, h, heading] (7D)
        self.H = np.zeros((7, 10), dtype=np.float64)
        self.H[0:3, 0:3] = np.eye(3)
        self.H[3:6, 6:9] = np.eye(3)
        self.H[6, 9] = 1.0

        # State covariance P
        self.P = np.eye(10, dtype=np.float64) * 10.0
        self.P[3:6, 3:6] *= 100.0 # High uncertainty on initial velocity

        # Process noise covariance Q
        self.Q = np.eye(10, dtype=np.float64) * 0.1
        self.Q[3:6, 3:6] *= 1.0

        # Measurement noise covariance R
        self.R = np.eye(7, dtype=np.float64) * 0.2
        self.R[3:6, 3:6] *= 0.5
        self.R[6, 6] *= 0.1

        self.time_since_update = 0
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def predict(self) -> np.ndarray:
        """Advance the state vector and return predicted 3D bounding box."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.get_state()

    def update(self, bbox_3d: np.ndarray):
        """Update the state with observed measurement [x, y, z, l, w, h, heading]."""
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1

        z = np.zeros((7, 1), dtype=np.float64)
        z[:, 0] = bbox_3d

        # Innovation
        y = z - self.H @ self.x
        # Normalize heading innovation to [-pi, pi]
        y[6, 0] = (y[6, 0] + np.pi) % (2 * np.pi) - np.pi

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(10) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Returns [x, y, z, l, w, h, heading]."""
        return np.array([
            self.x[0, 0], self.x[1, 0], self.x[2, 0],
            self.x[6, 0], self.x[7, 0], self.x[8, 0],
            self.x[9, 0]
        ], dtype=np.float32)

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Returns (vx, vy, vz) in m/s."""
        return float(self.x[3, 0]), float(self.x[4, 0]), float(self.x[5, 0])

    @property
    def speed(self) -> float:
        """2D planar speed in m/s."""
        return float(np.sqrt(self.x[3, 0]**2 + self.x[4, 0]**2))
