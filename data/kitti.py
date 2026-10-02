"""Frame-wise KITTI Odometry/SemanticKITTI reader; no full-sequence caching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile
import numpy as np

from data.semantic_kitti import decode_instance_labels, decode_semantic_labels


class KittiDataError(RuntimeError):
    """A clear boundary error for invalid KITTI layout or binary content."""


def discover_dataset_root(root: str | Path) -> Path:
    """Resolve canonical KITTI layouts and a directly supplied flat frame directory.

    KITTI odometry exports are sometimes supplied as ``000000.bin`` files in
    one directory.  That layout is valid for *label-free replay* but not a
    substitute for SemanticKITTI labels; callers that require labels remain
    strict below.
    """
    p = Path(root).expanduser().resolve()
    if not p.exists():
        raise KittiDataError(f"KITTI dataset root does not exist: {p}")
    if (p / "sequences").is_dir():
        return p
    # A sequence directory may itself be passed as --dataset-root.
    if (p / "velodyne").is_dir():
        return p.parent
    if any((p / s / "velodyne").is_dir() for s in ("00", "01", "02")):
        return p
    if any(p.glob("*.bin")):
        return p
    if any(p.glob("*labels*.zip")) and any(candidate.is_dir() for candidate in p.rglob("velodyne")):
        return p
    raise KittiDataError(
        f"No KITTI sequence layout found under {p}. Expected sequences/00/velodyne or 00/velodyne."
    )


def _sequence_dir(root: Path, sequence: str) -> Path:
    sequence = str(sequence).zfill(2)
    candidates = (root / "sequences" / sequence, root / sequence)
    for candidate in candidates:
        if (candidate / "velodyne").is_dir():
            return candidate
    raise KittiDataError(f"Sequence {sequence} has no velodyne directory beneath {root}")


def _archive_velodyne_dirs(root: Path) -> list[Path]:
    return [path for path in root.rglob("velodyne") if path.is_dir() and any(path.glob("*.bin"))]


def load_velodyne_bin(path: str | Path) -> np.ndarray:
    path = Path(path)
    n_bytes = path.stat().st_size
    if n_bytes % (4 * np.dtype(np.float32).itemsize):
        raise KittiDataError(f"Corrupt .bin file {path}: {n_bytes} bytes is not divisible by 16")
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise KittiDataError(f"Corrupt .bin file {path}: float count {values.size} is not divisible by 4")
    return values.reshape((-1, 4))


def load_packed_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    n_bytes = path.stat().st_size
    if n_bytes % np.dtype(np.uint32).itemsize:
        raise KittiDataError(f"Corrupt .label file {path}: {n_bytes} bytes is not divisible by 4")
    return np.fromfile(path, dtype=np.uint32)


def _parse_calibration_text(text: str) -> dict[str, np.ndarray]:
    """Parse KITTI colon-delimited calibration text without assuming fields."""
    out: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        try:
            out[key.strip()] = np.fromstring(values, sep=" ", dtype=np.float64)
        except ValueError as exc:
            raise KittiDataError(f"Invalid calibration line: {line!r}") from exc
    return out


def parse_calibration(path: str | Path) -> dict[str, np.ndarray]:
    """Parse KITTI colon-delimited calibration values without assuming fields."""
    path = Path(path)
    if not path.exists():
        return {}
    return _parse_calibration_text(path.read_text(encoding="utf-8"))


def _parse_poses_text(text: str, source: str) -> np.ndarray | None:
    rows = [np.fromstring(line, sep=" ", dtype=np.float64) for line in text.splitlines() if line.strip()]
    if any(row.size != 12 for row in rows):
        raise KittiDataError(f"Invalid pose file {source}: expected 12 values per line")
    return np.asarray(rows, dtype=np.float64).reshape((-1, 3, 4))


def load_poses(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    return _parse_poses_text(path.read_text(), str(path))


@dataclass(frozen=True)
class KittiFrame:
    sequence: str
    frame_idx: int
    points: np.ndarray
    intensity: np.ndarray
    semantic_labels: np.ndarray | None
    instance_labels: np.ndarray | None
    calibration: dict[str, np.ndarray]
    pose: np.ndarray | None


class KittiSequenceDataset:
    """Index-only sequence dataset. Frame contents are loaded on demand."""
    def __init__(self, dataset_root: str | Path, sequence: str = "00", require_labels: bool = False):
        self.root = discover_dataset_root(dataset_root)
        self.sequence = str(sequence).zfill(2)
        # A flat export has no sequence container.  Treat the passed directory
        # as the requested sequence while retaining the same public interface.
        self.flat_layout = any(self.root.glob("*.bin"))
        self.archive_layout = False
        if self.flat_layout:
            self.sequence_dir, self.velodyne_dir = self.root, self.root
        elif any(self.root.glob("*labels*.zip")):
            candidates = _archive_velodyne_dirs(self.root)
            if not candidates:
                raise KittiDataError(f"No Velodyne directory found under archive dataset root {self.root}")
            selected = []
            for candidate in candidates:
                if candidate.parent.name.isdigit():
                    candidate_sequence = candidate.parent.name.zfill(2)
                elif candidate.parent.name == "data_odometry_velodyne":
                    candidate_sequence = "00"
                else:
                    candidate_sequence = None
                if candidate_sequence is None or candidate_sequence == self.sequence:
                    selected.append(candidate)
            if not selected:
                raise KittiDataError(f"Sequence {self.sequence} has no Velodyne frames beneath archive dataset root {self.root}")
            self.archive_layout = True
            self.sequence_dir, self.velodyne_dir = self.root, selected[0]
        else:
            self.sequence_dir = _sequence_dir(self.root, self.sequence)
            self.velodyne_dir = self.sequence_dir / "velodyne"
        self.labels_dir = self.sequence_dir / "labels"
        self.calib_path = self.sequence_dir / "calib.txt"
        self.poses_path = self.sequence_dir / "poses.txt"
        self.frame_paths = sorted(self.velodyne_dir.glob("*.bin"))
        if not self.frame_paths:
            raise KittiDataError(f"No .bin frames found in {self.velodyne_dir}")
        self._frame_ids = [int(path.stem) for path in self.frame_paths]
        self.labels_archive_path = next(iter(self.root.glob("*labels*.zip")), None) if self.archive_layout else None
        self.calib_archive_path = next(iter(self.root.glob("*calib*.zip")), None) if self.archive_layout else None
        self._label_member_prefix = f"sequences/{self.sequence}/labels/"
        self._label_members: dict[str, str] = {}
        self._labels_archive: ZipFile | None = None
        if self.labels_archive_path is not None:
            self._labels_archive = ZipFile(self.labels_archive_path)
            self._label_members = {Path(name).stem: name for name in self._labels_archive.namelist()
                                   if f"/{self._label_member_prefix}" in name and name.endswith(".label")}
        if require_labels and not self.has_labels:
            raise KittiDataError(f"SemanticKITTI labels directory is required but missing: {self.labels_dir}")
        if self.archive_layout:
            self.calibration = self._read_archive_calibration()
            self.poses = self._read_archive_poses()
        else:
            self.calibration, self.poses = parse_calibration(self.calib_path), load_poses(self.poses_path)

    @property
    def has_labels(self) -> bool:
        if self.archive_layout:
            return bool(self._label_members)
        return self.labels_dir.is_dir()

    def _archive_member(self, suffix: str, archive_path: Path | None) -> bytes | None:
        if archive_path is None:
            return None
        with ZipFile(archive_path) as archive:
            member = next((name for name in archive.namelist() if name.endswith(suffix)), None)
            return None if member is None else archive.read(member)

    def _read_archive_calibration(self) -> dict[str, np.ndarray]:
        raw = self._archive_member(f"sequences/{self.sequence}/calib.txt", self.calib_archive_path)
        return {} if raw is None else _parse_calibration_text(raw.decode("utf-8"))

    def _read_archive_poses(self) -> np.ndarray | None:
        raw = self._archive_member(f"sequences/{self.sequence}/poses.txt", self.labels_archive_path)
        return None if raw is None else _parse_poses_text(raw.decode("utf-8"), f"{self.labels_archive_path}!poses.txt")

    def __len__(self) -> int:
        return len(self.frame_paths)

    @property
    def frame_ids(self) -> list[int]:
        return self._frame_ids

    def label_path(self, index: int) -> Path:
        return self.labels_dir / f"{self.frame_paths[index].stem}.label"

    def _load_frame_labels(self, index: int) -> np.ndarray | None:
        if self.archive_layout:
            if self._labels_archive is None:
                return None
            member = self._label_members.get(self.frame_paths[index].stem)
            raw = None if member is None else self._labels_archive.read(member)
            if raw is None:
                return None
            if len(raw) % np.dtype(np.uint32).itemsize:
                raise KittiDataError(f"Corrupt archived .label for {self.frame_paths[index].name}")
            return np.frombuffer(raw, dtype=np.uint32)
        path = self.label_path(index)
        return load_packed_labels(path) if path.exists() else None

    def get_frame(self, index: int, require_labels: bool = False) -> KittiFrame:
        if not 0 <= index < len(self):
            raise IndexError(f"Frame index {index} outside [0, {len(self) - 1}]")
        points = load_velodyne_bin(self.frame_paths[index])
        semantic = instances = None
        packed = self._load_frame_labels(index)
        if packed is not None:
            if packed.size != points.shape[0]:
                raise KittiDataError(
                    f"Point/label mismatch for frame {self.frame_paths[index].stem}: "
                    f"{points.shape[0]} points != {packed.size} labels"
                )
            semantic = decode_semantic_labels(packed)
            instances = decode_instance_labels(packed)
        elif require_labels:
            raise KittiDataError(f"Missing label file for {self.frame_paths[index].name}: {self.label_path(index)}")
        pose = None if self.poses is None or index >= len(self.poses) else self.poses[index]
        return KittiFrame(self.sequence, self.frame_ids[index], points, points[:, 3], semantic, instances,
                          self.calibration, pose)

    def iter_frames(self, indices: list[int] | None = None, require_labels: bool = False) -> Iterator[KittiFrame]:
        for index in range(len(self)) if indices is None else indices:
            yield self.get_frame(index, require_labels=require_labels)


def available_sequences(dataset_root: str | Path) -> list[str]:
    root = discover_dataset_root(dataset_root)
    if any(root.glob("*.bin")):
        return ["00"]
    if any(root.glob("*labels*.zip")):
        candidates = _archive_velodyne_dirs(root)
        sequences = set()
        for candidate in candidates:
            if candidate.parent.name.isdigit():
                sequences.add(candidate.parent.name.zfill(2))
            elif candidate.parent.name == "data_odometry_velodyne":
                sequences.add("00")
        return sorted(sequences)
    base = root / "sequences" if (root / "sequences").is_dir() else root
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "velodyne").is_dir())
