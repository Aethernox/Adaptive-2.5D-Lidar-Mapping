import json
import tempfile
from zipfile import ZipFile
from pathlib import Path
import numpy as np

from config import CLASS_MAPPING_VERSION, FEATURE_NORMALIZATION_VERSION, NUM_CLASSES
from data.kitti import KittiDataError, KittiSequenceDataset, available_sequences, load_velodyne_bin
from data.semantic_kitti import decode_instance_labels, decode_semantic_labels, map_to_prototype
from data.validation import validate_dataset
from perception.features import extract_features
from perception.model import PointSegModel
from pipeline import Pipeline


def _fixture_root(with_bad_bin=False):
    root = Path(tempfile.mkdtemp()) / "dataset"
    seq = root / "sequences" / "00"; vel = seq / "velodyne"; labels = seq / "labels"
    vel.mkdir(parents=True); labels.mkdir()
    (seq / "calib.txt").write_text("P0: 1 0 0 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    (seq / "poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n" * 3)
    for index in range(3):
        points = np.array([[1 + index, 0, 0, .5], [2, 1, .2, .7], [3, -1, 1, .2]], dtype=np.float32)
        points.tofile(vel / f"{index:06d}.bin")
        packed = np.array([(7 << 16) | 40, (8 << 16) | 10, (0 << 16) | 80], dtype=np.uint32)
        packed.tofile(labels / f"{index:06d}.label")
    if with_bad_bin: (vel / "000003.bin").write_bytes(b"bad")
    return root


def test_bin_loader_and_shape():
    root = _fixture_root(); frame = KittiSequenceDataset(root, "00", True).get_frame(0, True)
    assert frame.points.dtype == np.float32 and frame.points.shape == (3, 4)
    assert frame.intensity.shape == (3,) and frame.calibration["P0"].size == 4


def test_semantic_and_instance_label_decoding():
    packed = np.array([(513 << 16) | 10, (7 << 16) | 40], dtype=np.uint32)
    assert decode_semantic_labels(packed).tolist() == [10, 40]
    assert decode_instance_labels(packed).tolist() == [513, 7]


def test_mapping_ignored_and_coverage():
    labels = np.array([40, 10, 80, 11, 0], dtype=np.uint16)
    mapped = map_to_prototype(labels)
    assert mapped.tolist() == [0, 5, 3, -1, -1]


def test_corrupt_bin_detection():
    root = _fixture_root(with_bad_bin=True)
    try:
        load_velodyne_bin(root / "sequences" / "00" / "velodyne" / "000003.bin")
        assert False, "corrupt bin must fail"
    except KittiDataError as exc:
        assert "divisible" in str(exc)


def test_bin_label_match_and_validation():
    root = _fixture_root(); report = validate_dataset(root, "00")
    assert report["valid_frames"] == 3 and report["mapped_points"] == 9
    assert report["ignored_points"] == 0


def test_train_inference_feature_parity():
    points = np.array([[1, 2, 3, .4], [5, -1, 0, .7]], dtype=np.float32)
    assert np.array_equal(extract_features(points), extract_features(points.copy()))


def test_archive_dataset_layout_loads_labels_calibration_and_poses():
    root = Path(tempfile.mkdtemp())
    velodyne = root / "data_odometry_velodyne" / "velodyne"; velodyne.mkdir(parents=True)
    points = np.array([[1, 0, 0, .5], [2, 1, .2, .7]], dtype=np.float32)
    points.tofile(velodyne / "000000.bin")
    packed = np.array([(13 << 16) | 40, (29 << 16) | 10], dtype=np.uint32)
    with ZipFile(root / "data_odometry_labels.zip", "w") as archive:
        archive.writestr("dataset/sequences/00/labels/000000.label", packed.tobytes())
        archive.writestr("dataset/sequences/00/poses.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n")
    with ZipFile(root / "data_odometry_calib.zip", "w") as archive:
        archive.writestr("dataset/sequences/00/calib.txt", "P0: 1 0 0 0\n")
    dataset = KittiSequenceDataset(root, "00", require_labels=True)
    frame = dataset.get_frame(0, require_labels=True)
    assert frame.points.shape == (2, 4)
    assert frame.semantic_labels.tolist() == [40, 10]
    assert frame.instance_labels.tolist() == [13, 29]
    assert frame.calibration["P0"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert frame.pose.shape == (3, 4)
    assert available_sequences(root) == ["00"]
    try:
        KittiSequenceDataset(root, "01")
        assert False, "unavailable point-cloud sequences must not be selectable"
    except KittiDataError as exc:
        assert "no Velodyne frames" in str(exc)


def test_kitti_pipeline_seek_and_payload_schema():
    root = _fixture_root(); artifact = root / "artifact" / "model"; artifact.mkdir(parents=True)
    model_path = artifact / "model.npz"; PointSegModel(seed=0).save(model_path)
    (artifact / "metadata.json").write_text(json.dumps({"num_classes": NUM_CLASSES, "feature_normalization_version": FEATURE_NORMALIZATION_VERSION, "class_mapping_version": CLASS_MAPPING_VERSION}))
    pipeline = Pipeline(dataset_type="kitti", dataset_root=str(root), sequence="00", model_path=str(model_path))
    frame = pipeline.step()
    assert frame["dataset_type"] == "kitti" and frame["sequence"] == "00"
    assert frame["n_points"] == 3 and len(frame["point_cloud"][0]) == 8
    assert frame["ground_truth_available"] and frame["point_cloud"][0][6] in range(6)
    assert pipeline.seek(2)["frame_idx"] == 2
