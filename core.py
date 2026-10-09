"""Small, offline LAS/LAZ ML prototype. See README_ZH.txt for scope."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import laspy
import numpy as np
import sklearn
from pyproj import CRS
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "0.1.0-prototype"
HEADERS = ["object ID", "x", "y", "z", "Nr_of_points", "Intensity", "Area", "confidence"]
FEATURES = ["local_height", "roughness", "linearity", "planarity", "sphericity",
            "radius", "relative_intensity", "raw_intensity_scaled", "return_position",
            "multi_return", "last_return", "scan_angle", "return_missing"]


@dataclass
class Settings:
    neighbors: int = 20
    max_points: int = 500_000
    threshold: float = 0.7
    cluster_eps: float = 0.3
    min_points: int = 10
    area_method: str = "unresolved"
    crs_override: str = ""
    coordinates_in_meters: bool = False
    vertical_datum: str = "unspecified"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def validate_settings(cfg):
    if not 3 <= cfg.neighbors <= 128 or not 10 <= cfg.max_points <= 2_000_000:
        raise ValueError("neighbors must be 3..128; max_points must be 10..2000000")
    if not 0 <= cfg.threshold <= 1 or cfg.cluster_eps <= 0 or cfg.min_points < 2:
        raise ValueError("Invalid probability threshold / cluster radius / minimum points")
    if cfg.area_method not in ("unresolved", "convex_hull_xy"):
        raise ValueError("Unsupported area method")


def read_cloud(path, cfg):
    validate_settings(cfg)
    path = Path(path)
    if path.suffix.lower() not in (".las", ".laz"):
        raise ValueError("Only LAS / LAZ files are supported")
    with laspy.open(path) as reader:
        count = reader.header.point_count
        if count < 3:
            raise ValueError("Point cloud must contain at least 3 points")
        if count > cfg.max_points:
            raise ValueError(f"{count:,} points exceeds prototype limit {cfg.max_points:,}; no silent downsampling")
        header = reader.header
        cloud = reader.read()
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite coordinates in input")
    dims = set(cloud.point_format.dimension_names)
    required = {"intensity", "return_number", "number_of_returns"}
    if not required <= dims:
        raise ValueError(f"Required attributes missing: {required - dims}")
    notes = []
    native_crs = header.parse_crs()
    selected_crs = native_crs or (CRS.from_user_input(cfg.crs_override) if cfg.crs_override else None)
    if selected_crs is not None:
        if selected_crs.is_geographic or not selected_crs.is_projected:
            raise ValueError("Prototype requires projected metric coordinates; reproject before import")
        if any(abs(a.unit_conversion_factor - 1.0) > 1e-8 for a in selected_crs.axis_info[:2]):
            raise ValueError("CRS horizontal units are not metres; reproject before import")
    else:
        raise ValueError("CRS missing: specify the known projected CRS (e.g. EPSG:32632)")
    if not cfg.coordinates_in_meters:
        raise ValueError("Confirm that all X/Y/Z coordinates use metres, including Z")
    if native_crs is None:
        notes.append("CRS supplied by operator; no reprojection performed")
    if cfg.vertical_datum == "unspecified" or not cfg.vertical_datum.strip():
        notes.append("Vertical datum unspecified; must be confirmed before acceptance")
    waveform_dims = sorted(d for d in dims if "wave" in d or d in ("byte_offset_to_waveform_data",))
    sidecar = path.with_suffix(".wdp")
    waveform_info = {
        "pointer_fields": waveform_dims,
        "external_wdp_exists": sidecar.exists(),
        "internal_waveform_flag": bool(header.global_encoding.waveform_data_packets_internal),
        "external_waveform_flag": bool(header.global_encoding.waveform_data_packets_external),
        "samples_decoded": False,
        "used_by_model": False,
    }
    notes.append("Full waveform is not decoded or used; pointer fields do not prove samples are available")
    intensity = np.asarray(cloud.intensity, dtype=np.float64)
    if not intensity.any():
        notes.append("All intensity values are zero; retained as recorded, not assumed calibrated or missing")
    if "scan_angle" in dims:
        angle = np.asarray(cloud.scan_angle, dtype=np.float64) * 0.006
    elif "scan_angle_rank" in dims:
        angle = np.asarray(cloud.scan_angle_rank, dtype=np.float64)
    else:
        raise ValueError("Scan angle is unavailable")
    return {
        "xyz": xyz, "intensity": intensity,
        "return_number": np.asarray(cloud.return_number, dtype=np.float64),
        "number_of_returns": np.asarray(cloud.number_of_returns, dtype=np.float64),
        "scan_angle": angle,
        "info": {"source": str(path.resolve()), "point_count": len(xyz),
                 "las_version": str(header.version), "point_format": header.point_format.id,
                 "crs": selected_crs.to_string(), "crs_wkt": selected_crs.to_wkt(),
                 "units": "metres (operator confirmed)", "vertical_datum": cfg.vertical_datum,
                 "scales": header.scales.tolist(), "offsets": header.offsets.tolist(),
                 "fields": sorted(dims), "waveform": waveform_info, "warnings": notes},
    }


def extract_features(cloud, neighbors=20):
    """Full resolution; query in chunks to bound the N*k temporary arrays."""
    xyz = cloud["xyz"]
    local = xyz - xyz[0]
    tree = cKDTree(local)
    k = min(neighbors, len(xyz))
    features = np.empty((len(xyz), len(FEATURES)), dtype=np.float32)
    rn, nr = cloud["return_number"], cloud["number_of_returns"]
    for start in range(0, len(xyz), 8192):
        stop = min(start + 8192, len(xyz))
        distances, idx = tree.query(local[start:stop], k=k, workers=1)
        points = local[idx]
        centered = points - points.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / k
        eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
        largest = np.maximum(eigen[:, 2], 1e-12)
        heights = local[start:stop, 2] - np.quantile(points[:, :, 2], 0.1, axis=1)
        values = cloud["intensity"][idx]
        relative = (cloud["intensity"][start:stop] - np.median(values, axis=1)) / (values.std(axis=1) + 1)
        r, n = rn[start:stop], nr[start:stop]
        invalid = (n < 1) | (r < 1) | (r > n)
        features[start:stop] = np.column_stack((
            heights, np.sqrt(eigen[:, 0]), (eigen[:, 2] - eigen[:, 1]) / largest,
            (eigen[:, 1] - eigen[:, 0]) / largest, eigen[:, 0] / largest,
            distances[:, -1], relative, cloud["intensity"][start:stop] / 65535,
            np.where(invalid, 0, (r - 1) / np.maximum(n - 1, 1)),
            (~invalid & (n > 1)), (~invalid & (r == n)),
            cloud["scan_angle"][start:stop] / 90, invalid,
        ))
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    return features


def label_template(path, output, cfg):
    cloud = read_cloud(path, cfg)
    with Path(output).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "class_id"])
        writer.writerows((i, "") for i in range(len(cloud["xyz"])))
    return cloud["info"]


def load_labels(path, count):
    labels = np.full(count, -1, dtype=np.int32)
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"point_index", "class_id"} <= set(reader.fieldnames or []):
            raise ValueError("Labels must have point_index,class_id columns")
        for row in reader:
            idx = int(row["point_index"])
            if not 0 <= idx < count or idx in seen:
                raise ValueError("Duplicate / out-of-range point_index in labels")
            seen.add(idx)
            value = row["class_id"].strip()
            if value:
                label = int(value)
                if label < -1:
                    raise ValueError("class_id: -1=unlabelled, 0=background, positive=target")
                labels[idx] = label
    return labels


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    classes = manifest.get("classes", {})
    if "0" not in classes or len(classes) < 2 or any(int(c) < 0 for c in classes):
        raise ValueError("classes must define 0=background and at least one positive target")
    records, groups, files, hashes = [], {}, set(), set()
    for entry in manifest["datasets"]:
        entry = dict(entry)
        if entry["split"] not in ("train", "validation", "test") or not entry.get("scene_id"):
            raise ValueError("Every dataset needs split and scene_id")
        previous = groups.setdefault(entry["scene_id"], entry["split"])
        if previous != entry["split"]:
            raise ValueError("A scene_id occurs in multiple splits; spatial leakage")
        for key in ("path", "labels"):
            entry[key] = str((path.parent / entry[key]).resolve())
        resolved = entry["path"]
        digest = sha256(resolved)
        if resolved in files or digest in hashes:
            raise ValueError("Repeated point cloud / identical file content in manifest")
        files.add(resolved)
        hashes.add(digest)
        entry["sha256"] = digest
        records.append(entry)
    if set(e["split"] for e in records) != {"train", "validation", "test"}:
        raise ValueError("Provide separate train, validation and test datasets")
    return {"classes": classes, "datasets": records}


def metrics(y, pred, classes):
    ids = sorted(int(k) for k in classes)
    return {"point_metrics": classification_report(y, pred, labels=ids, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=ids).tolist(),
            "class_order": ids,
            "note": "Point metrics only; object detection AP and instance IoU not evaluated"}


def collect(records, split, cfg, classes, log):
    xs, ys = [], []
    count = 0
    for record in records:
        if record["split"] != split:
            continue
        log(f"{split}: {Path(record['path']).name}")
        cloud = read_cloud(record["path"], cfg)
        labels = load_labels(record["labels"], len(cloud["xyz"]))
        selected = labels >= 0
        if not selected.any():
            raise ValueError(f"No labels: {record['labels']}")
        if not set(np.unique(labels[selected])) <= {int(k) for k in classes}:
            raise ValueError("Label class not defined in manifest")
        count += int(selected.sum())
        if count > cfg.max_points:
            raise ValueError("Total labelled points in split exceeds prototype max_points limit")
        xs.append(extract_features(cloud, cfg.neighbors)[selected])
        ys.append(labels[selected])
    return np.concatenate(xs), np.concatenate(ys)


def train_model(manifest_path, output_path, cfg, backend="random_forest", iterations=100, log=print):
    validate_settings(cfg)
    if Path(output_path).exists():
        raise FileExistsError("Model output already exists; choose a new filename")
    if not 1 <= iterations <= 10000:
        raise ValueError("Training iterations / trees must be 1..10000")
    manifest = load_manifest(manifest_path)
    x, y = collect(manifest["datasets"], "train", cfg, manifest["classes"], log)
    if set(np.unique(y)) != {int(k) for k in manifest["classes"]}:
        raise ValueError("Training data must contain every declared class, including background")
    if backend == "random_forest":
        estimator = RandomForestClassifier(n_estimators=iterations, class_weight="balanced",
                                           min_samples_leaf=2, max_depth=24, random_state=42, n_jobs=2)
    elif backend == "mlp":
        estimator = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 64, 32), max_iter=iterations, random_state=42,
            early_stopping=False, batch_size=min(256, len(y))))
    else:
        raise ValueError("Unknown model backend")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, y)
    training_warnings = [str(w.message) for w in caught]
    for message in training_warnings:
        log(message)
    xv, yv = collect(manifest["datasets"], "validation", cfg, manifest["classes"], log)
    validation = metrics(yv, estimator.predict(xv), manifest["classes"])
    # Preserve provenance and add relocatable references for a model/data folder.
    for record in manifest["datasets"]:
        for key in ("path", "labels"):
            try:
                record[key + "_relative_to_model"] = os.path.relpath(record[key], Path(output_path).resolve().parent)
            except ValueError:  # Different Windows drive: preserve absolute reference.
                pass
    payload = {"format": "MOST_ML_V1", "software_version": VERSION,
               "model_id": str(uuid.uuid4()), "created_at": datetime.now().astimezone().isoformat(),
               "backend": backend, "estimator": estimator, "classes": manifest["classes"],
               "features": FEATURES, "neighbors": cfg.neighbors, "training_settings": asdict(cfg),
               "datasets": manifest["datasets"], "validation": validation,
               "training_warnings": training_warnings, "sklearn_version": sklearn.__version__}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    write_json(str(output_path) + ".json", {k: v for k, v in payload.items() if k != "estimator"})
    return str(output_path)


def load_model(path):
    # joblib models are executable pickle objects. Load only locally trusted models.
    model = joblib.load(path)
    if model.get("format") != "MOST_ML_V1" or model.get("features") != FEATURES:
        raise ValueError("Model format or feature schema is incompatible")
    if model.get("sklearn_version") != sklearn.__version__:
        raise ValueError("scikit-learn version differs from model; use its recorded environment")
    return model


def evaluate_model(model_path, output_path, cfg, log=print):
    if Path(output_path).exists():
        raise FileExistsError("Evaluation output already exists")
    model = load_model(model_path)
    cfg = Settings(**{**asdict(cfg), "neighbors": model["neighbors"]})
    for record in model["datasets"]:
        for key in ("path", "labels"):
            if key + "_relative_to_model" in record:
                record[key] = str((Path(model_path).resolve().parent / record[key + "_relative_to_model"]).resolve())
        if record["split"] == "test" and sha256(record["path"]) != record["sha256"]:
            raise ValueError("Test file changed since split registration")
    x, y = collect(model["datasets"], "test", cfg, model["classes"], log)
    result = metrics(y, model["estimator"].predict(x), model["classes"])
    result["model_id"] = model["model_id"]
    write_json(output_path, result)
    return str(output_path)


def make_run_directory(first_path, output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(first_path).stem}_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    for suffix in range(10000):
        candidate = root / (stem if suffix == 0 else f"{stem}_{suffix:03d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("Too many duplicate run directories")


def instance_groups(xyz, classes, probabilities, cfg):
    """Global per-file clustering: no processing tiles, hence no tile duplicates."""
    groups = []
    for cls in sorted(np.unique(classes)):
        if cls == 0:
            continue
        idx = np.flatnonzero((classes == cls) & (probabilities >= cfg.threshold))
        if len(idx) < cfg.min_points:
            continue
        # Sparse radius graph is built one row at a time. Cap adjacency to avoid
        # DBSCAN's quadratic-memory worst case on duplicate / very dense clouds.
        local = xyz[idx] - xyz[idx[0]]
        tree = cKDTree(local)
        counts = tree.query_ball_point(local, cfg.cluster_eps, return_length=True)
        if int(counts.sum()) > 5_000_000:
            raise ValueError("Clustering neighbourhood too dense; reduce radius / prepare smaller ROI")
        labels = DBSCAN(eps=cfg.cluster_eps, min_samples=cfg.min_points, n_jobs=1).fit_predict(local)
        for label in sorted(set(labels) - {-1}):
            object_idx = idx[labels == label]
            if len(object_idx) >= cfg.min_points:
                groups.append((int(cls), object_idx, float(probabilities[object_idx].mean())))
    return groups


def projected_area(xyz, method):
    if method == "unresolved":
        return None
    xy = np.unique(xyz[:, :2] - xyz[0, :2], axis=0)
    if len(xy) < 3 or np.linalg.matrix_rank(xy - xy[0]) < 2:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)  # scipy's 2D volume is polygon area
    except QhullError:
        return None


def process_batch(paths, model_path, output_root, cfg, log=print):
    validate_settings(cfg)
    if not paths:
        raise ValueError("Select at least one LAS / LAZ")
    model = load_model(model_path)
    cfg = Settings(**{**asdict(cfg), "neighbors": model["neighbors"]})
    run = make_run_directory(paths[0], output_root)
    (run / "objects").mkdir()
    (run / "datasets").mkdir()
    summary = {"software_version": VERSION, "started_at": datetime.now().astimezone().isoformat(),
               "timezone": str(datetime.now().astimezone().tzinfo), "settings": asdict(cfg),
               "model_id": model["model_id"], "model_sha256": sha256(model_path),
               "backend": model["backend"], "classes": model["classes"],
               "definitions": {"centroid": "mean of exported original coordinates",
                   "Intensity": "arithmetic mean of recorded raw LAS intensity; no radiometric correction",
                   "Area": cfg.area_method,
                   "confidence": "mean winning-class point probability over final exported instance; uncalibrated",
                   "missing_value": "empty CSV cell / null JSON", "downsampling": "none for processing/export",
                   "instance_method": "per-class global DBSCAN within each dataset; separate files are not merged"},
               "platform": platform.platform(), "datasets": []}
    write_json(run / "run_metadata.json", summary)
    for number, path in enumerate(paths, 1):
        dataset_id = f"D{number:04d}"
        folder = run / "datasets" / dataset_id
        folder.mkdir()
        entry = {"dataset_id": dataset_id, "source": str(Path(path).resolve()),
                 "result_path": str(folder.relative_to(run)), "status": "processing"}
        summary["datasets"].append(entry)
        log(f"{dataset_id}: {Path(path).name}")
        exported = []
        try:
            cloud = read_cloud(path, cfg)
            entry["input"] = cloud["info"]
            entry["source_sha256"] = sha256(path)
            features = extract_features(cloud, cfg.neighbors)
            proba = model["estimator"].predict_proba(features)
            winning = proba.argmax(axis=1)
            classes = np.asarray(model["estimator"].classes_)[winning]
            confidence = proba[np.arange(len(features)), winning]
            groups = instance_groups(cloud["xyz"], classes, confidence, cfg)
            point_instance = np.zeros(len(features), dtype=np.int32)
            objects, rows = [], []
            for index, (cls, selected, score) in enumerate(groups, 1):
                object_id = f"{dataset_id}_O{index:05d}"
                points = cloud["xyz"][selected]
                point_instance[selected] = index
                center = points.mean(axis=0)
                area = projected_area(points, cfg.area_method)
                intensity = float(cloud["intensity"][selected].mean())
                dest = run / "objects" / f"{object_id}.csv"
                exported.append(dest)
                with dest.open("x", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["x", "y", "z"])
                    writer.writerows(points)
                rows.append([object_id, *center, len(points), intensity, area, score])
                objects.append({"object_id": object_id, "instance_index": index, "class_id": cls,
                                "class_name": model["classes"][str(cls)], "point_count": len(points),
                                "centroid": center.tolist(), "confidence": score, "area": area,
                                "csv": str(dest.relative_to(run))})
            with (folder / "report.csv").open("x", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADERS)
                writer.writerows(rows)
            np.savez_compressed(folder / "view.npz", xyz=cloud["xyz"],
                                point_index=np.arange(len(features)), instance=point_instance,
                                predicted_class=classes, confidence=confidence)
            write_json(folder / "objects.json", objects)
            entry.update(status="success" if objects else "success_no_objects", object_count=len(objects))
            log(f"{dataset_id}: {len(objects)} objects")
        except Exception as exc:
            # Never expose partial outputs as successful objects.
            for dest in exported:
                dest.unlink(missing_ok=True)
            for filename in ("report.csv", "view.npz", "objects.json"):
                (folder / filename).unlink(missing_ok=True)
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log(f"{dataset_id}: FAILED: {exc}")
        write_json(folder / "status.json", entry)
        write_json(run / "run_metadata.json", summary)
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(run / "run_metadata.json", summary)
    return str(run)
