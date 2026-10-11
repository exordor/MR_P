"""Small, offline LAS/LAZ ML prototype. See README_ZH.txt for scope."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import platform
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import laspy
import numpy as np
import sklearn
from pyproj import CRS
from scipy.ndimage import generic_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "0.1.0-prototype"
DISPLAY_LIMIT = 30_000
WEB_LIMIT = 3_000_000  # Points in the browser (WebGL) 3D view.
WEB_TEMPLATE = Path(__file__).resolve().parent / "viewer3d.html"
# Measured peak on real YellowScan files is ~330 bytes per point; keep headroom.
BYTES_PER_POINT = 600
WORKERS = os.cpu_count() or 1
HEADERS = ["object ID", "x", "y", "z", "Nr_of_points", "Intensity", "Area", "confidence"]
FEATURES = ["local_height", "roughness", "linearity", "planarity", "sphericity",
            "radius", "relative_intensity", "raw_intensity_scaled", "return_position",
            "multi_return", "last_return", "scan_angle", "return_missing",
            "height_above_bed", "bed_known", "las_bathymetric_bottom", "las_water_surface",
            "las_water_column", "las_ground", "las_noise"]
# ASPRS LAS 1.4 classification codes (including the topo-bathy codes YellowScan uses).
# Input codes are kept on every point the model does not mark as a target.
LAS_CLASSES = {0: "never classified", 1: "unassigned", 2: "ground", 3: "low vegetation",
               4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "low noise",
               9: "water", 17: "bridge deck", 18: "high noise", 40: "bathymetric bottom",
               41: "water surface", 42: "derived water surface", 43: "submerged object",
               44: "IHO S-57 object", 45: "water column"}
USER_CLASS_MIN = 64  # 64..255 are user-definable; trained target classes should use these codes.
BED_CELL = 1.0  # metres; grid used to estimate the local riverbed / seabed surface.
BED_WINDOW = 5  # cells; median over this window, so objects up to ~2 m do not lift the bed.


@dataclass
class Settings:
    neighbors: int = 20
    max_points: int = 0  # 0 = no limit; whole file is processed
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
    if not 3 <= cfg.neighbors <= 128 or not (cfg.max_points == 0 or cfg.max_points >= 10):
        raise ValueError("neighbors must be 3..128; max_points must be 0 (no limit) or at least 10")
    if not 0 <= cfg.threshold <= 1 or cfg.cluster_eps <= 0 or cfg.min_points < 2:
        raise ValueError("Invalid probability threshold / cluster radius / minimum points")
    if cfg.area_method not in ("unresolved", "convex_hull_xy"):
        raise ValueError("Unsupported area method")


def physical_memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # Windows: no sysconf; skip the check.
        return None


def check_memory(count):
    """Refuse early, before decoding, when a file clearly cannot fit in RAM."""
    total = physical_memory()
    need = count * BYTES_PER_POINT
    if total and need > total * 0.8:
        raise ValueError(f"{count:,} points needs about {need / 1024**3:.1f} GB RAM; this computer has "
                         f"{total / 1024**3:.1f} GB. Split the file (e.g. by area) before processing.")


def inspect_file(path):
    """Header-only summary: no points are decoded and no settings are needed."""
    path = Path(path)
    with laspy.open(path) as reader:
        header = reader.header
        dims = set(header.point_format.dimension_names)
        try:
            crs = header.parse_crs()
        except Exception as exc:  # Malformed CRS records should not hide the rest of the summary.
            crs, crs_note = None, f"CRS unreadable: {exc}"
        else:
            crs_note = None
    count = header.point_count
    issues = []
    missing = {"intensity", "return_number", "number_of_returns"} - dims
    if missing:
        issues.append(f"missing fields: {', '.join(sorted(missing))}")
    if not {"scan_angle", "scan_angle_rank"} & dims:
        issues.append("no scan angle")
    if crs is None:
        issues.append(crs_note or "no CRS in file: enter Fallback CRS")
        crs_text = ""
    else:
        crs_text = crs.to_string()
        if crs.is_geographic or not crs.is_projected:
            issues.append("CRS is not projected: reproject to metres first")
        elif any(abs(a.unit_conversion_factor - 1.0) > 1e-8 for a in crs.axis_info[:2]):
            issues.append("CRS horizontal units are not metres")
    total = physical_memory()
    need = count * BYTES_PER_POINT
    if total and need > total * 0.8:
        issues.append("too large for this computer's memory: split the file")
    if count < 3:
        issues.append("fewer than 3 points")
    return {"source": str(path.resolve()), "name": path.name, "file_bytes": path.stat().st_size,
            "point_count": count, "las_version": str(header.version), "point_format": header.point_format.id,
            "mins": header.mins.tolist(), "maxs": header.maxs.tolist(), "crs": crs_text,
            "horizontal_unit": crs.axis_info[0].unit_name if crs is not None and crs.axis_info else "",
            "fields": sorted(dims), "memory_bytes": need, "ram_bytes": total, "issues": issues}


def read_cloud(path, cfg):
    validate_settings(cfg)
    path = Path(path)
    if path.suffix.lower() not in (".las", ".laz"):
        raise ValueError("Only LAS / LAZ files are supported")
    with laspy.open(path) as reader:
        count = reader.header.point_count
        if count < 3:
            raise ValueError("Point cloud must contain at least 3 points")
        if cfg.max_points and count > cfg.max_points:
            raise ValueError(f"{count:,} points exceeds the Max points / file setting {cfg.max_points:,} "
                             "(set it to 0 for no limit); no silent downsampling")
        check_memory(count)
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
        "classification": np.asarray(cloud.classification, dtype=np.uint8),
        "info": {"source": str(path.resolve()), "point_count": len(xyz),
                 "las_version": str(header.version), "point_format": header.point_format.id,
                 "crs": selected_crs.to_string(), "crs_wkt": selected_crs.to_wkt(),
                 "units": "metres (operator confirmed)", "vertical_datum": cfg.vertical_datum,
                 "scales": header.scales.tolist(), "offsets": header.offsets.tolist(),
                 "fields": sorted(dims), "waveform": waveform_info, "warnings": notes},
    }


def bed_height(xyz, classification):
    """Height of each point above the local bathymetric bottom (LAS class 40) and whether it is known.

    Bottom points are binned into BED_CELL cells; each cell's median Z is smoothed by a BED_WINDOW median,
    so a target lying on the bed does not raise the bed estimate beneath itself.
    """
    bed = classification == 40
    height, known = np.zeros(len(xyz), dtype=np.float64), np.zeros(len(xyz), dtype=bool)
    if bed.sum() < 10:
        return height, known
    low = xyz[bed, :2].min(axis=0)
    cell = np.floor((xyz[:, :2] - low) / BED_CELL).astype(np.int64)
    shape = cell[bed].max(axis=0) + 1
    order = np.lexsort((xyz[bed, 2], cell[bed, 1], cell[bed, 0]))
    flat = (cell[bed, 0] * shape[1] + cell[bed, 1])[order]
    z = xyz[bed, 2][order]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    counts = np.diff(np.r_[starts, len(flat)])
    grid = np.full(shape[0] * shape[1], np.nan)
    grid[flat[starts]] = z[starts + counts // 2]  # Z is sorted within each cell: middle value is the median.
    grid = grid.reshape(shape)
    smooth = generic_filter(grid, lambda v: np.nan if np.isnan(v).all() else np.nanmedian(v),
                            size=BED_WINDOW, mode="constant", cval=np.nan)
    inside = (cell >= 0).all(axis=1) & (cell[:, 0] < shape[0]) & (cell[:, 1] < shape[1])
    surface = np.full(len(xyz), np.nan)
    surface[inside] = smooth[cell[inside, 0], cell[inside, 1]]
    known = np.isfinite(surface)
    height[known] = xyz[known, 2] - surface[known]
    return np.clip(height, -5, 5), known


def extract_features(cloud, neighbors=20, log=None):
    """Full resolution; chunks bound the N*k temporaries and run on all CPU cores."""
    xyz = cloud["xyz"]
    classification = cloud.get("classification", np.zeros(len(xyz), dtype=np.uint8))
    above_bed, bed_known = bed_height(xyz, classification)
    local = xyz - xyz[0]
    tree = cKDTree(local)
    k = min(neighbors, len(xyz))
    features = np.empty((len(xyz), len(FEATURES)), dtype=np.float32)
    rn, nr = cloud["return_number"], cloud["number_of_returns"]

    def chunk(start):
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
            above_bed[start:stop], bed_known[start:stop], *class_flags(classification[start:stop]),
        ))
    starts = range(0, len(xyz), 8192)
    # Threads share the arrays; numpy/scipy release the GIL, and slices are disjoint.
    with ThreadPoolExecutor(WORKERS) as pool:
        for done, _ in enumerate(pool.map(chunk, starts), 1):
            if log and len(starts) > 100 and done % (len(starts) // 4) == 0:
                log(f"  features {100 * done // len(starts)}%")
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    return features


def class_flags(codes):
    """Vendor LAS classes the model can use: bottom, water surface, water column, ground, noise."""
    return (codes == 40, np.isin(codes, (9, 41, 42)), codes == 45, codes == 2, np.isin(codes, (7, 18)))


def label_template(path, output, cfg):
    cloud = read_cloud(path, cfg)
    with Path(output).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "class_id"])
        writer.writerows((i, "") for i in range(len(cloud["xyz"])))
    return cloud["info"]


def load_labels(path, count, targets=None):
    """point_index,class_id CSV, or a LAS/LAZ whose classification field marks the targets.

    In a LAS/LAZ label file every point whose code is one of the manifest's target classes is that target;
    all other points are background (0), whatever their vendor class.
    """
    if Path(path).suffix.lower() in (".las", ".laz"):
        codes = np.asarray(laspy.read(path).classification, dtype=np.int32)
        if len(codes) != count:
            raise ValueError("Label LAS/LAZ has a different point count from its point cloud")
        return np.where(np.isin(codes, list(targets or ())), codes, 0)
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
    if "0" not in classes or len(classes) < 2 or any(not 0 <= int(c) <= 255 for c in classes):
        raise ValueError("classes must define 0=background and at least one target code 1..255 (LAS classes)")
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
    balance = manifest.get("balance_classes", True)
    if not isinstance(balance, bool):
        raise ValueError("balance_classes must be true or false")
    return {"classes": classes, "balance_classes": balance, "datasets": records}


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
        labels = load_labels(record["labels"], len(cloud["xyz"]), {int(k) for k in classes if int(k)})
        selected = labels >= 0
        if not selected.any():
            raise ValueError(f"No labels: {record['labels']}")
        if not set(np.unique(labels[selected])) <= {int(k) for k in classes}:
            raise ValueError("Label class not defined in manifest")
        count += int(selected.sum())
        if cfg.max_points and count > cfg.max_points:
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
        estimator = RandomForestClassifier(n_estimators=iterations,
                                           class_weight="balanced" if manifest["balance_classes"] else None,
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
               "balance_classes": manifest["balance_classes"],
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


def predict(estimator, features):
    """Class probabilities in bounded batches, using all cores for random forests."""
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = WORKERS
    return np.concatenate([estimator.predict_proba(features[a:a + 1_000_000])
                           for a in range(0, len(features), 1_000_000)])


def dbscan(points, eps, min_points, chunk=32768):
    """DBSCAN with memory bounded by chunk size instead of the whole neighbour graph.

    Core points (>= min_points within eps, self included) and their clusters match
    scikit-learn exactly. A border point joins the cluster of its nearest core point.
    """
    n = len(points)
    tree = cKDTree(points)
    counts = np.concatenate([tree.query_ball_point(points[a:a + chunk], eps, return_length=True, workers=WORKERS)
                             for a in range(0, n, chunk)])
    core = np.flatnonzero(counts >= min_points)
    labels = np.full(n, -1, dtype=np.int64)
    if not len(core):
        return labels
    core_xyz = points[core]
    core_tree = cKDTree(core_xyz)
    root = np.arange(len(core))  # Representative (smallest member) of each core point's cluster.
    for a in range(0, len(core), chunk):
        pairs = cKDTree(core_xyz[a:a + chunk]).sparse_distance_matrix(core_tree, eps, output_type="ndarray")
        i, j = root[pairs["i"] + a], root[pairs["j"]]
        keep = i != j
        if not keep.any():
            continue
        nodes, inverse = np.unique(np.concatenate((i[keep], j[keep])), return_inverse=True)
        m = int(keep.sum())
        graph = coo_matrix((np.ones(m, dtype=np.int8), (inverse[:m], inverse[m:])), shape=(len(nodes), len(nodes)))
        _, component = connected_components(graph, directed=False)
        smallest = np.full(component.max() + 1, len(core))
        np.minimum.at(smallest, component, nodes)
        mapping = np.arange(len(core))
        mapping[nodes] = smallest[component]
        root = mapping[root]
    labels[core] = np.unique(root, return_inverse=True)[1]
    border = np.flatnonzero(counts < min_points)
    for a in range(0, len(border), chunk):
        ids = border[a:a + chunk]
        dist, nearest = core_tree.query(points[ids], k=1, distance_upper_bound=np.nextafter(eps, np.inf),
                                        workers=WORKERS)
        found = np.isfinite(dist)
        labels[ids[found]] = labels[core[nearest[found]]]
    return labels


def instance_groups(xyz, classes, probabilities, cfg):
    """Global per-file clustering: no processing tiles, hence no tile duplicates."""
    groups = []
    for cls in sorted(np.unique(classes)):
        if cls == 0:
            continue
        idx = np.flatnonzero((classes == cls) & (probabilities >= cfg.threshold))
        if len(idx) < cfg.min_points:
            continue
        labels = dbscan(xyz[idx] - xyz[idx[0]], cfg.cluster_eps, cfg.min_points)
        # One sort instead of a full scan per cluster; stable, so point order is kept.
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(labels.max() + 2))
        for label in range(labels.max() + 1):
            object_idx = idx[order[bounds[label]:bounds[label + 1]]]
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


def display_indices(instance, limit=DISPLAY_LIMIT, maximum=DISPLAY_LIMIT):
    """Bound display size, retaining a representative of each instance if possible."""
    if not 1 <= limit <= maximum:
        raise ValueError(f"Display limit must be 1..{maximum}")
    if len(instance) <= limit:
        return np.arange(len(instance))
    labels, first = np.unique(instance, return_index=True)
    representatives = first[labels > 0]
    rng = np.random.default_rng(42)
    if len(representatives) > limit:
        return np.sort(rng.choice(representatives, limit, replace=False))
    available = np.ones(len(instance), dtype=bool)
    available[representatives] = False
    extra = rng.choice(np.flatnonzero(available), limit - len(representatives), replace=False)
    return np.sort(np.concatenate((representatives, extra)))


def web_indices(instance, limit=WEB_LIMIT, keep=None):
    """Browser sample: object points get at least half the budget, so objects stay dense.

    keep: optional mask of further points to treat like object points (e.g. ground-truth targets).
    """
    if len(instance) <= limit:
        return np.arange(len(instance))
    important = instance > 0 if keep is None else (instance > 0) | keep
    if keep is not None:  # Sample over the combined set; display_indices only needs group labels.
        instance = np.where(instance > 0, instance, important.astype(instance.dtype) * (instance.max() + 1))
    objects, background = np.flatnonzero(important), np.flatnonzero(~important)
    take_objects = min(len(objects), max(limit // 2, limit - len(background)))
    take_background = min(len(background), limit - take_objects)
    chosen = objects[display_indices(instance[objects], take_objects, len(objects))] if take_objects else objects[:0]
    rng = np.random.default_rng(42)
    return np.sort(np.concatenate((chosen, rng.choice(background, take_background, replace=False))))


def write_web_view(dest, name, xyz, instance, objects, origin, total, predicted=None, class_names=None,
                   intensity=None, las_classes=None, truth=None, truth_objects=None):
    """Self-contained offline HTML page drawing the points with WebGL (GPU).

    predicted: class ID per point, class_names: {class ID: name}; intensity: raw LAS intensity per point;
    las_classes: input LAS classification code per point; truth: per-point ground-truth status
    (TRUTH_STATUS) and truth_objects: ground-truth targets, when the input has a ground-truth file.
    """
    def encode(array):
        return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")
    data = {"name": name, "total": int(total), "origin_z": float(origin[2]),
            "xyz": encode((xyz - origin).astype("<f4")), "instance": encode(instance.astype("<i4")),
            "objects": [[o["instance_index"], o["object_id"], o["class_name"], o["point_count"], o["confidence"],
                         *(np.asarray(o["centroid"]) - origin).tolist()] for o in objects]}
    if predicted is not None and class_names:
        ids = sorted(int(k) for k in class_names)
        data["classes"] = [[i, class_names[str(i)]] for i in ids]
        data["cls"] = encode(np.searchsorted(ids, predicted).astype(np.uint8))
    if intensity is not None and len(intensity):
        low, high = np.percentile(intensity, [2, 98])
        scaled = np.clip((intensity - low) / max(high - low, 1e-9) * 255, 0, 255)
        data["intensity"] = encode(scaled.astype(np.uint8))
    if las_classes is not None:
        codes = sorted(int(c) for c in np.unique(las_classes))
        data["las"] = encode(np.asarray(las_classes, dtype=np.uint8))
        data["las_names"] = [[c, LAS_CLASSES.get(c, "user-defined" if c >= USER_CLASS_MIN else "reserved")]
                             for c in codes]
    if truth is not None:
        data["truth"] = encode(np.asarray(truth, dtype=np.uint8))
        data["truth_objects"] = [[t["object_id"], t["class_name"], t["status"], *(np.asarray(t["center"]) - origin).tolist()]
                                 for t in truth_objects]
    # "</" is escaped so no value can close the <script> element that holds the data.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = WEB_TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", payload, 1)
    Path(dest).write_text(page, encoding="utf-8")


def refresh_web_view(page):
    """Move a saved 3D page onto the current viewer template, keeping its embedded data."""
    page = Path(page)
    text = page.read_text(encoding="utf-8")
    marker = '<script id="data" type="application/json">'
    start = text.find(marker) + len(marker)
    end = text.find("</script>", start)
    if start < len(marker) or end < 0:
        return
    head, tail = WEB_TEMPLATE.read_text(encoding="utf-8").split("/*DATA*/", 1)
    if text[:start] == head and text[end:] == tail:
        return
    page.write_text(head + text[start:end] + tail, encoding="utf-8")


TRUTH_STATUS = {0: "not a target", 1: "detected target", 2: "false alarm", 3: "missed target"}


def truth_path(source):
    """Ground truth for an input lives next to it: scene.laz -> scene.truth.json."""
    return Path(source).with_suffix(".truth.json")


def load_truth(path, count):
    truth = json.loads(Path(path).read_text(encoding="utf-8"))
    if truth.get("point_count") != count:
        raise ValueError("Ground truth was made for a different point cloud (point count differs)")
    instance = np.zeros(count, dtype=np.int32)
    for number, target in enumerate(truth["objects"], 1):
        if "point_range" in target:
            start, stop = target["point_range"]
            instance[start:stop] = number
        else:
            instance[np.asarray(target["point_indices"], dtype=np.int64)] = number
    return truth["objects"], instance


def evaluate_objects(point_instance, objects, targets, truth_instance, min_iou=0.25):
    """Match detected objects to ground-truth targets one-to-one by point IoU (greedy, highest first).

    A match needs IoU >= min_iou; it is class-correct when the detected class equals the target class.
    Objects and targets are annotated in place; returns the summary and per-point TRUTH_STATUS.
    """
    detected = np.bincount(point_instance, minlength=len(objects) + 1)
    true_size = np.bincount(truth_instance, minlength=len(targets) + 1)
    both = (point_instance > 0) & (truth_instance > 0)
    pairs, overlap = np.unique(np.stack((point_instance[both], truth_instance[both])), axis=1, return_counts=True)
    iou = overlap / (detected[pairs[0]] + true_size[pairs[1]] - overlap)
    match_target, match_object = {}, {}
    for k in np.argsort(-iou, kind="stable"):
        d, t = int(pairs[0, k]), int(pairs[1, k])
        if iou[k] >= min_iou and d not in match_object and t not in match_target:
            match_object[d], match_target[t] = (t, float(iou[k])), (d, float(iou[k]))
    for obj in objects:
        found = match_object.get(obj["instance_index"])
        if found:
            target = targets[found[0] - 1]
            obj["truth"] = {"target_id": target["object_id"], "iou": found[1],
                            "class_correct": obj["class_id"] == target["class_id"],
                            "centre_error_m": float(np.linalg.norm(np.subtract(obj["centroid"], target["center"])))}
        else:
            obj["truth"] = {"target_id": None, "iou": 0.0, "class_correct": False, "centre_error_m": None}
    for number, target in enumerate(targets, 1):
        found = match_target.get(number)
        target["visible_points"] = int(true_size[number])
        target["status"] = "detected" if found else "missed"
        target["detected_as"] = objects[found[0] - 1]["object_id"] if found else None
        target["iou"] = found[1] if found else 0.0
    status = np.zeros(len(point_instance), dtype=np.uint8)
    status[truth_instance > 0] = 3
    matched = np.zeros(len(objects) + 1, dtype=bool)
    matched[list(match_object)] = True
    status[point_instance > 0] = np.where(matched[point_instance[point_instance > 0]], 1, 2)
    tp, fp, fn = len(match_object), len(objects) - len(match_object), len(targets) - len(match_object)
    matches = [o["truth"] for o in objects if o["truth"]["target_id"]]
    classes = sorted({t["class_name"] for t in targets})
    summary = {
        "targets": len(targets), "detections": len(objects), "true_positives": tp,
        "false_positives": fp, "missed": fn,
        "precision": tp / len(objects) if objects else None, "recall": tp / len(targets) if targets else None,
        "f1": 2 * tp / (len(objects) + len(targets)) if objects or targets else None,
        "class_correct": sum(m["class_correct"] for m in matches),
        "mean_iou": float(np.mean([m["iou"] for m in matches])) if matches else None,
        "mean_centre_error_m": float(np.mean([m["centre_error_m"] for m in matches])) if matches else None,
        "recall_by_class": {c: sum(t["status"] == "detected" for t in targets if t["class_name"] == c)
                            / sum(t["class_name"] == c for t in targets) for c in classes},
        "recall_by_visible_points": {
            label: f"{sum(t['status'] == 'detected' for t in group)}/{len(group)}"
            for label, group in (("<8", [t for t in targets if t["visible_points"] < 8]),
                                 ("8-19", [t for t in targets if 8 <= t["visible_points"] < 20]),
                                 (">=20", [t for t in targets if t["visible_points"] >= 20])) if group},
        "missed_targets": [t["object_id"] for t in targets if t["status"] == "missed"],
        "matching": f"one-to-one, point IoU >= {min_iou}",
    }
    return summary, status


POINT_FORMAT_WITH_FULL_CLASS = {0: 6, 1: 6, 2: 7, 3: 7, 4: 9, 5: 10}


def write_classified(source, dest, groups, point_instance):
    """Copy of the input with every attribute kept; detected target points get their target LAS class.

    Points not in a detected object keep the input (vendor) class. An extra dimension object_instance
    holds the object number (0 = none). Point formats 0-5 only store classes 0-31, so they are upgraded
    to the matching LAS 1.4 format when a target code is larger.
    """
    las = laspy.read(source)
    codes = np.asarray(las.classification, dtype=np.uint8).copy()
    for cls, selected, _ in groups:
        codes[selected] = cls
    if las.header.point_format.id < 6 and codes.max() > 31:
        las = laspy.convert(las, point_format_id=POINT_FORMAT_WITH_FULL_CLASS[las.header.point_format.id],
                            file_version="1.4")
    las.classification = codes
    las.add_extra_dim(laspy.ExtraBytesParams(name="object_instance", type=np.uint32,
                                             description="Detected object number, 0 = none"))
    las.object_instance = point_instance.astype(np.uint32)
    las.write(dest)


# Rough share of a file's processing time per stage, for the progress bar.
STAGES = {"reading": 0.0, "features": 0.15, "classifying": 0.65, "grouping": 0.85, "writing": 0.95}


def process_batch(paths, model_path, output_root, cfg, log=print, progress=None):
    """progress(file_number, file_count, name, stage, fraction_of_batch) is called as work advances."""
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
               "model_id": model["model_id"], "model_path": str(Path(model_path).resolve()),
               "model_sha256": sha256(model_path),
               "backend": model["backend"], "classes": model["classes"],
               "definitions": {"centroid": "mean of exported original coordinates",
                   "Intensity": "arithmetic mean of recorded raw LAS intensity; no radiometric correction",
                   "Area": cfg.area_method,
                   "confidence": "mean winning-class point probability over final exported instance; uncalibrated",
                   "missing_value": "empty CSV cell / null JSON", "downsampling": "none for processing/export",
                   "instance_method": "per-class global DBSCAN within each dataset; separate files are not merged",
                   "classified_laz": "input copy; detected target points get their target LAS class, others keep the input class"},
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

        def stage(name):
            if progress:
                progress(number, len(paths), Path(path).name, name, (number - 1 + STAGES[name]) / len(paths))
        exported = []
        try:
            stage("reading")
            cloud = read_cloud(path, cfg)
            entry["input"] = cloud["info"]
            entry["source_sha256"] = sha256(path)
            log(f"{dataset_id}: {len(cloud['xyz']):,} points")
            stage("features")
            features = extract_features(cloud, cfg.neighbors, log)
            stage("classifying")
            log(f"{dataset_id}: classifying points")
            proba = predict(model["estimator"], features)
            winning = proba.argmax(axis=1)
            classes = np.asarray(model["estimator"].classes_)[winning]
            confidence = proba[np.arange(len(features)), winning]
            stage("grouping")
            log(f"{dataset_id}: grouping points into objects")
            groups = instance_groups(cloud["xyz"], classes, confidence, cfg)
            stage("writing")
            log(f"{dataset_id}: writing {len(groups)} objects")
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
            write_classified(path, folder / "classified.laz", groups, point_instance)
            truth_status, targets = None, None
            if truth_path(path).exists():
                targets, truth_instance = load_truth(truth_path(path), len(features))
                summary_truth, truth_status = evaluate_objects(point_instance, objects, targets, truth_instance)
                write_json(folder / "evaluation.json", {"source": str(truth_path(path)), "summary": summary_truth,
                                                        "targets": targets})
                entry["evaluation"] = summary_truth
                log(f"{dataset_id}: ground truth: {summary_truth['true_positives']}/{len(targets)} targets found, "
                    f"{summary_truth['false_positives']} false alarms")
            web = web_indices(point_instance, keep=None if truth_status is None else truth_status > 0)
            write_web_view(folder / "view3d.html", Path(path).name, cloud["xyz"][web], point_instance[web],
                           objects, cloud["xyz"].mean(axis=0), len(features), classes[web], model["classes"],
                           cloud["intensity"][web], cloud["classification"][web],
                           None if truth_status is None else truth_status[web], targets)
            entry["display"] = {"source_points": len(features), "shown_points": len(web), "limit": WEB_LIMIT,
                                "page": "view3d.html", "method": "object points first, then random background"}
            write_json(folder / "objects.json", objects)
            entry.update(status="success" if objects else "success_no_objects", object_count=len(objects))
            log(f"{dataset_id}: {len(objects)} objects")
        except Exception as exc:
            # Never expose partial outputs as successful objects.
            for dest in exported:
                dest.unlink(missing_ok=True)
            for filename in ("report.csv", "view3d.html", "objects.json", "classified.laz", "evaluation.json"):
                (folder / filename).unlink(missing_ok=True)
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log(f"{dataset_id}: FAILED: {exc}")
        write_json(folder / "status.json", entry)
        write_json(run / "run_metadata.json", summary)
    if progress:
        progress(len(paths), len(paths), "", "done", 1.0)
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(run / "run_metadata.json", summary)
    return str(run)
