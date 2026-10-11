"""Meaningful workflow checks on synthetic LAZ. Run: python -m unittest -v."""
import base64
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import core
from core import (HEADERS, Settings, dbscan, display_indices, evaluate_model, extract_features, instance_groups,
                  load_manifest, load_model, make_run_directory, process_batch,
                  projected_area, read_cloud, train_model)
from create_demo import create_demo
from sklearn.cluster import DBSCAN



def web_data(folder):
    page = (Path(folder) / "view3d.html").read_text(encoding="utf-8")
    return json.loads(page.split('type="application/json">', 1)[1].split("</script>", 1)[0])


class WorkflowTest(unittest.TestCase):
    def test_display_sampling_bounds_and_preserves_small_objects(self):
        instance = np.zeros(100_000, dtype=np.int32)
        instance[123] = 1
        instance[-7:] = 2
        for budget in (1000, 5000, 30000):
            selected = display_indices(instance, budget)
            self.assertEqual(len(selected), budget)
            self.assertEqual(len(np.unique(selected)), budget)
            self.assertEqual(set(instance[selected]), {0, 1, 2})
            np.testing.assert_array_equal(selected, display_indices(instance, budget))
        np.testing.assert_array_equal(display_indices(instance[:10]), np.arange(10))
        with self.assertRaises(ValueError):
            display_indices(instance, 30001)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.manifest = create_demo(cls.root / "synthetic")
        cls.cfg = Settings(coordinates_in_meters=True, vertical_datum="synthetic_local_z",
                           area_method="convex_hull_xy")
        cls.model = cls.root / "model.joblib"
        train_model(cls.manifest, cls.model, cls.cfg, iterations=30, log=lambda x: None)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_batch_export_and_failure_isolation(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample, self.root / "missing.laz", sample], self.model,
                                 self.root / "runs", self.cfg, log=lambda x: None))
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual([e["status"] for e in metadata["datasets"]], ["success", "failed", "success"])
        all_ids = set()
        original = read_cloud(sample, self.cfg)["xyz"]
        for key in ("D0001", "D0003"):
            folder = run / "datasets" / key
            with (folder / "report.csv").open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, HEADERS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
            view = web_data(folder)
            self.assertEqual(view["total"], len(original))
            shown = np.frombuffer(base64.b64decode(view["xyz"]), dtype="<f4").reshape(-1, 3)
            np.testing.assert_allclose(shown, original - original.mean(axis=0), atol=1e-3)
            instance = np.frombuffer(base64.b64decode(view["instance"]), dtype="<i4")
            # Small cloud: every point is in the view, in source order.
            for row, obj in zip(rows, objects):
                object_id = row["object ID"]
                self.assertNotIn(object_id, all_ids)
                all_ids.add(object_id)
                self.assertEqual(object_id, obj["object_id"])
                point_file = run / "objects" / f"{object_id}.csv"
                points = np.loadtxt(point_file, delimiter=",", skiprows=1, encoding="utf-8-sig")
                self.assertEqual(len(points), int(row["Nr_of_points"]))
                self.assertTrue(np.allclose(points.mean(axis=0), [float(row[k]) for k in ("x", "y", "z")]))
                self.assertTrue(np.array_equal(points, original[instance == obj["instance_index"]]))
                self.assertGreaterEqual(float(row["confidence"]), self.cfg.threshold)
        other = Path(process_batch([sample], self.model, self.root / "runs", self.cfg, log=lambda x: None))
        self.assertNotEqual(run, other)

    def test_no_objects_is_not_failure(self):
        cfg = Settings(**{**self.cfg.__dict__, "min_points": 1000, "area_method": "unresolved"})
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample], self.model, self.root / "empty", cfg, log=lambda x: None))
        entry = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))["datasets"][0]
        self.assertEqual(entry["status"], "success_no_objects")
        with (run / "datasets/D0001/report.csv").open(encoding="utf-8-sig") as f:
            self.assertEqual(len(list(csv.reader(f))), 1)

    def test_area_definition(self):
        square = np.array([[500000, 6000000, -2], [500002, 6000000, -2],
                           [500002, 6000003, -2], [500000, 6000003, -2]])
        self.assertAlmostEqual(projected_area(square, "convex_hull_xy"), 6.0)
        self.assertIsNone(projected_area(square, "unresolved"))
        self.assertEqual(projected_area(square[:2], "convex_hull_xy"), 0)

    def test_units_limit_and_no_training_overwrite(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        with self.assertRaisesRegex(ValueError, "Confirm"):
            read_cloud(sample, Settings())
        with self.assertRaisesRegex(ValueError, "exceeds"):
            read_cloud(sample, Settings(max_points=100, coordinates_in_meters=True))
        with self.assertRaises(FileExistsError):
            train_model(self.manifest, self.model, self.cfg)

    def test_no_point_limit_by_default_but_memory_guard(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        self.assertEqual(Settings().max_points, 0)
        self.assertGreater(len(read_cloud(sample, self.cfg)["xyz"]), 0)
        total = core.physical_memory()
        if total:
            with self.assertRaisesRegex(ValueError, "Split the file"):
                core.check_memory(total // core.BYTES_PER_POINT)

    def test_browser_view_embeds_points_and_objects(self):
        sample = self.manifest.parent / "SYNTHETIC_test.laz"
        run = Path(process_batch([sample], self.model, self.root / "web", self.cfg, log=lambda x: None))
        folder = run / "datasets" / "D0001"
        data = web_data(folder)
        count = len(read_cloud(sample, self.cfg)["xyz"])
        self.assertEqual(data["total"], count)
        self.assertEqual(len(base64.b64decode(data["xyz"])), count * 12)
        instance = np.frombuffer(base64.b64decode(data["instance"]), dtype="<i4")
        objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
        self.assertEqual({o[1] for o in data["objects"]}, {o["object_id"] for o in objects})
        self.assertEqual(set(instance) - {0}, {o["instance_index"] for o in objects})
        self.assertEqual(data["classes"], [[0, "background"], [1, "synthetic_object"]])
        cls = np.frombuffer(base64.b64decode(data["cls"]), dtype=np.uint8)
        self.assertEqual(len(cls), count)
        self.assertTrue(set(cls[instance > 0]) - {0})  # Object points carry a target class.
        self.assertEqual(len(base64.b64decode(data["intensity"])), count)

    def test_chunked_dbscan_matches_sklearn_core_clusters(self):
        rng = np.random.default_rng(0)
        centres = rng.uniform(0, 20, (40, 3))
        points = np.concatenate([c + rng.normal(0, 0.25, (300, 3)) for c in centres] +
                                [rng.uniform(0, 20, (2000, 3))])
        reference = DBSCAN(eps=0.3, min_samples=10).fit(points)
        core_idx = reference.core_sample_indices_
        # Small chunks force clusters to be joined across many chunk boundaries.
        labels = dbscan(points, 0.3, 10, chunk=500)
        np.testing.assert_array_equal(labels[core_idx], reference.labels_[core_idx])
        np.testing.assert_array_equal(labels == -1, reference.labels_ == -1)

    def test_scene_and_content_leakage_checks(self):
        source = json.loads(self.manifest.read_text())
        source["datasets"][1]["scene_id"] = source["datasets"][0]["scene_id"]
        changed = self.manifest.parent / "bad_scene.json"
        changed.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_manifest(changed)
        source["datasets"][1]["scene_id"] = "different"
        source["datasets"][1]["path"] = source["datasets"][0]["path"]
        changed.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "Repeated"):
            load_manifest(changed)

    def test_model_reload_test_evaluation_and_mlp_backend(self):
        loaded = load_model(self.model)
        self.assertIn("validation", loaded)
        self.assertNotIn("test_metrics", loaded)
        dest = self.root / "test_metrics.json"
        evaluate_model(self.model, dest, self.cfg, log=lambda x: None)
        result = json.loads(dest.read_text())
        self.assertEqual(result["class_order"], [0, 1])
        self.assertIn("confusion_matrix", result)
        mlp_path = self.root / "mlp.joblib"
        train_model(self.manifest, mlp_path, self.cfg, backend="mlp", iterations=2, log=lambda x: None)
        self.assertEqual(load_model(mlp_path)["backend"], "mlp")

    def test_relocated_model_and_test_data(self):
        relocated = self.root / "relocated"
        relocated.mkdir()
        shutil.copytree(self.manifest.parent, relocated / "synthetic")
        shutil.copy2(self.model, relocated / "model.joblib")
        evaluate_model(relocated / "model.joblib", relocated / "metrics.json", self.cfg, log=lambda x: None)
        self.assertTrue((relocated / "metrics.json").exists())

    def test_global_instances_cross_arbitrary_tile_boundary(self):
        xyz = np.array([[x, y, 0] for x in np.arange(-0.5, 0.51, 0.1) for y in (0, 0.1)])
        cfg = Settings(cluster_eps=0.16, min_points=3)
        groups = instance_groups(xyz, np.ones(len(xyz), dtype=int), np.ones(len(xyz)), cfg)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0][1]), len(xyz))

    def test_features_do_not_depend_on_map_translation(self):
        cloud = read_cloud(self.manifest.parent / "SYNTHETIC_test.laz", self.cfg)
        shifted = {**cloud, "xyz": cloud["xyz"] + [10000, -10000, 5]}
        self.assertTrue(np.allclose(extract_features(cloud), extract_features(shifted), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
