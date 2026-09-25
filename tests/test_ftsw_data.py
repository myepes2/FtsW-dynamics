"""Unit tests for src/ftsw_data.py — manifest-driven path resolution.

Run:  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import ftsw_data as fd  # noqa: E402


MANIFEST = """\
sims:
  '14':
    description: apo run
    folder: sims/14_apo
    engine: namd
    default: run1
    variants:
      run1:
        top: sims/14_apo/proc/run1/top.psf
        traj: sims/14_apo/proc/run1/traj.dcd
        time_factor_ns: 0.1
external:
  A:
    description: external apo
    engine: external
    time_factor: 0.2
    index_csv_pair:
      top: Cell_Division_Projects/ext/a.psf
      traj: Cell_Division_Projects/ext/a.dcd
"""


def _make_tree(root: Path) -> None:
    """Minimal FTSW_DATA tree with one internal sim + files."""
    sim_dir = root / "sims" / "14_apo"
    (sim_dir / "proc" / "run1").mkdir(parents=True)
    (sim_dir / "proc" / "run1" / "top.psf").write_text("PSF\n")
    (sim_dir / "proc" / "run1" / "traj.dcd").write_text("dcd")
    (root / "manifest.yaml").write_text(MANIFEST, encoding="utf-8")


class EnvGuard:
    """Set/restore env vars around each test."""

    def __init__(self, **kw):
        self.kw = kw
        self.saved = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestRoots(unittest.TestCase):
    def test_data_root_requires_env(self):
        with EnvGuard(FTSW_DATA=None):
            with self.assertRaises(EnvironmentError):
                fd.data_root()

    def test_data_root_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            with EnvGuard(FTSW_DATA=td):
                self.assertEqual(fd.data_root(), Path(td).resolve())


class TestLoadIndex(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        _make_tree(self.root)
        self._env = EnvGuard(FTSW_DATA=str(self.root), FTSW_INDEX_CSV=None)
        self._env.__enter__()

    def tearDown(self):
        self._env.__exit__()
        self._td.cleanup()

    def test_manifest_rows(self):
        df = fd.load_index()
        self.assertIn("14", set(df["sim_number"]))
        row = df[df["sim_number"] == "14"].iloc[0]
        self.assertEqual(row["sim_description"], "apo run")
        self.assertAlmostEqual(float(row["time_factor"]), 0.1)
        # paths resolved to absolute under root
        self.assertTrue(Path(row["psf_path"]).is_absolute())
        self.assertEqual(
            Path(row["psf_path"]),
            (self.root / "sims/14_apo/proc/run1/top.psf").resolve(),
        )

    def test_sims_csv_fallback(self):
        (self.root / "manifest.yaml").unlink()
        csv = self.root / "sims.csv"
        csv.write_text(
            "sim_number,sim_description,psf_path,dcd_path,time_factor\n"
            "14,apo run,sims/14_apo/proc/run1/top.psf,sims/14_apo/proc/run1/traj.dcd,0.1\n",
            encoding="utf-8",
        )
        df = fd.load_index()
        row = df[df["sim_number"] == "14"].iloc[0]
        self.assertEqual(
            Path(row["psf_path"]),
            (self.root / "sims/14_apo/proc/run1/top.psf").resolve(),
        )

    def test_numeric_sim_id_stays_string(self):
        (self.root / "manifest.yaml").unlink()
        (self.root / "sims.csv").write_text(
            "sim_number,sim_description,psf_path,dcd_path,time_factor\n"
            "14,apo run,sims/14_apo/proc/run1/top.psf,sims/14_apo/proc/run1/traj.dcd,0.1\n",
            encoding="utf-8",
        )
        df = fd.load_index()
        self.assertIn("14", set(df["sim_number"]))


class TestLocations(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        _make_tree(self.root)
        self._env = EnvGuard(FTSW_DATA=str(self.root), FTSW_INDEX_CSV=None)
        self._env.__enter__()

    def tearDown(self):
        self._env.__exit__()
        self._td.cleanup()

    def test_sim_folder_and_analysis_dir(self):
        self.assertEqual(
            fd.sim_folder("14"),
            (self.root / "sims" / "14_apo").resolve(),
        )
        self.assertEqual(
            fd.analysis_dir("14"),
            (self.root / "sims" / "14_apo" / "analysis").resolve(),
        )
        self.assertTrue(fd.analysis_dir("14").is_dir())

    def test_sim_analysis_dirs_skips_synthetic(self):
        m = fd.sim_analysis_dirs(["14", "17full"])
        self.assertIn("14", m)
        self.assertNotIn("17full", m)

    def test_dynamics_dir(self):
        d = fd.dynamics_dir("MyVar")
        self.assertEqual(d, (self.root / "outputs" / "dynamics" / "MyVar").resolve())
        self.assertTrue(d.is_dir())
        d2 = fd.dynamics_dir("MyVar", "17-58")
        self.assertEqual(d2, d / "17-58")


class TestLegacyAndProvenance(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        _make_tree(self.root)
        self._env = EnvGuard(
            FTSW_DATA=str(self.root),
            FTSW_INDEX_CSV=None,
            MDFOLDER=str(self.root / "mdfolder"),
        )
        self._env.__enter__()

    def tearDown(self):
        self._env.__exit__()
        self._td.cleanup()

    def _legacy_dir(self) -> Path:
        d = self.root / "mdfolder" / "FtsW Manuscript" / "OldVar"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_legacy_cache_dirs(self):
        self._legacy_dir()
        dirs = fd.legacy_cache_dirs()
        self.assertTrue(any(d.name == "OldVar" for d in dirs))

    def test_find_cached_csv_prefers_new(self):
        new = fd.analysis_dir("14") / "14_MyVar.csv"
        new.write_text("Time,val\n")
        hit = fd.find_cached_csv("MyVar", "14", legacy_dirs=[self._legacy_dir()])
        self.assertEqual(hit, new)

    def test_find_cached_csv_legacy_fallback(self):
        legacy = self._legacy_dir() / "14_MyVar.csv"
        legacy.write_text("Time,val\n")
        hit = fd.find_cached_csv("MyVar", "14", legacy_dirs=[legacy.parent])
        self.assertEqual(hit, legacy)
        # legacy untouched
        self.assertTrue(legacy.is_file())

    def test_find_cached_csv_deep_dir_name_match(self):
        """Real layout: dir 'FtsW_L198_L236_Ca_dist' holds '8_FtsW_L198-L236.csv'
        at depth 2 under FtsW Manuscript/Analysis_Visualization/."""
        deep = (self.root / "mdfolder" / "FtsW Manuscript"
                / "Analysis_Visualization" / "FtsW_L198_L236_Ca_dist")
        deep.mkdir(parents=True)
        f = deep / "8_FtsW_L198-L236.csv"
        f.write_text("Time,X\n")

        # depth-2 dir must be discovered by default scan
        self.assertIn(deep, fd.legacy_cache_dirs())

        # dir-name rule: observable token == dir name, filename differs
        hit = fd.find_cached_csv("FtsW_L198_L236_Ca_dist", "8")
        self.assertEqual(hit, f)
        # exact-name rule still works when caller uses the file token
        hit2 = fd.find_cached_csv("FtsW_L198-L236", "8")
        self.assertEqual(hit2, f)

    def test_analysis_csv_map(self):
        new = fd.analysis_dir("14") / "14_MyVar.csv"
        new.write_text("Time,val\n")
        m = fd.analysis_csv_map("MyVar", ["14", "17full"])
        self.assertEqual(m, {"14": str(new)})

    def test_write_provenance(self):
        out = fd.dynamics_dir("MyVar", "17-58")
        p = fd.write_provenance(out, sims=["14"], params={"a": 1, "p": Path("x")})
        payload = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(payload["sims"], ["14"])
        self.assertEqual(payload["params"]["a"], 1)
        self.assertIn("git_sha", payload)
        self.assertIn("created_utc", payload)


if __name__ == "__main__":
    unittest.main()
