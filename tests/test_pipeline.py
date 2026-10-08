import numpy as np

from fitting_room import artifacts, hypotheses, report, slim
from fixtures import make_artifacts


def test_load_and_registry(tmp_path):
    art = make_artifacts(tmp_path / "artifacts")
    A = artifacts.load(art)
    assert set(A["weights"]) >= {"baseline", "dropout_0.2"} and len(A["weights"]["baseline"]) == 6
    out = artifacts.write_registry(art)
    reg = artifacts.load(art)["registry"]
    assert out.exists() and len(reg) == 2 and reg.config_hash.str.len().eq(10).all()


def test_every_hypothesis_gets_a_verdict(tmp_path):
    A = artifacts.load(make_artifacts(tmp_path / "artifacts"))
    results = hypotheses.evaluate(A)
    assert len(results) == len(hypotheses.HYPOTHESES)
    assert {r.verdict for r in results} <= {hypotheses.SUPPORTED, hypotheses.NOT_SUPPORTED, hypotheses.INCONCLUSIVE}
    assert all(r.conclusion and "{" not in r.conclusion for r in results)
    by = {r.hypothesis.id: r.verdict for r in results}
    assert by["H5"] == hypotheses.SUPPORTED and by["H12"] == hypotheses.SUPPORTED


def test_missing_artifacts_are_inconclusive(tmp_path):
    art = make_artifacts(tmp_path / "artifacts")
    (art / "optuna_importance.json").unlink()
    (art / "calibration.csv").unlink()
    by = {r.hypothesis.id: r.verdict for r in hypotheses.evaluate(artifacts.load(art))}
    assert by["H12"] == hypotheses.INCONCLUSIVE and by["H9"] == hypotheses.INCONCLUSIVE


def test_report_and_readme_block(tmp_path):
    A = artifacts.load(make_artifacts(tmp_path / "artifacts"))
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n")
    info = report.write_all(A, tmp_path / "docs", readme)
    text = (tmp_path / "docs" / "RESULTS.md").read_text()
    assert "## Hypothesis scorecard" in text and "## Conclusions" in text and "## References" in text
    assert info["readme_updated"] and "old" not in readme.read_text() and "outro" in readme.read_text()


def test_slim_is_smaller_and_dashboard_compatible(tmp_path):
    art = make_artifacts(tmp_path / "artifacts", n_val=300, n_test=300)
    info = slim.slim(art, tmp_path / "slim", n_val=100)
    S = artifacts.load(tmp_path / "slim")
    assert info["after_mb"] < info["before_mb"] and S["slim"] and len(S["X_val"]) == 100
    assert (S["preds"]["test_probs_final"].argmax(1) != S["preds"]["y_test"]).all()     # only errors are kept
    assert S["weights"]["baseline"][0].dtype == np.float32                              # upcast on load
