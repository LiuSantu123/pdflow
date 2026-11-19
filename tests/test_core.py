from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from biotite.structure import AtomArray

from pdflow import metrics, predict, setup, structures
from pdflow.naming import format_header, parse_header, read_fasta, write_fasta
from pdflow.predict import build_job, write_af3_json, write_af3like_json, write_boltz_yaml


def _write_two_chain_pdb(path: Path) -> None:
    array = AtomArray(4)
    array.coord = np.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [20.0, 0.0, 0.0], [23.8, 0.0, 0.0]]
    )
    array.chain_id = np.array(["A", "A", "B", "B"])
    array.res_id = np.array([1, 2, 1, 2])
    array.res_name = np.array(["ALA", "GLY", "SER", "THR"])
    array.atom_name = np.array(["CA", "CA", "CA", "CA"])
    array.element = np.array(["C", "C", "C", "C"])
    array.hetero = np.array([False, False, False, False])
    structures.save_structure_file(path, array)


def test_header_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.fasta"
    write_fasta(
        path,
        [(format_header("seq1|A", chain="A", backbone="bb1", designer="rfd4"), "ACDE")],
    )
    header, sequence = read_fasta(path)[0]
    assert sequence == "ACDE"
    parsed = parse_header(header)
    assert parsed["id"] == "seq1|A"
    assert parsed["backbone"] == "bb1"
    assert parsed["designer"] == "rfd4"


def test_build_job_replaces_trimmed_target(tmp_path: Path) -> None:
    _write_two_chain_pdb(tmp_path / "full.cif")
    (tmp_path / "target.a3m").write_text(">query\nSER THR\n")
    sequence_path = tmp_path / "seq.fasta"
    write_fasta(
        sequence_path,
        [
            (format_header("seq|A", chain="A", backbone="bb"), "ACDE"),
            (format_header("seq|B", chain="B", backbone="bb"), "ST"),
        ],
    )
    campaign = {
        "targets": {
            "T1": {
                "full": "full.cif",
                "msa": "target.a3m",
                "design_chain": "B",
                "predict_chain": "B",
            }
        }
    }
    job = build_job(sequence_path, campaign, tmp_path, tmp_path / "job")
    assert [chain.id for chain in job.chains] == ["A", "B"]
    target = job.chains[1]
    assert target.sequence == "ST"
    assert target.msa == str((tmp_path / "target.a3m").resolve())


def test_input_writers(tmp_path: Path) -> None:
    _write_two_chain_pdb(tmp_path / "full.cif")
    (tmp_path / "target.a3m").write_text(">query\nSER THR\n")
    sequence_path = tmp_path / "seq.fasta"
    write_fasta(
        sequence_path,
        [
            (format_header("seq|A", chain="A"), "ACDE"),
            (format_header("seq|B", chain="B"), "ST"),
        ],
    )
    campaign = {
        "targets": {
            "T1": {
                "full": "full.cif",
                "msa": "target.a3m",
                "design_chain": "B",
                "predict_chain": "B",
            }
        }
    }
    job = build_job(sequence_path, campaign, tmp_path, tmp_path / "job")

    af3 = write_af3_json(job, tmp_path / "af3.json")
    text = af3.read_text()
    assert '"templates": []' in text
    assert "target.a3m" in text

    boltz = write_boltz_yaml(job, tmp_path / "boltz.yaml", tmp_path / "boltz_msa")
    boltz_text = boltz.read_text()
    assert "msa:" in boltz_text
    assert (tmp_path / "boltz_msa" / "A.a3m").is_file()

    protenix = write_af3like_json(job, tmp_path / "protenix.json", tmp_path / "msa")
    protenix_text = protenix.read_text()
    assert "precomputed_msa_dir" in protenix_text
    assert (tmp_path / "msa" / "B" / "non_pairing.a3m").is_file()


def test_nucleic_acid_target(tmp_path: Path) -> None:
    array = AtomArray(4)
    array.coord = np.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [20.0, 0.0, 0.0], [23.8, 0.0, 0.0]]
    )
    array.chain_id = np.array(["A", "A", "B", "B"])
    array.res_id = np.array([1, 2, 1, 2])
    array.res_name = np.array(["ALA", "GLY", "DA", "DC"])
    array.atom_name = np.array(["CA", "CA", "P", "P"])
    array.element = np.array(["C", "C", "P", "P"])
    array.hetero = np.array([False, False, False, False])
    structures.save_structure_file(tmp_path / "dna.cif", array)
    sequence_path = tmp_path / "seq.fasta"
    write_fasta(sequence_path, [(format_header("seq|A", chain="A"), "AG")])
    campaign = {
        "targets": {
            "T1": {
                "full": "dna.cif",
                "design_chain": "B",
                "predict_chain": "B",
            }
        }
    }
    job = build_job(sequence_path, campaign, tmp_path, tmp_path / "job")
    assert job.chains[1].kind == "dna"
    assert job.chains[1].sequence == "AC"
    af3_text = write_af3_json(job, tmp_path / "af3.json").read_text()
    assert '"dna"' in af3_text
    assert "dnaSequence" in write_af3like_json(
        job, tmp_path / "p.json", tmp_path / "msa"
    ).read_text()


def test_usalign_parser() -> None:
    text = "Aligned length= 125, RMSD=   1.25, Seq_ID=n_identical/n_aligned= 0.5\nTM-score= 0.81234 (normalized by length of Structure_1: L=125, d0=4.14)"
    parsed = metrics._parse_usalign_output(text)
    assert parsed["rmsd"] == pytest.approx(1.25)
    assert parsed["tm_score"] == pytest.approx(0.81234)


def test_physchem_and_low_complexity() -> None:
    values = metrics.physchem("ACDEFGHIKLMNPQRSTVWY")
    assert values["length"] == 20
    assert values["molecular_weight_kda"] > 2
    assert 0 <= values["low_complexity_fraction"] <= 1


def test_discover_models_prefers_per_sample(tmp_path: Path) -> None:
    root = tmp_path / "out"
    sample = root / "job" / "seed-1_sample-0"
    sample.mkdir(parents=True)
    (root / "job" / "job_model.cif").write_text("data_x\n")
    (sample / "job_seed-1_sample-0_model.cif").write_text("data_x\n")
    artifacts = metrics.discover_models("af3", root)
    assert len(artifacts) == 1
    assert artifacts[0].structure.parent.name == "seed-1_sample-0"


def test_setup_materializes_native_profile(tmp_path: Path) -> None:
    values = setup.default_values()
    values.update(
        {
            "HOME": str(tmp_path),
            "CONDA_BASE": str(tmp_path / "conda"),
            "MODEL_ROOT": str(tmp_path / "models"),
            "KIT_ROOT": str(tmp_path / "kit"),
            "AF3_ROOT": str(tmp_path / "af3"),
            "USALIGN": "USalign",
        }
    )
    dest = tmp_path / "backends.yaml"
    setup.materialize_profile("native", dest, values, force=True)
    text = dest.read_text()
    for key in setup.TOKENS:
        assert f"__{key}__" not in text
    data = yaml.safe_load(text)
    assert "predictors" in data
    assert data["predictors"]["esmfold2_overlay"]["enabled"] is False


def test_esmfold2_overlay_is_reserved_but_safe(tmp_path: Path) -> None:
    job = predict.Job("demo", [predict.Chain("A", sequence="ACDE")])
    result = predict.run_esmfold2_overlay(job, tmp_path, {})
    assert result["returncode"] == 0
    assert "disabled" in result["skipped"]
