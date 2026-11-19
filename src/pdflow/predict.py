from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import structures
from .config import get, run_command, tool_env
from .naming import parse_header, read_fasta


@dataclass
class Chain:
    id: str
    kind: str = "protein"           # protein | dna | rna | ligand
    sequence: str = ""
    ccd: list[str] = field(default_factory=list)
    smiles: str = ""
    msa: str = ""
    templates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Job:
    name: str
    chains: list[Chain]
    seed: int = 1

    @property
    def proteins(self) -> list[Chain]:
        return [chain for chain in self.chains if chain.kind == "protein"]


def _target_entries(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    targets = get(campaign, "targets", {}) or {}
    entries: list[dict[str, Any]] = []
    for name, spec in targets.items():
        if not isinstance(spec, dict):
            continue
        entries.append({"name": name, **spec})
    return entries


def build_job(
    sequence_path: Path,
    campaign: dict[str, Any],
    campaign_dir: Path,
    template_dir: Path,
) -> Job:
    """Combine designed chains with full-length target chains and ligands."""
    records = read_fasta(sequence_path)
    designed: dict[str, Chain] = {}
    for header, sequence in records:
        metadata = parse_header(header)
        chain_id = metadata.get("chain", "A")
        designed[chain_id] = Chain(id=chain_id, kind="protein", sequence=sequence)

    targets = _target_entries(campaign)
    replaced: set[str] = set()
    full_chains: list[Chain] = []
    for target in targets:
        design_chain = str(target.get("design_chain", ""))
        predict_chain = str(target.get("predict_chain", design_chain))
        full_path = target.get("full")
        if not full_path:
            continue
        full_path = (campaign_dir / full_path).resolve()
        if not full_path.is_file():
            continue
        full_array = structures.load_structure_file(full_path)
        sequence = structures.chain_sequence(full_array, predict_chain)
        sequence_kind = structures.chain_kind(full_array, predict_chain)
        if not sequence:
            chains = structures.chain_ids(full_array)
            if not chains:
                continue
            predict_chain = chains[0]
            sequence = structures.chain_sequence(full_array, predict_chain)
            sequence_kind = structures.chain_kind(full_array, predict_chain)
        kind = str(target.get("kind", sequence_kind))
        msa = target.get("msa")
        msa_path = ""
        if msa and kind == "protein":
            candidate = (campaign_dir / msa).resolve()
            if candidate.is_file():
                msa_path = str(candidate)
        full_chains.append(
            Chain(
                id=predict_chain,
                kind=kind,
                sequence=sequence,
                msa=msa_path,
                templates=list(target.get("templates", []) or []),
            )
        )
        if design_chain:
            replaced.add(design_chain)

    chains: list[Chain] = []
    for chain_id, chain in designed.items():
        if chain_id in replaced:
            continue
        chains.append(chain)
    chains.extend(full_chains)

    for ligand in campaign.get("ligands", []) or []:
        chains.append(
            Chain(
                id=str(ligand.get("id", "L")),
                kind="ligand",
                ccd=[str(item) for item in ligand.get("ccd", [])],
                smiles=str(ligand.get("smiles", "")),
            )
        )

    name = sequence_path.stem
    template_dir.mkdir(parents=True, exist_ok=True)
    return Job(name=name, chains=chains, seed=int(get(campaign, "design.seed", 1) or 1))


# --------------------------------------------------------------------------- #
# Input writers
# --------------------------------------------------------------------------- #

def write_af3_json(job: Job, path: Path) -> Path:
    sequences: list[dict[str, Any]] = []
    for chain in job.chains:
        if chain.kind == "protein":
            entry: dict[str, Any] = {
                "id": chain.id,
                "sequence": chain.sequence,
                "unpairedMsa": chain.msa or "",
                "pairedMsa": "",
                "templates": list(chain.templates),
            }
            sequences.append({"protein": entry})
        elif chain.kind == "dna":
            sequences.append({"dna": {"id": chain.id, "sequence": chain.sequence}})
        elif chain.kind == "rna":
            sequences.append({"rna": {"id": chain.id, "sequence": chain.sequence}})
        elif chain.kind == "ligand":
            if chain.smiles:
                sequences.append({"ligand": {"id": chain.id, "smiles": chain.smiles}})
            elif chain.ccd:
                sequences.append({"ligand": {"id": chain.id, "ccdCodes": chain.ccd}})
    payload = {
        "name": job.name,
        "modelSeeds": [job.seed],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_boltz_yaml(job: Job, path: Path, msa_root: Path | None = None) -> Path:
    sequences: list[dict[str, Any]] = []
    for chain in job.chains:
        if chain.kind == "protein":
            entry: dict[str, Any] = {"id": chain.id, "sequence": chain.sequence}
            if chain.msa:
                entry["msa"] = chain.msa
            elif msa_root is not None:
                # Boltz requires an MSA entry even when no homologues are used.
                msa_root.mkdir(parents=True, exist_ok=True)
                query_msa = msa_root / f"{chain.id}.a3m"
                if not query_msa.is_file():
                    query_msa.write_text(
                        f">query\n{chain.sequence}\n", encoding="utf-8"
                    )
                entry["msa"] = str(query_msa)
            sequences.append({"protein": entry})
        elif chain.kind == "dna":
            sequences.append({"dna": {"id": chain.id, "sequence": chain.sequence}})
        elif chain.kind == "rna":
            sequences.append({"rna": {"id": chain.id, "sequence": chain.sequence}})
        elif chain.kind == "ligand":
            if chain.ccd:
                sequences.append({"ligand": {"id": chain.id, "ccd": chain.ccd[0]}})
            elif chain.smiles:
                sequences.append({"ligand": {"id": chain.id, "smiles": chain.smiles}})
    payload = {"version": 1, "sequences": sequences}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_af3like_json(
    job: Job,
    path: Path,
    msa_root: Path | None,
    query_msa_fallback: bool = False,
) -> Path:
    """Protenix / OpenDDE / Mol-1 share the proteinChain JSON dialect."""
    sequences: list[dict[str, Any]] = []
    for chain in job.chains:
        if chain.kind == "protein":
            entry: dict[str, Any] = {"sequence": chain.sequence, "count": 1}
            if msa_root is not None:
                chain_dir = msa_root / chain.id
                chain_dir.mkdir(parents=True, exist_ok=True)
                target = chain_dir / "non_pairing.a3m"
                if chain.msa:
                    if Path(chain.msa).resolve() != target.resolve():
                        shutil.copyfile(chain.msa, target)
                elif query_msa_fallback and not target.is_file():
                    target.write_text(f">query\n{chain.sequence}\n", encoding="utf-8")
                if target.is_file():
                    entry["msa"] = {"precomputed_msa_dir": str(chain_dir)}
            sequences.append({"proteinChain": entry})
        elif chain.kind == "dna":
            sequences.append({"dnaSequence": {"sequence": chain.sequence, "count": 1}})
        elif chain.kind == "rna":
            sequences.append({"rnaSequence": {"sequence": chain.sequence, "count": 1}})
        elif chain.kind == "ligand":
            if chain.smiles:
                sequences.append({"ligand": {"ligand": chain.smiles, "count": 1}})
            elif chain.ccd:
                sequences.append(
                    {"ligand": {"ligand": f"CCD_{chain.ccd[0]}", "count": 1}}
                )
    payload = [
        {
            "name": job.name,
            "sequences": sequences,
            "modelSeeds": [job.seed],
        }
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Runner adapters
# --------------------------------------------------------------------------- #

def _common_env(
    predictor_cfg: dict[str, Any], pythonpath: str | None = None
) -> dict[str, str]:
    return tool_env(
        {},
        extra=get(predictor_cfg, "env", {}) or {},
        clean_ld=bool(predictor_cfg.get("clean_ld_library_path", False)),
        pythonpath=pythonpath,
        path_prepend=get(predictor_cfg, "path_prepend"),
    )


def run_af3(job: Job, outdir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    predictor = get(cfg, "predictors.af3", {}) or {}
    input_path = write_af3_json(job, outdir / "input.json")
    command = [
        str(predictor["python"]),
        str(predictor["script"]),
        "--json_path", str(input_path),
        "--output_dir", str(outdir / "output"),
        "--model_dir", str(predictor["model_dir"]),
        "--run_data_pipeline=false",
        "--num_diffusion_samples", str(int(predictor.get("num_samples", 5))),
    ]
    if predictor.get("db_dir"):
        command += ["--db_dir", str(predictor["db_dir"])]
    command += [str(item) for item in predictor.get("extra_args", [])]
    result = run_command(
        command,
        env=_common_env(predictor),
        log_path=outdir / "run.log",
    )
    return {"returncode": result.returncode, "command": command}


def run_boltz2(job: Job, outdir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    predictor = get(cfg, "predictors.boltz2", {}) or {}
    input_path = write_boltz_yaml(job, outdir / "input.yaml", outdir / "msa")
    samples = int(predictor.get("num_samples", 5))
    launcher = predictor.get("launcher")
    if launcher:
        command = [str(item) for item in launcher]
        command += [
            "--input", str(input_path),
            "--out_dir", str(outdir / "output"),
            "--seed", str(job.seed),
            "--diffusion_samples", str(samples),
            "--output_format", "mmcif",
            "--write_full_pae",
        ]
    else:
        command = [
            str(predictor["binary"]),
            "predict", str(input_path),
            "--out_dir", str(outdir / "output"),
            "--cache", str(predictor.get("cache", "")),
            "--diffusion_samples", str(samples),
            "--output_format", "mmcif",
            "--write_full_pae",
            "--seed", str(job.seed),
        ]
    command += [str(item) for item in predictor.get("extra_args", [])]
    result = run_command(
        command,
        env=_common_env(predictor),
        log_path=outdir / "run.log",
    )
    return {"returncode": result.returncode, "command": command}


def _run_af3like(
    job: Job,
    outdir: Path,
    predictor: dict[str, Any],
    binary_key: str = "binary",
    model_flag: str = "--model_name",
) -> dict[str, Any]:
    msa_root = outdir / "msa"
    input_path = write_af3like_json(
        job,
        outdir / "input.json",
        msa_root,
        query_msa_fallback=bool(predictor.get("query_msa_fallback", False)),
    )
    launcher = predictor.get("launcher")
    if launcher:
        command = [str(item) for item in launcher]
    else:
        command = [str(predictor[binary_key]), "pred"]
    command += [
        "-i", str(input_path),
        "-o", str(outdir / "output"),
        "-s", str(job.seed),
        "-e", str(int(predictor.get("num_samples", 5))),
        model_flag, str(predictor.get("model_name", "")),
        "--use_msa", "True",
    ]
    command += [str(item) for item in predictor.get("extra_args", [])]
    result = run_command(
        command,
        env=_common_env(predictor),
        log_path=outdir / "run.log",
    )
    return {"returncode": result.returncode, "command": command}


def run_protenix(job: Job, outdir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_af3like(job, outdir, get(cfg, "predictors.protenix2", {}) or {})


def run_opendde(job: Job, outdir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_af3like(job, outdir, get(cfg, "predictors.opendde", {}) or {})


def run_mol1(job: Job, outdir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_af3like(job, outdir, get(cfg, "predictors.mol1", {}) or {})


def run_esmfold(job: Job, outdir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    predictor = get(cfg, "predictors.esmfold2", {}) or {}
    proteins = job.proteins
    if len(job.chains) != 1 or len(proteins) != 1:
        return {
            "returncode": 0,
            "skipped": "ESMFold only folds single-chain proteins; "
                       "use AF3/Boltz2/Protenix for complexes.",
        }
    fasta = outdir / "input.fasta"
    fasta.write_text(f">{job.name}\n{proteins[0].sequence}\n", encoding="utf-8")
    command = [
        str(predictor["python"]),
        str(predictor["script"]),
        "-i", str(fasta),
        "-o", str(outdir / "output"),
        "--weights", str(predictor.get("weights", "")),
    ]
    command += [str(item) for item in predictor.get("extra_args", [])]
    env = tool_env(
        {},
        extra=get(predictor, "env", {}) or {},
        clean_ld=bool(predictor.get("clean_ld_library_path", False)),
        pythonpath=get(predictor, "pythonpath"),
    )
    result = run_command(command, env=env, log_path=outdir / "run.log")
    return {"returncode": result.returncode, "command": command}


def run_esmfold2_overlay(
    job: Job, outdir: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Reserved adapter for the ESMFold2 + ESMC-6B RTX-50 overlay.

    The overlay consumes a directory of JSON inputs and writes
    ``structures/*.cif`` + ``confidences/*.npz``. It is disabled by default
    until ESMFOLD2_WEIGHTS and ESMFOLD2_ESMC are configured.
    """
    predictor = get(cfg, "predictors.esmfold2_overlay", {}) or {}
    if not predictor.get("enabled", False):
        return {"returncode": 0, "skipped": "esmfold2_overlay is disabled"}
    proteins = job.proteins
    if len(job.chains) != 1 or len(proteins) != 1:
        return {
            "returncode": 0,
            "skipped": "ESMFold2 overlay supports one protein chain per job",
        }
    repo = Path(str(predictor.get("repo", ""))).expanduser()
    script = repo / "scripts" / "esmfold2_50.sh"
    if not script.is_file():
        return {
            "returncode": -1,
            "error": f"ESMFold2 overlay script not found: {script}",
        }

    input_dir = outdir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    chain = proteins[0]
    record: dict[str, Any] = {"sequence": chain.sequence, "chains": ["A"]}
    if chain.msa:
        msa_dst = input_dir / "query.a3m"
        shutil.copyfile(chain.msa, msa_dst)
        record["msa"] = "query.a3m"
    (input_dir / f"{job.name}.json").write_text(
        json.dumps({"proteins": [record]}), encoding="utf-8"
    )

    command = [
        "bash",
        str(script),
        str(input_dir),
        str(outdir / "output"),
        "--mode",
        str(predictor.get("mode", "blackwell")),
    ]
    command += [str(item) for item in predictor.get("extra_args", [])]
    env_extra = dict(get(predictor, "env", {}) or {})
    if predictor.get("python"):
        env_extra["ESMFOLD2_PYTHON"] = str(predictor["python"])
    env = tool_env(
        {},
        extra=env_extra,
        clean_ld=bool(predictor.get("clean_ld_library_path", False)),
    )
    result = run_command(command, env=env, log_path=outdir / "run.log")
    return {"returncode": result.returncode, "command": command}


RUNNERS = {
    "af3": run_af3,
    "boltz2": run_boltz2,
    "protenix2": run_protenix,
    "opendde": run_opendde,
    "mol1": run_mol1,
    "esmfold2": run_esmfold,
    "esmfold2_overlay": run_esmfold2_overlay,
}


def run_jobs(
    cfg: dict[str, Any],
    jobs: list[Job],
    predictors: list[str],
    outroot: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    unknown = [name for name in predictors if name not in RUNNERS]
    if unknown:
        raise ValueError(f"unknown predictors: {unknown}")
    for job in jobs:
        for predictor_name in predictors:
            outdir = outroot / job.name / predictor_name
            outdir.mkdir(parents=True, exist_ok=True)
            try:
                payload = RUNNERS[predictor_name](job, outdir, cfg)
            except Exception as exc:  # keep the campaign running
                payload = {"returncode": -1, "error": str(exc)}
            payload.update(
                {"job": job.name, "predictor": predictor_name, "outdir": str(outdir)}
            )
            produced = any(
                path.suffix.lower() in {".cif", ".pdb"}
                for path in outdir.rglob("*")
                if path.is_file()
            )
            accepted = set(
                get(cfg, f"predictors.{predictor_name}.accept_returncodes", [0])
            )
            if produced and payload.get("returncode") in accepted:
                if payload.get("returncode") != 0:
                    payload["original_returncode"] = payload["returncode"]
                    payload["partial"] = True
                    payload["returncode"] = 0
            elif (
                payload.get("returncode") == 0
                and not produced
                and not payload.get("skipped")
            ):
                payload["returncode"] = -1
                payload["error"] = "predictor exited successfully but produced no structures"
            (outdir / "status.json").write_text(json.dumps(payload, indent=2))
            results.append(payload)
    return results
