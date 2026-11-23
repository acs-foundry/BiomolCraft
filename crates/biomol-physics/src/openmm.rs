#![forbid(unsafe_code)]

//! Tier-2 Opt-In GPU OpenMM (`CUDA`/`OpenCL`/`CPU`) Biomolecular Molecular Dynamics Bridge
//! and `GFN2-xTB` QM Active-Site Cluster Preparation Bridge.
//!
//! Keeps `BiomolCraft` 100% pure Rust (`#![forbid(unsafe_code)]`, `0 GB` bundled weights) while
//! seamlessly driving the local OpenMM GPU environment (`$CRAFT_DATA_DIR/envs/openmm/bin/python`)
//! on RTX 4080 CUDA GPUs for proteins, antibodies/VHH, enzymes, and RNA/ncRNA structures.

use std::path::{Path, PathBuf};
use std::process::Command;

use craftbio_report::craft_data_dir;
use serde::{Deserialize, Serialize};

/// Detected status of the local OpenMM GPU MD runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenMmEnvStatus {
    pub available: bool,
    pub python_path: Option<String>,
    pub openmm_version: Option<String>,
    pub platforms: Vec<String>,
    pub cuda_ready: bool,
}

/// Configuration for an OpenMM biomolecular MD run (`implicit_gbn2` or `explicit_tip3p`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenMmBiomolConfig {
    pub input_pdb: PathBuf,
    pub out_dir: PathBuf,
    /// `implicit_gbn2` (ultra-fast GPU implicit solvent) or `explicit_tip3p` (periodic box + 150 mM NaCl).
    pub solvent_model: String,
    pub production_ps: f64,
    pub temperature_k: f64,
    pub timestep_fs: f64,
    pub report_interval_ps: f64,
    /// `auto` | `CUDA` | `OpenCL` | `CPU`
    pub platform: String,
}

impl Default for OpenMmBiomolConfig {
    fn default() -> Self {
        Self {
            input_pdb: PathBuf::new(),
            out_dir: PathBuf::from("biomol_openmm_out"),
            solvent_model: "implicit_gbn2".to_string(),
            production_ps: 50.0,
            temperature_k: 300.0,
            timestep_fs: 2.0,
            report_interval_ps: 2.0,
            platform: "auto".to_string(),
        }
    }
}

/// Structured results returned by a live OpenMM GPU biomolecular MD run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenMmBiomolResult {
    pub platform_used: String,
    pub openmm_version: String,
    pub solvent_model: String,
    pub atom_count: usize,
    pub frames_saved: usize,
    pub initial_energy_kj: f64,
    pub minimized_energy_kj: f64,
    pub final_potential_kj: f64,
    pub mean_backbone_rmsd_a: f64,
    pub final_backbone_rmsd_a: f64,
    pub mean_rg_a: f64,
    pub elapsed_seconds: f64,
    pub topology_pdb_path: String,
    pub trajectory_dcd_path: String,
    pub final_pdb_path: String,
    pub rmsd_trace_a: Vec<f64>,
}

/// Locate the isolated CraftBio OpenMM Python binary.
#[must_use]
pub fn resolve_openmm_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CRAFT_OPENMM_PYTHON") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let candidate = craft_data_dir().join("envs/openmm/bin/python");
    if candidate.is_file() {
        return Some(candidate);
    }
    None
}

/// Probe OpenMM version and available hardware platforms (`CUDA`, `OpenCL`, `CPU`).
#[must_use]
pub fn check_openmm_biomol_env() -> OpenMmEnvStatus {
    let Some(py) = resolve_openmm_python() else {
        return OpenMmEnvStatus {
            available: false,
            python_path: None,
            openmm_version: None,
            platforms: Vec::new(),
            cuda_ready: false,
        };
    };
    let probe = "import json, openmm; print(json.dumps({'version': openmm.__version__, 'platforms': [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]}))";
    match Command::new(&py).args(["-c", probe]).output() {
        Ok(out) if out.status.success() => {
            let v: serde_json::Value =
                serde_json::from_slice(&out.stdout).unwrap_or(serde_json::Value::Null);
            let version = v
                .get("version")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            let platforms: Vec<String> = v
                .get("platforms")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let cuda_ready = platforms.iter().any(|p| p == "CUDA");
            OpenMmEnvStatus {
                available: version.is_some(),
                python_path: Some(py.display().to_string()),
                openmm_version: version,
                platforms,
                cuda_ready,
            }
        }
        _ => OpenMmEnvStatus {
            available: false,
            python_path: Some(py.display().to_string()),
            openmm_version: None,
            platforms: Vec::new(),
            cuda_ready: false,
        },
    }
}

const OPENMM_DRIVER_PY: &str = r#"
import json, math, os, sys, time
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = json.load(f)

t0 = time.time()
out_dir = cfg["out_dir"]
os.makedirs(out_dir, exist_ok=True)

# Clean PDB to standard polymer residues (proteins + RNA/DNA) so Amber14 builds cleanly without external ligands breaking topology
STD_RES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HID","HIE","HIP","CYX","ASH","GLH",
    "A","U","G","C","DA","DT","DG","DC","RA","RU","RG","RC"
}
clean_pdb_path = os.path.join(out_dir, "input_polymer.pdb")
with open(cfg["input_pdb"], "r") as fin, open(clean_pdb_path, "w") as fout:
    for line in fin:
        if line.startswith("ATOM  "):
            res = line[17:20].strip()
            alt = line[16:17]
            if res in STD_RES and alt in (" ", "A"):
                fout.write(line[:16] + " " + line[17:])
        elif line.startswith("TER"):
            fout.write("TER\n")
    fout.write("END\n")

pdb = app.PDBFile(clean_pdb_path)
solvent_model = cfg.get("solvent_model", "implicit_gbn2")
if solvent_model == "explicit_tip3p":
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
else:
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")

modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.deleteWater()
# Remove incomplete terminal atoms or add missing hydrogens cleanly
try:
    modeller.addHydrogens(ff, pH=7.4)
except Exception:
    # Fallback: add missing heavy atoms if pdbfixer is installed
    try:
        from pdbfixer import PDBFixer
        fixer = PDBFixer(filename=clean_pdb_path)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)
        modeller = app.Modeller(fixer.topology, fixer.positions)
    except Exception as e2:
        raise RuntimeError(f"Failed to add hydrogens: {e2}")

if solvent_model == "explicit_tip3p":
    modeller.addSolvent(ff, model="tip3p", padding=0.9*unit.nanometer, ionicStrength=0.15*unit.molar)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=1.5*unit.amu,
    )
    system.addForce(openmm.MonteCarloBarostat(1.0*unit.bar, cfg["temperature_k"]*unit.kelvin))
else:
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )

# Identify heavy solute atoms (non-water, non-ion, non-H) so topology PDB + DCD match craftbio-struct 1:1
WATER_IONS = {"HOH","WAT","NA","CL","K","MG","ZN","CA"}
solute_indices = [
    a.index
    for a in modeller.topology.atoms()
    if a.residue.name not in WATER_IONS and (a.element is None or a.element.symbol != "H")
]

# Build heavy-solute topology & write exact matching topology.pdb so BiomolCraft DCD loader matches 1:1
solute_mod = app.Modeller(modeller.topology, modeller.positions)
to_delete = [
    a
    for a in solute_mod.topology.atoms()
    if a.residue.name in WATER_IONS or (a.element is not None and a.element.symbol == "H")
]
if to_delete:
    solute_mod.delete(to_delete)

topo_pdb = os.path.join(out_dir, "openmm_topology.pdb")
with open(topo_pdb, "w") as f:
    app.PDBFile.writeFile(solute_mod.topology, solute_mod.positions, f, keepIds=True)

# Pick requested hardware platform (preferring CUDA -> OpenCL -> CPU)
req_plat = cfg.get("platform", "auto")
avail = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
if req_plat != "auto" and req_plat in avail:
    plat_name = req_plat
elif "CUDA" in avail:
    plat_name = "CUDA"
elif "OpenCL" in avail:
    plat_name = "OpenCL"
else:
    plat_name = "CPU"

platform = openmm.Platform.getPlatformByName(plat_name)
props = {"Precision": "mixed"} if plat_name in ("CUDA", "OpenCL") else {}

integrator = openmm.LangevinMiddleIntegrator(
    cfg["temperature_k"]*unit.kelvin,
    1.0/unit.picosecond,
    cfg["timestep_fs"]*unit.femtosecond,
)
sim = app.Simulation(modeller.topology, system, integrator, platform, props)
sim.context.setPositions(modeller.positions)

e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
sim.minimizeEnergy(maxIterations=400)
e_min = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
sim.context.setVelocitiesToTemperature(cfg["temperature_k"]*unit.kelvin, 20261122)

# Backbone guide atoms (CA or C4') within solute for exact RMSD & Rg tracking
guide_local = [i for i, a in enumerate(solute_mod.topology.atoms()) if a.name in ("CA", "C4'")]
if len(guide_local) < 3:
    guide_local = list(range(min(len(solute_indices), 200)))

def get_solute_angstrom():
    pos_nm = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    return pos_nm[solute_indices]

ref_xyz = get_solute_angstrom()
ref_g = ref_xyz[guide_local]
ref_g_c = ref_g - ref_g.mean(axis=0)

def kabsch_rmsd(P, Q):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, -1.0 if d < 0 else 1.0])
    R = Vt.T @ D @ U.T
    diff = (Pc @ R.T) - Qc
    return float(np.sqrt((diff * diff).sum() / len(P)))

dcd_path = os.path.join(out_dir, "openmm_traj.dcd")
total_steps = max(250, int(round(cfg["production_ps"] * 1000.0 / cfg["timestep_fs"])))
n_frames = max(10, min(60, int(round(cfg["production_ps"] / max(cfg["report_interval_ps"], 0.2)))))
step_chunk = max(10, total_steps // n_frames)

rmsd_trace = []
rg_trace = []
with open(dcd_path, "wb") as df:
    dcd = app.DCDFile(df, solute_mod.topology, cfg["timestep_fs"]*unit.femtosecond, interval=step_chunk)
    # Write frame 0 (minimized)
    dcd.writeModel(ref_xyz * unit.angstrom)
    rmsd_trace.append(0.0)
    rg0 = float(np.sqrt(((ref_g_c * ref_g_c).sum(axis=1)).mean()))
    rg_trace.append(rg0)
    for _ in range(n_frames):
        sim.step(step_chunk)
        cur_xyz = get_solute_angstrom()
        dcd.writeModel(cur_xyz * unit.angstrom)
        cur_g = cur_xyz[guide_local]
        rmsd_trace.append(round(kabsch_rmsd(cur_g, ref_g), 4))
        cur_gc = cur_g - cur_g.mean(axis=0)
        rg_trace.append(float(np.sqrt(((cur_gc * cur_gc).sum(axis=1)).mean())))

e_final = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
final_xyz = get_solute_angstrom()
final_pdb = os.path.join(out_dir, "openmm_final.pdb")
with open(final_pdb, "w") as f:
    app.PDBFile.writeFile(solute_mod.topology, final_xyz * unit.angstrom, f, keepIds=True)

res = {
    "platform_used": plat_name,
    "openmm_version": openmm.__version__,
    "solvent_model": solvent_model,
    "atom_count": len(solute_indices),
    "frames_saved": len(rmsd_trace),
    "initial_energy_kj": round(float(e0), 2),
    "minimized_energy_kj": round(float(e_min), 2),
    "final_potential_kj": round(float(e_final), 2),
    "mean_backbone_rmsd_a": round(float(np.mean(rmsd_trace)), 4),
    "final_backbone_rmsd_a": round(float(rmsd_trace[-1]), 4),
    "mean_rg_a": round(float(np.mean(rg_trace)), 3),
    "elapsed_seconds": round(time.time() - t0, 2),
    "topology_pdb_path": topo_pdb,
    "trajectory_dcd_path": dcd_path,
    "final_pdb_path": final_pdb,
    "rmsd_trace_a": rmsd_trace,
}
with open(os.path.join(out_dir, "openmm_metrics.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res))
"#;

/// Execute a real OpenMM GPU/CPU biomolecular MD run (`implicit_gbn2` or `explicit_tip3p`),
/// returning the parsed metrics + `.dcd` trajectory path ready to load into `BiomolCraft`.
pub fn run_openmm_biomol_md(cfg: &OpenMmBiomolConfig) -> Result<OpenMmBiomolResult, String> {
    let py = resolve_openmm_python().ok_or_else(|| {
        "OpenMM Python environment not found at $CRAFT_DATA_DIR/envs/openmm/bin/python".to_string()
    })?;
    std::fs::create_dir_all(&cfg.out_dir).map_err(|e| format!("create out_dir failed: {e}"))?;
    let driver_path = cfg.out_dir.join("_biomol_openmm_driver.py");
    let cfg_path = cfg.out_dir.join("_biomol_openmm_cfg.json");
    std::fs::write(&driver_path, OPENMM_DRIVER_PY)
        .map_err(|e| format!("write driver failed: {e}"))?;
    let cfg_json = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(&cfg_path, cfg_json).map_err(|e| format!("write cfg failed: {e}"))?;

    let out = Command::new(&py)
        .args([driver_path.as_os_str(), cfg_path.as_os_str()])
        .output()
        .map_err(|e| format!("failed to launch OpenMM python: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        return Err(format!(
            "OpenMM MD failed (exit {:?}): {stderr} {stdout}",
            out.status.code()
        ));
    }
    let metrics_path = cfg.out_dir.join("openmm_metrics.json");
    let raw = std::fs::read_to_string(&metrics_path)
        .map_err(|e| format!("read openmm_metrics.json failed: {e}"))?;
    serde_json::from_str(&raw).map_err(|e| format!("parse openmm_metrics.json failed: {e}"))
}

/// Probe whether `xtb` (GFN2-xTB semi-empirical QM for metalloprotein active sites) is installed.
#[must_use]
pub fn check_xtb_env() -> bool {
    Command::new("which")
        .arg("xtb")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Write an active-site `.xyz` cluster + `run_gfn2_xtb.sh` script for opt-in `GFN2-xTB` QM evaluation.
pub fn prepare_xtb_active_site_job(
    atoms: &[(&str, [f64; 3])],
    charge: i32,
    out_dir: &Path,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let xyz_path = out_dir.join("active_site.xyz");
    let mut xyz = format!(
        "{}\nBiomolCraft GFN2-xTB Active Site Cluster (charge={charge})\n",
        atoms.len()
    );
    for (el, p) in atoms {
        xyz.push_str(&format!(
            "{el:<2} {:>12.6} {:>12.6} {:>12.6}\n",
            p[0], p[1], p[2]
        ));
    }
    std::fs::write(&xyz_path, xyz).map_err(|e| e.to_string())?;
    let sh_path = out_dir.join("run_gfn2_xtb.sh");
    let sh = format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nxtb active_site.xyz --gfn 2 --alpb water --chrg {charge} --opt tight\n"
    );
    std::fs::write(&sh_path, sh).map_err(|e| e.to_string())?;
    Ok(sh_path)
}
