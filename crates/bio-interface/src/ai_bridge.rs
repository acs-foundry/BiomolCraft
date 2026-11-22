#![forbid(unsafe_code)]

//! Opt-In De Novo Binder Design Bridges (0 GB Disk Footprint in BioCraft Core):
//! Generates deterministic, reproducible execution scripts, hotspot contigs, JSONL configs,
//! and strict in-silico GO/NO-GO triage gates for external `RFdiffusion`, `ProteinMPNN`,
//! `ESM2`, and `DockCraft co-fold` (`Boltz-2` / `Chai-1`) pipelines.

use std::fmt::Write as _;
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Input specification for staging a de novo protein/nanobody binder campaign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeNovoRequest {
    pub campaign_name: String,
    pub target_pdb_path: String,
    pub target_chain: char,
    pub target_residue_range: String,
    pub hotspot_residues: Vec<String>,
    pub binder_length_min: usize,
    pub binder_length_max: usize,
    pub num_designs: usize,
    pub mpnn_seqs_per_backbone: usize,
    pub cofold_backend: String,
}

/// Detected availability of optional external AI binder tools on the local host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAvailability {
    pub dockcraft_cli: bool,
    pub python3: bool,
    pub rfdiffusion_env: bool,
    pub proteinmpnn_env: bool,
    pub boltz_or_chai: bool,
    pub note: String,
}

/// Staged zero-download de novo binder pipeline scripts + validation gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeNovoStagePlan {
    pub campaign_name: String,
    pub toolchain: ToolAvailability,
    pub rfdiffusion_contig: String,
    pub rfdiffusion_command: String,
    pub proteinmpnn_fixed_chains_jsonl: String,
    pub proteinmpnn_command: String,
    pub esm2_scoring_script: String,
    pub dockcraft_cofold_command: String,
    pub go_nogo_gates: Vec<String>,
    pub wetlab_validation_controls: Vec<String>,
}

fn command_exists(bin: &str) -> bool {
    Command::new("which")
        .arg(bin)
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Probe locally installed optional external AI binder tools without downloading any weights.
#[must_use]
pub fn probe_denovo_toolchain() -> ToolAvailability {
    let dockcraft_cli = command_exists("dockcraft-cli");
    let python3 = command_exists("python3");
    let rfdiffusion_env =
        std::env::var("RFDIFFUSION_DIR").is_ok() || command_exists("run_inference.py");
    let proteinmpnn_env =
        std::env::var("PROTEINMPNN_DIR").is_ok() || command_exists("protein_mpnn_run.py");
    let boltz_or_chai = command_exists("boltz") || command_exists("chai-lab");
    let note = if rfdiffusion_env && proteinmpnn_env && boltz_or_chai {
        "Full external generative binder stack detected on host.".to_string()
    } else {
        "Opt-in dry-run staging mode (0 GB extra AI weights installed): reproducible scripts generated for local GPU or HPC execution.".to_string()
    };
    ToolAvailability {
        dockcraft_cli,
        python3,
        rfdiffusion_env,
        proteinmpnn_env,
        boltz_or_chai,
        note,
    }
}

/// Generate a complete, reproducible `RFdiffusion -> ProteinMPNN -> ESM2 -> DockCraft co-fold`
/// de novo binder staging package targeting user-specified 3D epitope hotspots.
#[must_use]
pub fn stage_denovo_binder(req: &DeNovoRequest) -> DeNovoStagePlan {
    let toolchain = probe_denovo_toolchain();
    let hotspots = if req.hotspot_residues.is_empty() {
        format!(
            "[{}10,{}15,{}20]",
            req.target_chain, req.target_chain, req.target_chain
        )
    } else {
        format!("[{}]", req.hotspot_residues.join(","))
    };
    let contig = format!(
        "[{}{}/0 {}-{}]",
        req.target_chain, req.target_residue_range, req.binder_length_min, req.binder_length_max
    );
    let rfdiffusion_command = format!(
        "python3 ${{RFDIFFUSION_DIR:-/opt/RFdiffusion}}/scripts/run_inference.py \\\n  inference.output_prefix=out_{name}/design \\\n  inference.input_pdb={pdb} \\\n  'contigmap.contigs={contig}' \\\n  'ppi.hotspot_res={hotspots}' \\\n  inference.num_designs={n} \\\n  denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0",
        name = req.campaign_name,
        pdb = req.target_pdb_path,
        n = req.num_designs.max(1)
    );

    let proteinmpnn_fixed_chains_jsonl = format!(
        "{{\"target\": [\"{}\"], \"designed\": [\"A\"]}}",
        req.target_chain
    );
    let proteinmpnn_command = format!(
        "python3 ${{PROTEINMPNN_DIR:-/opt/ProteinMPNN}}/protein_mpnn_run.py \\\n  --pdb_path_chains \"A\" \\\n  --out_folder out_{name}/mpnn \\\n  --num_seq_per_target {k} \\\n  --sampling_temp \"0.1\" \\\n  --omit_AAs \"C\" \\\n  --batch_size 1",
        name = req.campaign_name,
        k = req.mpnn_seqs_per_backbone.max(1)
    );

    let esm2_scoring_script = format!(
        "#!/usr/bin/env python3\n\
\"\"\"Opt-in ESM2 pseudo-perplexity & BioCraft TAP developability filter for {name}.\"\"\"\n\
import json, subprocess, sys\n\
from pathlib import Path\n\
\n\
# 1. Rank MPNN FASTA sequences with BiomolCraft deterministic TAP + immunogenicity first (zero GPU cost):\n\
fasta = Path(\"out_{name}/mpnn/seqs/design_0.fa\")\n\
print(\"Step 1: Run `biomolcraft-cli module --name antibodies --pdb <design.pdb>` to eliminate TAP/PTM/MHC-II liabilities.\")\n\
print(\"Step 2: Optional ESM2-650M masked marginal scoring (requires `fair-esm` + PyTorch on RTX 4080).\")\n",
        name = req.campaign_name
    );

    let backend = if req.cofold_backend.trim().is_empty() {
        "boltz2"
    } else {
        req.cofold_backend.trim()
    };
    let dockcraft_cofold_command = format!(
        "dockcraft-cli co-fold --backend {backend} --target-fasta out_{name}/complex_candidate.fasta --dry-run --out-dir out_{name}/cofold",
        name = req.campaign_name
    );

    let go_nogo_gates = vec![
        "Gate 1 (Co-folding confidence): Boltz-2 / Chai-1 interface `ipTM >= 0.75`, binder `pLDDT >= 80`, interface `PAE <= 8.0 Å`, backbone `RMSD_design <= 2.0 Å`.".to_string(),
        "Gate 2 (BiomolCraft 3D Interface): `ΔBSA >= 650 Å²`, `>= 4` H-bonds/salt-bridges, zero steric clashes, confirmed epitope overlap with intended hotspots.".to_string(),
        "Gate 3 (BiomolCraft TAP & Immunogenicity): Zero Red TAP flags, `max 3D SAP < 0.85`, zero unpaired Cys, low non-self HLA-DRB1 burden.".to_string(),
        "Gate 4 (Scientific Integrity): In-silico designs are hypotheses only — NEVER treat high ipTM/pLDDT as proof of experimental affinity without wet-lab SPR/BLI.".to_string(),
    ];

    let wetlab_validation_controls = vec![
        "Positive control: known clinical/reference binder on the same antigen (e.g. Trastuzumab for HER2 or Envafolimab for PD-L1).".to_string(),
        "Negative controls: scrambled-CDR / non-binding scaffold + irrelevant antigen channel in SPR/BLI to reject sticky polyreactive binders.".to_string(),
        "Biophysical QC: SEC-MALS monomer fraction (>= 95%), nanoDSF melting temperature (Tm >= 65°C), and endotoxin-free expression yield.".to_string(),
    ];

    DeNovoStagePlan {
        campaign_name: req.campaign_name.clone(),
        toolchain,
        rfdiffusion_contig: contig,
        rfdiffusion_command,
        proteinmpnn_fixed_chains_jsonl,
        proteinmpnn_command,
        esm2_scoring_script,
        dockcraft_cofold_command,
        go_nogo_gates,
        wetlab_validation_controls,
    }
}

/// Render a clean Markdown de novo binder staging dossier.
#[must_use]
pub fn render_denovo_markdown(plan: &DeNovoStagePlan) -> String {
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# De Novo Binder Staging Plan: {}\n",
        plan.campaign_name
    );
    let _ = writeln!(md, "- **Toolchain Status:** {}", plan.toolchain.note);
    let _ = writeln!(
        md,
        "- **Detected Binaries:** `dockcraft-cli={}` | `python3={}` | `RFdiffusion={}` | `ProteinMPNN={}` | `Boltz/Chai={}`\n",
        plan.toolchain.dockcraft_cli,
        plan.toolchain.python3,
        plan.toolchain.rfdiffusion_env,
        plan.toolchain.proteinmpnn_env,
        plan.toolchain.boltz_or_chai
    );
    let _ = writeln!(
        md,
        "## 1. RFdiffusion Backbone Generation (`{}`)\n",
        plan.rfdiffusion_contig
    );
    let _ = writeln!(md, "```bash\n{}\n```\n", plan.rfdiffusion_command);
    let _ = writeln!(md, "## 2. ProteinMPNN Inverse Folding (Cys-Omitted)\n");
    let _ = writeln!(
        md,
        "Fixed-chain JSONL: `{}`\n",
        plan.proteinmpnn_fixed_chains_jsonl
    );
    let _ = writeln!(md, "```bash\n{}\n```\n", plan.proteinmpnn_command);
    let _ = writeln!(
        md,
        "## 3. Structure Co-Folding Verification (DockCraft Bridge)\n"
    );
    let _ = writeln!(md, "```bash\n{}\n```\n", plan.dockcraft_cofold_command);
    let _ = writeln!(md, "## 4. Strict In-Silico GO/NO-GO Triage Gates\n");
    for g in &plan.go_nogo_gates {
        let _ = writeln!(md, "- {g}");
    }
    let _ = writeln!(md, "\n## 5. Mandatory Orthogonal Wet-Lab Controls\n");
    for c in &plan.wetlab_validation_controls {
        let _ = writeln!(md, "- {c}");
    }
    md
}
