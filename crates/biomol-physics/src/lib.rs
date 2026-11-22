#![forbid(unsafe_code)]

//! `biomol-physics` — Hybrid 2-Tier Biomolecular Physics Engine for `BiomolCraft`:
//! - **Tier 1 (Pure Rust, 0 external deps):** ANM/GNM Normal Modes + animation frames + porcupine vectors,
//!   Still GB/SA solvation & interface `ΔG_bind`, 3D Electrostatic Dipole Moment, and Local Langevin MD.
//! - **Tier 2 (Opt-in GPU RTX 4080):** Live OpenMM 8.6 CUDA MD bridge (Proteins/Antibodies/Enzymes/RNA)
//!   with binary `.dcd` export + `GFN2-xTB` QM active-site cluster exporter.

pub mod anm;
pub mod gbsa;
pub mod md_rust;
pub mod openmm;

use std::fmt::Write as _;

pub use anm::{
    AnmReport, NormalMode, ResidueMobility, anm_mode_porcupine_overlays, compute_anm_modes,
    generate_anm_trajectory_frames,
};
use craftbio_report::escape_markup;
pub use gbsa::{GbSaEnergyTerms, GbSaReport, compute_gbsa_energy};
pub use md_rust::{LocalMdReport, MdTracePoint, run_local_langevin_md};
pub use openmm::{
    OpenMmBiomolConfig, OpenMmBiomolResult, OpenMmEnvStatus, check_openmm_biomol_env,
    check_xtb_env, prepare_xtb_active_site_job, resolve_openmm_python, run_openmm_biomol_md,
};

/// Render a concise Markdown summary of ANM Normal Modes.
#[must_use]
pub fn render_anm_markdown(rep: &AnmReport) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# ANM Normal Mode Analysis: {}\n", rep.title);
    let _ = writeln!(
        md,
        "- **Coarse Polymer Nodes:** `{}` (`CA`/`C4'`, cutoff `{:.1} Å`)",
        rep.guide_nodes, rep.cutoff_a
    );
    if !rep.hinge_residues.is_empty() {
        let _ = writeln!(
            md,
            "- **Detected Collective Hinge Residues:** `{}`",
            rep.hinge_residues.join(", ")
        );
    }
    md.push_str("\n| Mode | Eigenvalue (λ) | Rel. Frequency (ω/ω₁) | Collectivity (κ) |\n|---:|---:|---:|---:|\n");
    for m in &rep.modes {
        let _ = writeln!(
            md,
            "| Mode {} | {:.4} | {:.2}× | {:.3} |",
            m.mode_index, m.eigenvalue, m.relative_frequency, m.collectivity
        );
    }
    md
}

/// Render a concise Markdown summary of GB/SA solvation, Dipole Moment & `ΔG_bind`.
#[must_use]
pub fn render_gbsa_markdown(rep: &GbSaReport) -> String {
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# GB/SA Solvation & Electrostatic Dipole: {}\n",
        rep.title
    );
    let _ = writeln!(
        md,
        "- **Net Charge:** `{:+.2} e` | **3D Dipole Moment:** **`{:.2} D`** (`[{:.1}, {:.1}, {:.1}]`)",
        rep.dipole.net_charge_e,
        rep.dipole.dipole_magnitude_debye,
        rep.dipole.dipole_vector_debye[0],
        rep.dipole.dipole_vector_debye[1],
        rep.dipole.dipole_vector_debye[2],
    );
    let _ = writeln!(
        md,
        "- **Complex Energy (kcal/mol):** Total `{:.2}` (Coul `{:.2}`, GB `{:.2}`, vdW `{:.2}`, SA `{:.2}`, SASA `{:.1} Å²`)",
        rep.complex_energy.total_kcal,
        rep.complex_energy.e_coulomb_kcal,
        rep.complex_energy.e_gb_polar_kcal,
        rep.complex_energy.e_vdw_kcal,
        rep.complex_energy.e_sa_nonpolar_kcal,
        rep.complex_energy.total_sasa_a2
    );
    if let Some(dg) = rep.delta_g_bind_kcal {
        let _ = writeln!(
            md,
            "\n## Single-Trajectory Interface Binding Free Energy (`{:?}` vs `{:?}`)\n",
            rep.interface_chains_a, rep.interface_chains_b
        );
        let _ = writeln!(
            md,
            "- **Estimated `ΔG_bind` (GB/SA-lite):** **`{dg:.2} kcal/mol`** (ΔE_elec+GB `{:.2}`, ΔE_vdW `{:.2}`, ΔE_SA `{:.2}`)",
            rep.delta_e_elec_gb_kcal.unwrap_or(0.0),
            rep.delta_e_vdw_kcal.unwrap_or(0.0),
            rep.delta_e_sa_kcal.unwrap_or(0.0),
        );
        if !rep.top_interface_hotspots.is_empty() {
            md.push_str("\n### Top Interface Energy Contributors\n\n");
            for (res, e) in rep.top_interface_hotspots.iter().take(6) {
                let _ = writeln!(md, "- `{res}`: `{e:+.2} kcal/mol`");
            }
        }
    }
    md
}

/// Render an XSS-safe SVG polyline chart for MD RMSD traces (`Å` vs frame).
#[must_use]
pub fn render_rmsd_trace_svg(title: &str, rmsd_values: &[f64]) -> String {
    let w = 640.0_f64;
    let h = 240.0_f64;
    let pad_l = 52.0_f64;
    let pad_r = 24.0_f64;
    let pad_t = 34.0_f64;
    let pad_b = 34.0_f64;
    let max_y = rmsd_values.iter().copied().fold(1.0_f64, f64::max) * 1.15;
    let n = rmsd_values.len().max(2);
    let mut pts = String::new();
    for (i, &val) in rmsd_values.iter().enumerate() {
        let x = pad_l + (i as f64 / ((n - 1) as f64)) * (w - pad_l - pad_r);
        let y = (h - pad_b) - (val / max_y) * (h - pad_t - pad_b);
        let _ = write!(pts, "{x:.1},{y:.1} ");
    }
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"#0f172a\" rx=\"8\"/>\
         <text x=\"{pad_l}\" y=\"22\" fill=\"#e2e8f0\" font-family=\"monospace\" font-size=\"13\" font-weight=\"bold\">{}</text>\
         <line x1=\"{pad_l}\" y1=\"{pad_t}\" x2=\"{pad_l}\" y2=\"{}\" stroke=\"#475569\" stroke-width=\"1.2\"/>\
         <line x1=\"{pad_l}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#475569\" stroke-width=\"1.2\"/>\
         <text x=\"8\" y=\"{}\" fill=\"#94a3b8\" font-family=\"monospace\" font-size=\"10\">{max_y:.2} A</text>\
         <text x=\"14\" y=\"{}\" fill=\"#94a3b8\" font-family=\"monospace\" font-size=\"10\">0.0 A</text>\
         <polyline fill=\"none\" stroke=\"#38bdf8\" stroke-width=\"2.4\" points=\"{pts}\"/>\
         </svg>",
        escape_markup(title),
        h - pad_b,
        h - pad_b,
        w - pad_r,
        h - pad_b,
        pad_t + 8.0,
        h - pad_b
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use craftbio_struct::parse_pdb;

    #[test]
    fn computes_anm_modes_gbsa_dipole_and_local_md() -> Result<(), Box<dyn std::error::Error>> {
        let mut pdb = String::new();
        for i in 1..=12 {
            let x = (i as f64) * 3.8;
            let y = ((i as f64) * 0.6).sin() * 2.5;
            let z = ((i as f64) * 0.4).cos() * 1.5;
            let res = if i % 4 == 0 {
                "ASP"
            } else if i % 5 == 0 {
                "LYS"
            } else {
                "ALA"
            };
            let _ = writeln!(
                pdb,
                "ATOM  {:>5}  CA  {res} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 85.00           C",
                i, i, x, y, z
            );
        }
        pdb.push_str("END\n");
        let mut mol = parse_pdb("phys_test", &pdb)?;

        let anm = compute_anm_modes(&mol, 13.0, 6)?;
        assert_eq!(anm.modes.len(), 6);
        assert!(anm.modes[0].eigenvalue > 0.0);
        let n_f = generate_anm_trajectory_frames(&mut mol, &anm, 1, 2.0, 12)?;
        assert_eq!(n_f, 12);
        assert_eq!(mol.frames.len(), 12);

        let gbsa = compute_gbsa_energy(&mol, None, None);
        assert!(gbsa.complex_energy.total_sasa_a2 > 50.0);

        let md = run_local_langevin_md(&mut mol, &[], 30, 5, 300.0, 1.5, 42)?;
        assert!(md.frames_recorded >= 5);
        assert!(md.final_rmsd_a >= 0.0);

        let svg = render_rmsd_trace_svg("Test <RMSD>", &[0.0, 0.3, 0.5]);
        assert!(svg.contains("&lt;RMSD&gt;"));
        Ok(())
    }
}
