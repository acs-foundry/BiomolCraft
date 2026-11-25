#![forbid(unsafe_code)]

//! General-Purpose 3D Protein Surface Aggregation & Solubility Rescue Engine
//! (Aggrescan3D-inspired structural projection over Shrake-Rupley SASA).
//!
//! Works on **any** protein structure (globular enzymes, kinases, scaffolds, antibodies):
//! 1. Combines experimental Aggrescan `a3v` amino-acid aggregation propensity with
//!    Shrake-Rupley relative solvent accessibility (`rSASA`) so buried hydrophobic cores
//!    are NOT penalized while solvent-exposed hydrophobic patches stand out clearly.
//! 2. Projects neighborhood propensity across a `10 Å` 3D spatial radius (`exp(-d/10)`).
//! 3. Supports an optional **Dynamic Mode** that weights flexible surface regions by
//!    normalized crystallographic B-factors.
//! 4. Clusters contiguous exposed hotspots (`score > 0.50`) into 3D aggregation patches
//!    and simulates single-point solubility-rescue mutations (`-> D / E / K / R`) with an
//!    explicit stability/function trade-off warning.

use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;

use craftbio_scene::SphereOverlay;
use craftbio_struct::{Biomolecule, ResidueClass, compute_sasa, dist3, max_residue_sasa};
use serde::{Deserialize, Serialize};

use crate::mutate_and_repack_best_rotamer;

/// Per-residue 3D structural aggregation score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueAggregationScore {
    pub chain: char,
    pub res_seq: i32,
    pub res_name: String,
    pub res_code: char,
    pub rsa: f64,
    pub b_factor: f64,
    pub intrinsic_a3v: f64,
    pub score_3d: f64,
    pub center: [f64; 3],
}

/// Contiguous 3D solvent-exposed aggregation-prone patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationPatch {
    pub patch_id: usize,
    pub residues: Vec<String>,
    pub residue_keys: Vec<(char, i32)>,
    pub peak_score: f64,
    pub mean_score: f64,
    pub total_score: f64,
    pub centroid: [f64; 3],
    pub radius_a: f64,
}

/// Complete 3D structural aggregation audit report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregation3DReport {
    pub title: String,
    pub dynamic_mode: bool,
    pub residue_scores: Vec<ResidueAggregationScore>,
    pub patches: Vec<AggregationPatch>,
    pub avg_score: f64,
    pub max_score: f64,
    pub min_score: f64,
    pub total_positive_score: f64,
    pub hotspot_count: usize,
}

/// Result of simulating a single-point solubility-rescue mutation on an exposed patch residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolubilityRescueReport {
    pub chain: char,
    pub res_seq: i32,
    pub wild_type: char,
    pub mutant: char,
    pub before_avg_score: f64,
    pub after_avg_score: f64,
    pub delta_avg_score: f64,
    pub before_max_score: f64,
    pub after_max_score: f64,
    pub before_res_score: f64,
    pub after_res_score: f64,
    pub clashes_introduced: usize,
    pub caveat: String,
}

/// Experimental Aggrescan `a3v` intrinsic aggregation propensity per amino acid
/// (Conchillo-Solé et al., *BMC Bioinformatics* 2007; Zambrano et al., *NAR* 2015).
#[must_use]
pub fn aggrescan_a3v(aa: char) -> f64 {
    match aa.to_ascii_uppercase() {
        'I' => 1.822,
        'F' => 1.754,
        'V' => 1.594,
        'L' => 1.380,
        'Y' => 1.159,
        'W' => 1.037,
        'M' => 0.622,
        'C' => 0.604,
        'A' => -0.036,
        'T' => -0.159,
        'S' => -0.294,
        'P' => -0.334,
        'G' => -0.535,
        'K' => -0.931,
        'H' => -1.033,
        'Q' => -1.231,
        'R' => -1.240,
        'N' => -1.302,
        'E' => -1.412,
        'D' => -1.836,
        _ => 0.0,
    }
}

/// Run Aggrescan3D-inspired 3D surface aggregation analysis on any protein structure.
#[must_use]
pub fn analyze_aggregation_3d(mol: &Biomolecule, dynamic_mode: bool) -> Aggregation3DReport {
    let sasa = compute_sasa(&mol.atoms, 1.4, 42);

    // Gather protein residues with relative SASA & sidechain/CA centroids
    struct ResInfo {
        chain: char,
        res_seq: i32,
        res_name: String,
        res_code: char,
        rsa: f64,
        bf: f64,
        a3v: f64,
        intrinsic_exposed: f64,
        center: [f64; 3],
    }

    let mut b_vals: Vec<f64> = mol
        .chains
        .iter()
        .flat_map(|c| c.residues.iter())
        .filter(|r| r.class == ResidueClass::Protein && r.mean_bfactor.is_finite())
        .map(|r| r.mean_bfactor)
        .collect();
    b_vals.sort_by(f64::total_cmp);
    let b_med = if b_vals.is_empty() {
        25.0
    } else {
        b_vals[b_vals.len() / 2]
    };
    let b_iqr = if b_vals.len() >= 4 {
        (b_vals[(b_vals.len() * 3) / 4] - b_vals[b_vals.len() / 4]).max(8.0)
    } else {
        15.0
    };

    let mut items = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            if r.class != ResidueClass::Protein || r.atom_indices.is_empty() {
                continue;
            }
            let res_sasa: f64 = r
                .atom_indices
                .iter()
                .filter_map(|&ai| sasa.get(ai).copied())
                .sum();
            let max_s = max_residue_sasa(r.code).max(60.0);
            let rsa = (res_sasa / max_s).clamp(0.0, 1.0);

            // Sidechain centroid (or CA if Gly)
            let mut sum = [0.0; 3];
            let mut cnt = 0.0;
            for &ai in &r.atom_indices {
                let a = &mol.atoms[ai];
                if !matches!(a.name.as_str(), "N" | "C" | "O") && a.element != "H" {
                    sum[0] += a.coords[0];
                    sum[1] += a.coords[1];
                    sum[2] += a.coords[2];
                    cnt += 1.0;
                }
            }
            if cnt < 1.0 {
                for &ai in &r.atom_indices {
                    sum[0] += mol.atoms[ai].coords[0];
                    sum[1] += mol.atoms[ai].coords[1];
                    sum[2] += mol.atoms[ai].coords[2];
                    cnt += 1.0;
                }
            }
            let center = [sum[0] / cnt, sum[1] / cnt, sum[2] / cnt];
            let a3v = aggrescan_a3v(r.code);
            let flex_mult = if dynamic_mode && a3v > 0.0 {
                let z = ((r.mean_bfactor - b_med) / b_iqr).clamp(0.0, 1.5);
                1.0 + 0.28 * z
            } else {
                1.0
            };
            // Strictly zero out deeply buried core residues (RSA < 0.05) so native core isn't flagged
            let intrinsic_exposed = if rsa < 0.05 {
                0.0
            } else {
                a3v * rsa.powf(0.65) * flex_mult
            };

            items.push(ResInfo {
                chain: ch.id,
                res_seq: r.res_seq,
                res_name: r.res_name.clone(),
                res_code: r.code,
                rsa,
                bf: r.mean_bfactor,
                a3v,
                intrinsic_exposed,
                center,
            });
        }
    }

    let n = items.len();
    let mut residue_scores = Vec::with_capacity(n);
    let mut sum_all = 0.0_f64;
    let mut max_score = f64::NEG_INFINITY;
    let mut min_score = f64::INFINITY;
    let mut total_pos = 0.0_f64;
    let mut hotspot_count = 0usize;

    for i in 0..n {
        let mut s = items[i].intrinsic_exposed;
        if items[i].rsa >= 0.05 {
            for j in 0..n {
                if i == j || items[j].rsa < 0.05 {
                    continue;
                }
                let d = dist3(items[i].center, items[j].center);
                if d <= 10.0 {
                    s += items[j].intrinsic_exposed * (-d / 10.0).exp() * 0.25;
                }
            }
        }
        let score_3d = (s * 1000.0).round() / 1000.0;
        sum_all += score_3d;
        max_score = max_score.max(score_3d);
        min_score = min_score.min(score_3d);
        if score_3d > 0.0 {
            total_pos += score_3d;
        }
        if score_3d > 0.50 {
            hotspot_count += 1;
        }
        residue_scores.push(ResidueAggregationScore {
            chain: items[i].chain,
            res_seq: items[i].res_seq,
            res_name: items[i].res_name.clone(),
            res_code: items[i].res_code,
            rsa: (items[i].rsa * 1000.0).round() / 1000.0,
            b_factor: (items[i].bf * 10.0).round() / 10.0,
            intrinsic_a3v: items[i].a3v,
            score_3d,
            center: items[i].center,
        });
    }

    // Cluster exposed aggregation hotspots (score_3d > 0.50, within 9.0 Å) into 3D patches
    let hot_indices: Vec<usize> = residue_scores
        .iter()
        .enumerate()
        .filter(|(_, r)| r.score_3d > 0.50)
        .map(|(i, _)| i)
        .collect();
    let mut visited = vec![false; hot_indices.len()];
    let mut patches = Vec::new();
    for hi in 0..hot_indices.len() {
        if visited[hi] {
            continue;
        }
        visited[hi] = true;
        let mut q = VecDeque::new();
        q.push_back(hi);
        let mut comp = Vec::new();
        while let Some(u) = q.pop_front() {
            comp.push(hot_indices[u]);
            let pu = residue_scores[hot_indices[u]].center;
            for vi in 0..hot_indices.len() {
                if !visited[vi] && dist3(pu, residue_scores[hot_indices[vi]].center) <= 9.0 {
                    visited[vi] = true;
                    q.push_back(vi);
                }
            }
        }
        let mut cent = [0.0; 3];
        let mut peak = f64::NEG_INFINITY;
        let mut tot = 0.0;
        let mut labels = Vec::new();
        let mut keys = Vec::new();
        for &ri in &comp {
            let r = &residue_scores[ri];
            cent[0] += r.center[0];
            cent[1] += r.center[1];
            cent[2] += r.center[2];
            peak = peak.max(r.score_3d);
            tot += r.score_3d;
            labels.push(format!("{}:{}{}", r.chain, r.res_name, r.res_seq));
            keys.push((r.chain, r.res_seq));
        }
        let k = comp.len().max(1) as f64;
        cent = [cent[0] / k, cent[1] / k, cent[2] / k];
        let mut rad = 2.0_f64;
        for &ri in &comp {
            rad = rad.max(dist3(cent, residue_scores[ri].center) + 1.5);
        }
        patches.push(AggregationPatch {
            patch_id: 0,
            residues: labels,
            residue_keys: keys,
            peak_score: (peak * 100.0).round() / 100.0,
            mean_score: ((tot / k) * 100.0).round() / 100.0,
            total_score: (tot * 100.0).round() / 100.0,
            centroid: cent,
            radius_a: (rad * 10.0).round() / 10.0,
        });
    }
    patches.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
    for (idx, p) in patches.iter_mut().enumerate() {
        p.patch_id = idx + 1;
    }

    let avg_score = if n > 0 {
        ((sum_all / n as f64) * 1000.0).round() / 1000.0
    } else {
        0.0
    };

    Aggregation3DReport {
        title: mol.title.clone(),
        dynamic_mode,
        residue_scores,
        patches,
        avg_score,
        max_score: if max_score.is_finite() { max_score } else { 0.0 },
        min_score: if min_score.is_finite() { min_score } else { 0.0 },
        total_positive_score: (total_pos * 100.0).round() / 100.0,
        hotspot_count,
    }
}

/// Visual palette (non-stick residue colors + 3D patch spheres) for an `Aggregation3DReport`.
pub struct AggregationVisuals {
    pub residue_colors: HashMap<(char, i32), [f32; 3]>,
    pub patch_spheres: Vec<SphereOverlay>,
}

/// Map an `Aggregation3DReport` to smooth blue (soluble) → white (neutral) → crimson
/// (aggregation-prone) per-residue RGB colors (for `residue_colors`, WITHOUT forcing sticks)
/// plus translucent 3D hotspot spheres over top surface aggregation patches.
#[must_use]
pub fn aggregation_visual_palette(rep: &Aggregation3DReport) -> AggregationVisuals {
    let mut colors = HashMap::with_capacity(rep.residue_scores.len());
    for r in &rep.residue_scores {
        let t = ((r.score_3d / 2.2) as f32).clamp(-1.0, 1.0);
        let col = if t >= 0.0 {
            // white -> red/crimson
            [0.96, 0.94 * (1.0 - 0.80 * t), 0.94 * (1.0 - 0.84 * t)]
        } else {
            // white -> soluble blue
            let u = -t;
            [0.92 * (1.0 - 0.72 * u), 0.94 * (1.0 - 0.42 * u), 0.98]
        };
        colors.insert((r.chain, r.res_seq), col);
    }

    let mut spheres = Vec::new();
    for p in rep.patches.iter().take(6) {
        spheres.push(SphereOverlay {
            label: format!(
                "AggPatch #{} (peak {:+.2}, {} res)",
                p.patch_id,
                p.peak_score,
                p.residues.len()
            ),
            category: "AggregationPatch".to_string(),
            coords: p.centroid,
            radius_a: p.radius_a.clamp(2.2, 6.5),
            rgba: [0.96, 0.24, 0.16, 0.32],
        });
    }
    AggregationVisuals {
        residue_colors: colors,
        patch_spheres: spheres,
    }
}

/// Simulate a single-point solubility-rescue mutation (e.g. exposed hydrophobic -> `D/E/K/R`)
/// without modifying `mol`, comparing global & local 3D aggregation scores before vs after.
pub fn simulate_solubility_rescue(
    mol: &Biomolecule,
    chain: char,
    res_seq: i32,
    target_aa: char,
    dynamic_mode: bool,
) -> Result<SolubilityRescueReport, crate::ModelError> {
    let before = analyze_aggregation_3d(mol, dynamic_mode);
    let before_res = before
        .residue_scores
        .iter()
        .find(|r| r.chain == chain && r.res_seq == res_seq)
        .map_or(0.0, |r| r.score_3d);

    let mut probe = mol.clone();
    let mut_rep = mutate_and_repack_best_rotamer(&mut probe, chain, res_seq, target_aa)?;
    let after = analyze_aggregation_3d(&probe, dynamic_mode);
    let after_res = after
        .residue_scores
        .iter()
        .find(|r| r.chain == chain && r.res_seq == res_seq)
        .map_or(0.0, |r| r.score_3d);

    let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
    Ok(SolubilityRescueReport {
        chain,
        res_seq,
        wild_type: mut_rep.wild_type,
        mutant: mut_rep.mutant,
        before_avg_score: before.avg_score,
        after_avg_score: after.avg_score,
        delta_avg_score: r3(after.avg_score - before.avg_score),
        before_max_score: before.max_score,
        after_max_score: after.max_score,
        before_res_score: before_res,
        after_res_score: after_res,
        clashes_introduced: mut_rep.clash_count,
        caveat: "Solubility mutations can trade off fold stability (ΔΔG) or active-site/PPI binding; verify with ΔΔG/MD and experimental nanoDSF + SEC-MALS before committing.".to_string(),
    })
}

/// Render a clean Markdown report of the 3D surface aggregation audit.
#[must_use]
pub fn render_aggregation_markdown(rep: &Aggregation3DReport) -> String {
    let mut md = String::new();
    let mode = if rep.dynamic_mode {
        "Dynamic (B-factor weighted)"
    } else {
        "Static (Crystal/Model)"
    };
    let _ = writeln!(md, "## 3D Surface Aggregation Audit — {}\n", rep.title);
    let _ = writeln!(
        md,
        "- **Mode:** `{mode}` | **Avg Score:** `{:+.3}` | **Max Score:** `{:+.3}` | **Total Positive:** `{:.2}` | **Exposed Hotspots (>0.50):** `{}`",
        rep.avg_score, rep.max_score, rep.total_positive_score, rep.hotspot_count
    );

    if !rep.patches.is_empty() {
        md.push_str("\n### Top 3D Exposed Aggregation Patches\n\n| Patch | Peak | Mean | Total | Radius (Å) | Residues |\n|---:|---:|---:|---:|---:|---|\n");
        for p in rep.patches.iter().take(6) {
            let preview: Vec<&str> = p.residues.iter().take(8).map(String::as_str).collect();
            let _ = writeln!(
                md,
                "| #{} | {:+.2} | {:+.2} | {:.2} | {:.1} | {} |",
                p.patch_id,
                p.peak_score,
                p.mean_score,
                p.total_score,
                p.radius_a,
                preview.join(", ")
            );
        }
    }

    let mut ranked: Vec<&ResidueAggregationScore> = rep
        .residue_scores
        .iter()
        .filter(|r| r.score_3d > 0.25)
        .collect();
    ranked.sort_by(|a, b| b.score_3d.total_cmp(&a.score_3d));
    if !ranked.is_empty() {
        md.push_str("\n### Highest-Scoring Exposed Residues (Candidates for Solubility Rescue)\n\n| Residue | Score 3D | rSASA | a3v | B-factor |\n|---|---:|---:|---:|---:|\n");
        for r in ranked.iter().take(12) {
            let _ = writeln!(
                md,
                "| `{}:{}{}` | {:+.3} | {:.2} | {:+.2} | {:.1} |",
                r.chain, r.res_name, r.res_seq, r.score_3d, r.rsa, r.intrinsic_a3v, r.b_factor
            );
        }
    }
    md
}
