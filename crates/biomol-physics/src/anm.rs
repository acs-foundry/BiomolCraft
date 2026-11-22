#![forbid(unsafe_code)]

//! Anisotropic Network Model (ANM) Normal Mode Analysis (Atilgan *Biophys. J.* 2001)
//! in 100% pure safe Rust for proteins, antibodies, enzymes, and ncRNA/DNA complexes.
//!
//! Extracts the lowest-frequency non-trivial collective hinge/breathing modes (projecting out
//! the 6 rigid-body translations & rotations), computes theoretical per-residue RMSF/B-factors,
//! identifies collective hinge residues, and generates smooth animated multi-frame 3D trajectories.

use craftbio_scene::LineOverlay;
use craftbio_struct::{Biomolecule, ResidueClass, TrajectoryFrame, dist3};
use serde::{Deserialize, Serialize};

/// Single collective normal mode extracted from the ANM Hessian.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalMode {
    pub mode_index: usize,
    pub eigenvalue: f64,
    pub relative_frequency: f64,
    pub collectivity: f64,
    /// Per-guide-node 3D eigenvector `[dx, dy, dz]` (unit-norm over `3N`).
    pub displacements: Vec<[f64; 3]>,
}

/// Per-residue mobility summary from ANM slow modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueMobility {
    pub chain: char,
    pub res_seq: i32,
    pub res_name: String,
    pub rmsf_anm: f64,
    pub is_hinge: bool,
}

/// Complete ANM Normal Mode Analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnmReport {
    pub title: String,
    pub guide_nodes: usize,
    pub cutoff_a: f64,
    pub modes: Vec<NormalMode>,
    pub residue_mobility: Vec<ResidueMobility>,
    pub hinge_residues: Vec<String>,
    pub node_atom_indices: Vec<usize>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn normalize(v: &mut [f64]) -> f64 {
    let n = norm(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

fn orthogonalize_against(v: &mut [f64], basis: &[Vec<f64>]) {
    for b in basis {
        let proj = dot(v, b);
        for (vi, &bi) in v.iter_mut().zip(b.iter()) {
            *vi -= proj * bi;
        }
    }
}

/// Compute the 6 lowest non-trivial collective ANM normal modes for `mol`.
///
/// Uses polymer guide atoms (`CA` for proteins, `C4'/P` for RNA/DNA), smoothly subsampling
/// uniformly when a macromolecule has `> 160` residues so matrix operations stay under ~350 ms.
pub fn compute_anm_modes(
    mol: &Biomolecule,
    cutoff_a: f64,
    num_modes: usize,
) -> Result<AnmReport, String> {
    let mut all_guides: Vec<(usize, char, i32, String)> = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            if matches!(
                r.class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            ) && let Some(g) = r.guide_idx
            {
                all_guides.push((g, r.chain, r.res_seq, r.res_name.clone()));
            }
        }
    }
    if all_guides.len() < 4 {
        return Err("ANM requires at least 4 polymer guide atoms (CA / C4')".to_string());
    }

    // Subsample uniformly for very large assemblies so 3N x 3N stays interactive (< 0.4s)
    const MAX_NODES: usize = 160;
    let guides: Vec<(usize, char, i32, String)> = if all_guides.len() > MAX_NODES {
        let step = (all_guides.len() as f64) / (MAX_NODES as f64);
        (0..MAX_NODES)
            .map(|k| {
                let idx = ((k as f64) * step).floor() as usize;
                all_guides[idx.min(all_guides.len() - 1)].clone()
            })
            .collect()
    } else {
        all_guides
    };

    let n = guides.len();
    let dim = 3 * n;
    let coords: Vec<[f64; 3]> = guides.iter().map(|g| mol.atoms[g.0].coords).collect();

    // Build symmetric 3N x 3N ANM Hessian H (spring constant gamma = 1.0)
    // Store sparse neighbor blocks for ultra-fast mat-vec product!
    let mut blocks: Vec<Vec<(usize, [[f64; 3]; 3])>> = vec![Vec::new(); n];
    let mut diag: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; n];
    let c2 = cutoff_a * cutoff_a;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 > 1e-4 && r2 <= c2 {
                // Distance-weighted elastic spring (stronger along backbone, smooth decay)
                let w = if j == i + 1 { 2.5 } else { 1.0 } / r2;
                let off = [
                    [-w * dx * dx, -w * dx * dy, -w * dx * dz],
                    [-w * dy * dx, -w * dy * dy, -w * dy * dz],
                    [-w * dz * dx, -w * dz * dy, -w * dz * dz],
                ];
                blocks[i].push((j, off));
                blocks[j].push((i, off));
                for r in 0..3 {
                    for c in 0..3 {
                        diag[i][r][c] -= off[r][c];
                        diag[j][r][c] -= off[r][c];
                    }
                }
            }
        }
    }

    let matvec = |x: &[f64], y: &mut [f64]| {
        for i in 0..n {
            let xi = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
            let d = diag[i];
            let mut acc = [
                d[0][0] * xi[0] + d[0][1] * xi[1] + d[0][2] * xi[2],
                d[1][0] * xi[0] + d[1][1] * xi[1] + d[1][2] * xi[2],
                d[2][0] * xi[0] + d[2][1] * xi[1] + d[2][2] * xi[2],
            ];
            for &(j, blk) in &blocks[i] {
                let xj = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
                acc[0] += blk[0][0] * xj[0] + blk[0][1] * xj[1] + blk[0][2] * xj[2];
                acc[1] += blk[1][0] * xj[0] + blk[1][1] * xj[1] + blk[1][2] * xj[2];
                acc[2] += blk[2][0] * xj[0] + blk[2][1] * xj[1] + blk[2][2] * xj[2];
            }
            y[3 * i] = acc[0];
            y[3 * i + 1] = acc[1];
            y[3 * i + 2] = acc[2];
        }
    };

    // Estimate upper spectral bound lambda_max via Gershgorin / 18 power iterations
    let mut v_max = vec![0.0_f64; dim];
    for (k, val) in v_max.iter_mut().enumerate() {
        *val = ((k as f64) * 0.371).sin() + 0.5;
    }
    normalize(&mut v_max);
    let mut hv = vec![0.0_f64; dim];
    let mut lambda_max = 1.0_f64;
    for _ in 0..20 {
        matvec(&v_max, &mut hv);
        lambda_max = dot(&v_max, &hv).abs().max(1.0);
        let nrm = normalize(&mut hv);
        if nrm < 1e-12 {
            break;
        }
        v_max.copy_from_slice(&hv);
    }
    let shift = lambda_max * 1.15 + 1.0;

    // Build orthonormal basis for the 6 rigid-body modes (3 translations + 3 rotations)
    let mut center = [0.0_f64; 3];
    for c in &coords {
        center[0] += c[0];
        center[1] += c[1];
        center[2] += c[2];
    }
    for x in &mut center {
        *x /= n as f64;
    }
    let mut excluded: Vec<Vec<f64>> = Vec::with_capacity(6 + num_modes);
    for axis in 0..3 {
        let mut t = vec![0.0_f64; dim];
        for i in 0..n {
            t[3 * i + axis] = 1.0;
        }
        orthogonalize_against(&mut t, &excluded);
        if normalize(&mut t) > 1e-8 {
            excluded.push(t);
        }
    }
    // 3 rigid-body rotations around x, y, z axes
    for axis in 0..3 {
        let mut rot = vec![0.0_f64; dim];
        for i in 0..n {
            let r = [
                coords[i][0] - center[0],
                coords[i][1] - center[1],
                coords[i][2] - center[2],
            ];
            let cross = match axis {
                0 => [0.0, -r[2], r[1]],
                1 => [r[2], 0.0, -r[0]],
                _ => [-r[1], r[0], 0.0],
            };
            rot[3 * i] = cross[0];
            rot[3 * i + 1] = cross[1];
            rot[3 * i + 2] = cross[2];
        }
        orthogonalize_against(&mut rot, &excluded);
        if normalize(&mut rot) > 1e-8 {
            excluded.push(rot);
        }
    }

    // Extract k lowest non-trivial modes via shifted power iteration on M = shift*I - H
    let target_modes = num_modes.clamp(1, (dim.saturating_sub(6)).min(10));
    let mut modes = Vec::with_capacity(target_modes);
    for m_idx in 0..target_modes {
        let mut v = vec![0.0_f64; dim];
        for (i, vi) in v.iter_mut().enumerate().take(dim) {
            let seed = ((i + 1) as f64) * ((m_idx + 2) as f64) * 0.618_033_988_7;
            *vi = seed.sin() + 0.35 * (seed * 1.732).cos();
        }
        orthogonalize_against(&mut v, &excluded);
        if normalize(&mut v) < 1e-10 {
            break;
        }
        let mut next = vec![0.0_f64; dim];
        for _ in 0..220 {
            matvec(&v, &mut hv);
            for i in 0..dim {
                next[i] = shift * v[i] - hv[i];
            }
            orthogonalize_against(&mut next, &excluded);
            if normalize(&mut next) < 1e-12 {
                break;
            }
            let diff: f64 = v
                .iter()
                .zip(next.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            v.copy_from_slice(&next);
            if diff < 1e-6 {
                break;
            }
        }
        matvec(&v, &mut hv);
        let eig = dot(&v, &hv).max(1e-6);
        excluded.push(v.clone());

        let mut disps = Vec::with_capacity(n);
        let mut sq_mags = Vec::with_capacity(n);
        for i in 0..n {
            let d = [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
            sq_mags.push(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            disps.push(d);
        }
        // Bruschweiler collectivity kappa = (1/N) * exp(-sum p_i ln p_i)
        let mut entropy = 0.0_f64;
        for &p in &sq_mags {
            if p > 1e-12 {
                entropy -= p * p.ln();
            }
        }
        let collectivity = (entropy.exp() / (n as f64)).clamp(0.0, 1.0);
        modes.push(NormalMode {
            mode_index: m_idx + 1,
            eigenvalue: (eig * 10_000.0).round() / 10_000.0,
            relative_frequency: 1.0,
            collectivity: (collectivity * 1000.0).round() / 1000.0,
            displacements: disps,
        });
    }

    modes.sort_by(|a, b| a.eigenvalue.total_cmp(&b.eigenvalue));
    if let Some(first) = modes.first() {
        let base_freq = first.eigenvalue.sqrt().max(1e-6);
        for (idx, m) in modes.iter_mut().enumerate() {
            m.mode_index = idx + 1;
            m.relative_frequency = ((m.eigenvalue.sqrt() / base_freq) * 100.0).round() / 100.0;
        }
    }

    // Per-residue theoretical RMSF (sum_m ||u_m(i)||^2 / lambda_m) + hinge detection
    let mut raw_rmsf = vec![0.0_f64; n];
    for m in &modes {
        let inv_l = 1.0 / m.eigenvalue.max(1e-4);
        for (i, d) in m.displacements.iter().enumerate() {
            raw_rmsf[i] += inv_l * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
    }
    let max_rmsf = raw_rmsf
        .iter()
        .copied()
        .map(f64::sqrt)
        .fold(1e-6_f64, f64::max);
    let norm_rmsf: Vec<f64> = raw_rmsf
        .iter()
        .map(|&v| ((v.sqrt() / max_rmsf) * 2.5 * 100.0).round() / 100.0)
        .collect();

    let mut residue_mobility = Vec::with_capacity(n);
    let mut hinge_residues = Vec::new();
    for (i, (_, ch, rseq, rname)) in guides.iter().enumerate() {
        let is_hinge = if i > 1 && i + 2 < n {
            let cur = norm_rmsf[i];
            cur < 0.75
                && cur < norm_rmsf[i - 1]
                && cur < norm_rmsf[i + 1]
                && (norm_rmsf[i - 2] > cur + 0.25 || norm_rmsf[i + 2] > cur + 0.25)
        } else {
            false
        };
        if is_hinge && hinge_residues.len() < 12 {
            hinge_residues.push(format!("{ch}:{rname}{rseq}"));
        }
        residue_mobility.push(ResidueMobility {
            chain: *ch,
            res_seq: *rseq,
            res_name: rname.clone(),
            rmsf_anm: norm_rmsf[i],
            is_hinge,
        });
    }

    Ok(AnmReport {
        title: mol.title.clone(),
        guide_nodes: n,
        cutoff_a,
        modes,
        residue_mobility,
        hinge_residues,
        node_atom_indices: guides.into_iter().map(|g| g.0).collect(),
    })
}

/// Generate smooth sinusoidal multi-frame trajectory coordinates along `mode_1based` (1..=6)
/// and store them directly into `mol.frames` for real-time 60 FPS animation in `BiomolCraft`.
pub fn generate_anm_trajectory_frames(
    mol: &mut Biomolecule,
    report: &AnmReport,
    mode_1based: usize,
    amplitude_a: f64,
    num_frames: usize,
) -> Result<usize, String> {
    let mode = report
        .modes
        .get(mode_1based.saturating_sub(1))
        .ok_or_else(|| format!("ANM mode {mode_1based} not available"))?;
    let n_frames = num_frames.clamp(6, 60);
    let base_coords: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.coords).collect();
    let node_coords: Vec<[f64; 3]> = report
        .node_atom_indices
        .iter()
        .map(|&idx| base_coords[idx])
        .collect();

    // Scale eigenvector so maximum node displacement equals `amplitude_a`
    let max_d = mode
        .displacements
        .iter()
        .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
        .fold(1e-6_f64, f64::max);
    let scale = amplitude_a / max_d;

    // Interpolate smooth displacement vector onto every atom from nearest 3 guide nodes
    let mut atom_disp = vec![[0.0_f64; 3]; mol.atoms.len()];
    for (ai, &p) in base_coords.iter().enumerate() {
        let mut best: [(f64, usize); 3] = [(f64::INFINITY, 0); 3];
        for (ni, &nc) in node_coords.iter().enumerate() {
            let d = dist3(p, nc);
            if d < best[2].0 {
                best[2] = (d, ni);
                if best[2].0 < best[1].0 {
                    best.swap(1, 2);
                }
                if best[1].0 < best[0].0 {
                    best.swap(0, 1);
                }
            }
        }
        let mut w_sum = 0.0_f64;
        let mut disp = [0.0_f64; 3];
        for &(d, ni) in &best {
            if d.is_finite() {
                let w = (-0.08 * d * d).exp().max(1e-4);
                w_sum += w;
                let u = mode.displacements[ni];
                disp[0] += w * u[0] * scale;
                disp[1] += w * u[1] * scale;
                disp[2] += w * u[2] * scale;
            }
        }
        if w_sum > 1e-12 {
            atom_disp[ai] = [disp[0] / w_sum, disp[1] / w_sum, disp[2] / w_sum];
        }
    }

    let mut frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let phase = (2.0 * std::f64::consts::PI * (f as f64) / (n_frames as f64)).sin();
        let coords: Vec<[f64; 3]> = base_coords
            .iter()
            .zip(atom_disp.iter())
            .map(|(p, d)| {
                [
                    p[0] + phase * d[0],
                    p[1] + phase * d[1],
                    p[2] + phase * d[2],
                ]
            })
            .collect();
        frames.push(TrajectoryFrame {
            frame_index: f,
            time_ps: f as f64,
            coords,
        });
    }
    mol.frames = frames;
    mol.set_active_frame(0);
    Ok(n_frames)
}

/// Build 3D porcupine-plot vector overlays (`LineOverlay`) visualizing ANM mode displacements.
#[must_use]
pub fn anm_mode_porcupine_overlays(
    mol: &Biomolecule,
    report: &AnmReport,
    mode_1based: usize,
    arrow_scale_a: f64,
) -> Vec<LineOverlay> {
    let Some(mode) = report.modes.get(mode_1based.saturating_sub(1)) else {
        return Vec::new();
    };
    let max_d = mode
        .displacements
        .iter()
        .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
        .fold(1e-6_f64, f64::max);
    let scale = arrow_scale_a / max_d;
    let mut lines = Vec::new();
    for (i, &atom_idx) in report.node_atom_indices.iter().enumerate() {
        let p = mol.atoms[atom_idx].coords;
        let d = mode.displacements[i];
        let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * scale;
        if mag >= 0.85 {
            lines.push(LineOverlay {
                label: format!("ANM-M{mode_1based}"),
                from: p,
                to: [
                    p[0] + d[0] * scale,
                    p[1] + d[1] * scale,
                    p[2] + d[2] * scale,
                ],
                dashed: false,
                rgba: [0.98, 0.78, 0.16, 0.95],
            });
        }
    }
    lines
}
