#![forbid(unsafe_code)]

//! Pure-Rust Local Langevin / Velocity-Verlet Molecular Dynamics (`0` external dependencies)
//! with deterministic Box-Muller thermostat, covalent bond springs, soft-core nonbonded terms,
//! optional harmonic position restraints, and live trajectory frame recording.

use craftbio_struct::{Biomolecule, TrajectoryFrame, dist3, vdw_radius};
use serde::{Deserialize, Serialize};

/// Per-frame trace point recorded during local Langevin MD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdTracePoint {
    pub frame: usize,
    pub time_ps: f64,
    pub potential_kcal: f64,
    pub temperature_k: f64,
    pub rmsd_a: f64,
}

/// Summary report from pure-Rust local Langevin MD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalMdReport {
    pub mobile_atoms: usize,
    pub total_steps: usize,
    pub frames_recorded: usize,
    pub target_temp_k: f64,
    pub final_rmsd_a: f64,
    pub mean_potential_kcal: f64,
    pub trace: Vec<MdTracePoint>,
}

struct XorShift64(u64);
impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Run deterministic pure-Rust local Langevin MD on `mobile_mask` (or all atoms if small)
/// and populate `mol.frames` with trajectory snapshots.
pub fn run_local_langevin_md(
    mol: &mut Biomolecule,
    mobile_mask: &[bool],
    steps: usize,
    record_every: usize,
    temperature_k: f64,
    restraint_k: f64,
    seed: u64,
) -> Result<LocalMdReport, String> {
    if mol.atoms.is_empty() {
        return Err("cannot run MD on empty biomolecule".to_string());
    }
    let n = mol.atoms.len();
    let mut mobile: Vec<usize> = mobile_mask
        .iter()
        .enumerate()
        .filter(|&(i, &sel)| sel && i < n)
        .map(|(i, _)| i)
        .collect();
    if mobile.is_empty() {
        // Default: take up to 500 atoms (all of small structures like 1Y26/5JDS or first active region)
        mobile = (0..n.min(500)).collect();
    } else if mobile.len() > 700 {
        mobile.truncate(700);
    }
    let mut is_mobile = vec![false; n];
    for &i in &mobile {
        is_mobile[i] = true;
    }

    let ref_coords: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.coords).collect();
    let mut pos = ref_coords.clone();
    let mut vel = vec![[0.0_f64; 3]; n];
    let mut forces = vec![[0.0_f64; 3]; n];
    let radii: Vec<f64> = mol.atoms.iter().map(|a| vdw_radius(&a.element)).collect();

    // Bonds touching mobile atoms with equilibrium lengths from initial geometry
    let bonds: Vec<(usize, usize, f64)> = mol
        .bonds
        .iter()
        .filter(|b| is_mobile[b.a] || is_mobile[b.b])
        .map(|b| {
            (
                b.a,
                b.b,
                dist3(ref_coords[b.a], ref_coords[b.b]).clamp(0.95, 2.5),
            )
        })
        .collect();

    // Neighbor list within 6.5 A of mobile atoms
    let mut neighbors: Vec<(usize, usize)> = Vec::new();
    for &i in &mobile {
        for j in 0..n {
            if j == i || (is_mobile[j] && j < i) {
                continue;
            }
            let d = dist3(ref_coords[i], ref_coords[j]);
            if d < 6.5 && d > 1.75 {
                neighbors.push((i, j));
            }
        }
    }

    let mut rng = XorShift64(seed.max(1));
    let kb_t = 0.001_987_204 * temperature_k.clamp(10.0, 500.0); // kcal/mol
    let mass = 12.0_f64; // effective heavy-atom amu
    let dt = 0.002_f64; // ps (scaled unit)
    let gamma = 2.0_f64; // Langevin friction
    let sigma_v = (kb_t / mass).sqrt();
    for &i in &mobile {
        vel[i] = [
            rng.next_normal() * sigma_v,
            rng.next_normal() * sigma_v,
            rng.next_normal() * sigma_v,
        ];
    }

    let steps = steps.clamp(10, 600);
    let stride = record_every.max(steps / 30).max(1);
    let mut frames = Vec::new();
    let mut trace = Vec::new();
    let mut pot_sum = 0.0_f64;

    for step in 0..=steps {
        for &i in &mobile {
            forces[i] = [0.0; 3];
        }
        let mut e_pot = 0.0_f64;

        // 1. Harmonic covalent bond springs
        for &(a, b, r0) in &bonds {
            let dx = pos[b][0] - pos[a][0];
            let dy = pos[b][1] - pos[a][1];
            let dz = pos[b][2] - pos[a][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-4);
            let dr = d - r0;
            let k_b = 180.0_f64;
            e_pot += 0.5 * k_b * dr * dr;
            let fmag = (k_b * dr / d).clamp(-120.0, 120.0);
            if is_mobile[a] {
                forces[a][0] += fmag * dx;
                forces[a][1] += fmag * dy;
                forces[a][2] += fmag * dz;
            }
            if is_mobile[b] {
                forces[b][0] -= fmag * dx;
                forces[b][1] -= fmag * dy;
                forces[b][2] -= fmag * dz;
            }
        }

        // 2. Soft-core LJ + Debye-Huckel electrostatics
        for &(a, b) in &neighbors {
            let dx = pos[b][0] - pos[a][0];
            let dy = pos[b][1] - pos[a][1];
            let dz = pos[b][2] - pos[a][2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > 49.0 {
                continue;
            }
            let d = d2.sqrt().max(0.8);
            let r_min = 0.85 * (radii[a] + radii[b]);
            if d < r_min {
                let overlap = r_min - d;
                let k_rep = 28.0_f64;
                e_pot += 0.5 * k_rep * overlap * overlap;
                let fmag = (k_rep * overlap / d).clamp(0.0, 80.0);
                if is_mobile[a] {
                    forces[a][0] -= fmag * dx;
                    forces[a][1] -= fmag * dy;
                    forces[a][2] -= fmag * dz;
                }
                if is_mobile[b] {
                    forces[b][0] += fmag * dx;
                    forces[b][1] += fmag * dy;
                    forces[b][2] += fmag * dz;
                }
            }
            let qa = mol.atoms[a].formal_charge;
            let qb = mol.atoms[b].formal_charge;
            if qa.abs() > 1e-4 && qb.abs() > 1e-4 {
                let c_e = 332.0 * qa * qb / (24.0 * d.max(1.8));
                e_pot += c_e;
                let fmag = (-c_e / (d * d)).clamp(-25.0, 25.0);
                if is_mobile[a] {
                    forces[a][0] += fmag * dx;
                    forces[a][1] += fmag * dy;
                    forces[a][2] += fmag * dz;
                }
                if is_mobile[b] {
                    forces[b][0] -= fmag * dx;
                    forces[b][1] -= fmag * dy;
                    forces[b][2] -= fmag * dz;
                }
            }
        }

        // 3. Harmonic position tether (ISOLDE/equilibration restraint)
        if restraint_k > 0.0 {
            for &i in &mobile {
                let dx = pos[i][0] - ref_coords[i][0];
                let dy = pos[i][1] - ref_coords[i][1];
                let dz = pos[i][2] - ref_coords[i][2];
                e_pot += 0.5 * restraint_k * (dx * dx + dy * dy + dz * dz);
                forces[i][0] -= restraint_k * dx;
                forces[i][1] -= restraint_k * dy;
                forces[i][2] -= restraint_k * dz;
            }
        }

        // BAOAB-style Langevin velocity/position update
        let damp = (-gamma * dt).exp();
        let noise_scale = ((1.0 - damp * damp) * kb_t / mass).sqrt();
        let mut ke = 0.0_f64;
        for &i in &mobile {
            for k in 0..3 {
                let a_k = (forces[i][k] / mass).clamp(-40.0, 40.0);
                let v_half = vel[i][k] + 0.5 * dt * a_k;
                let v_therm = damp * v_half + noise_scale * rng.next_normal();
                let disp = (dt * v_therm).clamp(-0.18, 0.18);
                pos[i][k] += disp;
                vel[i][k] = v_therm;
                ke += 0.5 * mass * vel[i][k] * vel[i][k];
            }
        }

        if step % stride == 0 || step == steps {
            let temp_inst = (2.0 * ke) / (3.0 * (mobile.len() as f64) * 0.001_987_204);
            let sq_disp: f64 = mobile
                .iter()
                .map(|&i| {
                    let d = dist3(pos[i], ref_coords[i]);
                    d * d
                })
                .sum();
            let rmsd = (sq_disp / (mobile.len() as f64)).sqrt();
            pot_sum += e_pot;
            let f_idx = frames.len();
            let time_ps = ((step as f64) * dt * 1000.0).round() / 1000.0;
            frames.push(TrajectoryFrame {
                frame_index: f_idx,
                time_ps,
                coords: pos.clone(),
            });
            trace.push(MdTracePoint {
                frame: f_idx,
                time_ps,
                potential_kcal: (e_pot * 100.0).round() / 100.0,
                temperature_k: (temp_inst * 10.0).round() / 10.0,
                rmsd_a: (rmsd * 1000.0).round() / 1000.0,
            });
        }
    }

    let final_rmsd = trace.last().map_or(0.0, |t| t.rmsd_a);
    let mean_pot = if trace.is_empty() {
        0.0
    } else {
        (pot_sum / (trace.len() as f64) * 100.0).round() / 100.0
    };
    let n_rec = frames.len();
    mol.frames = frames;
    mol.set_active_frame(n_rec.saturating_sub(1));

    Ok(LocalMdReport {
        mobile_atoms: mobile.len(),
        total_steps: steps,
        frames_recorded: n_rec,
        target_temp_k: temperature_k,
        final_rmsd_a: final_rmsd,
        mean_potential_kcal: mean_pot,
        trace,
    })
}
