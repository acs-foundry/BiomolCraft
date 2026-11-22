#![forbid(unsafe_code)]

//! `biomol-enzyme` — Enzymes & Protein Engineering Module for `BiomolCraft`:
//! - Automatic 3D catalytic triad (`Ser/Cys/Thr–His–Asp/Glu`) & aspartic dyad (`Asp–Asp`) detection.
//! - Metal & cofactor (`Zn`, `Mg`, `Ca`, `Fe`, `Mn`, `Cu`, `HEME`, `NAD/FAD/ATP`) coordination sphere geometry.
//! - 3D substrate access tunnel / channel finder emitting translucent 3D `SphereOverlay` beads.
//! - Geometric thermostabilizing disulfide bridge engineering (`Cβ–Cβ` 3.4–4.6 Å, `Cα–Cα` 4.2–6.8 Å).
//! - Fast fold stability `ΔΔG_fold` mutation heuristic.

use std::fmt::Write as _;

use craftbio_scene::{LineOverlay, SphereOverlay};
use craftbio_struct::{Biomolecule, ResidueClass, compute_sasa, dist3};
use serde::{Deserialize, Serialize};

/// Detected 3D catalytic triad or dyad motif.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalyticMotif {
    pub motif_kind: String,
    pub residues: Vec<(char, i32, String)>,
    pub key_distances_a: Vec<f64>,
    pub center: [f64; 3],
}

/// Detected metal ion or organic cofactor coordination sphere.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalCofactorSite {
    pub label: String,
    pub center: [f64; 3],
    pub coordination_number: usize,
    pub geometry_class: String,
    pub coordinating_residues: Vec<(char, i32, String, f64)>,
}

/// 3D substrate access tunnel leading from an internal active-site point to bulk solvent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateTunnel {
    pub id: usize,
    pub length_a: f64,
    pub bottleneck_radius_a: f64,
    pub beads: Vec<([f64; 3], f64)>,
    pub lining_residues: Vec<String>,
}

/// Candidate engineered disulfide bridge for thermostabilization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisulfideProposal {
    pub chain_a: char,
    pub res_seq_a: i32,
    pub res_name_a: String,
    pub chain_b: char,
    pub res_seq_b: i32,
    pub res_name_b: String,
    pub ca_dist_a: f64,
    pub cb_dist_a: f64,
    pub score_0_100: f64,
    pub est_delta_tm_c: f64,
    pub point_a: [f64; 3],
    pub point_b: [f64; 3],
}

/// Complete Enzyme & Protein Engineering audit report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeEngineeringReport {
    pub title: String,
    pub catalytic_motifs: Vec<CatalyticMotif>,
    pub metal_cofactor_sites: Vec<MetalCofactorSite>,
    pub tunnels: Vec<SubstrateTunnel>,
    pub disulfide_proposals: Vec<DisulfideProposal>,
    pub native_disulfide_count: usize,
}

fn find_sidechain_tip(
    mol: &Biomolecule,
    atom_indices: &[usize],
    names: &[&str],
) -> Option<[f64; 3]> {
    for &n in names {
        if let Some(&idx) = atom_indices.iter().find(|&&i| mol.atoms[i].name == n) {
            return Some(mol.atoms[idx].coords);
        }
    }
    None
}

/// Run the full `biomol-enzyme` 3D structural engineering pipeline on `mol`.
#[must_use]
pub fn analyze_enzyme_engineering(mol: &Biomolecule) -> EnzymeEngineeringReport {
    // 1. Detect catalytic triads (Ser/Cys/Thr - His - Asp/Glu) & Asp-Asp dyads
    let mut nucleophiles = Vec::new();
    let mut histidines = Vec::new();
    let mut acids = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            if r.class != ResidueClass::Protein {
                continue;
            }
            match r.code {
                'S' | 'C' | 'T' => {
                    if let Some(p) = find_sidechain_tip(mol, &r.atom_indices, &["OG", "SG", "OG1"])
                    {
                        nucleophiles.push((r.chain, r.res_seq, r.res_name.clone(), p));
                    }
                }
                'H' => {
                    if let Some(p) = find_sidechain_tip(mol, &r.atom_indices, &["NE2", "ND1"]) {
                        histidines.push((r.chain, r.res_seq, r.res_name.clone(), p));
                    }
                }
                'D' | 'E' => {
                    if let Some(p) =
                        find_sidechain_tip(mol, &r.atom_indices, &["OD1", "OD2", "OE1", "OE2"])
                    {
                        acids.push((r.chain, r.res_seq, r.res_name.clone(), p));
                    }
                }
                _ => {}
            }
        }
    }

    type ResTip = (char, i32, String, [f64; 3]);
    let mut catalytic_motifs = Vec::new();
    for (hc, hr, hn, hp) in &histidines {
        let mut best_nuc: Option<(&ResTip, f64)> = None;
        for nuc in &nucleophiles {
            let d = dist3(*hp, nuc.3);
            if d <= 4.3 && best_nuc.is_none_or(|(_, bd)| d < bd) {
                best_nuc = Some((nuc, d));
            }
        }
        let mut best_ac: Option<(&ResTip, f64)> = None;
        for ac in &acids {
            let d = dist3(*hp, ac.3);
            if d <= 4.3 && best_ac.is_none_or(|(_, bd)| d < bd) {
                best_ac = Some((ac, d));
            }
        }
        if let (Some((nuc, d1)), Some((ac, d2))) = (best_nuc, best_ac) {
            let center = [
                (hp[0] + nuc.3[0] + ac.3[0]) / 3.0,
                (hp[1] + nuc.3[1] + ac.3[1]) / 3.0,
                (hp[2] + nuc.3[2] + ac.3[2]) / 3.0,
            ];
            catalytic_motifs.push(CatalyticMotif {
                motif_kind: format!("{}-{}-{} Catalytic Triad", nuc.2, hn, ac.2),
                residues: vec![
                    (nuc.0, nuc.1, nuc.2.clone()),
                    (*hc, *hr, hn.clone()),
                    (ac.0, ac.1, ac.2.clone()),
                ],
                key_distances_a: vec![(d1 * 100.0).round() / 100.0, (d2 * 100.0).round() / 100.0],
                center,
            });
            if catalytic_motifs.len() >= 6 {
                break;
            }
        }
    }
    if catalytic_motifs.is_empty() {
        // Check Asp-Asp catalytic dyad (e.g., aspartic proteases / HIV-PR / BACE1)
        for i in 0..acids.len() {
            for j in (i + 1)..acids.len() {
                if acids[i].2 == "ASP" && acids[j].2 == "ASP" {
                    let d = dist3(acids[i].3, acids[j].3);
                    if (2.4..=4.4).contains(&d) {
                        catalytic_motifs.push(CatalyticMotif {
                            motif_kind: "Asp-Asp Catalytic Dyad".to_string(),
                            residues: vec![
                                (acids[i].0, acids[i].1, acids[i].2.clone()),
                                (acids[j].0, acids[j].1, acids[j].2.clone()),
                            ],
                            key_distances_a: vec![(d * 100.0).round() / 100.0],
                            center: [
                                0.5 * (acids[i].3[0] + acids[j].3[0]),
                                0.5 * (acids[i].3[1] + acids[j].3[1]),
                                0.5 * (acids[i].3[2] + acids[j].3[2]),
                            ],
                        });
                    }
                }
            }
            if catalytic_motifs.len() >= 4 {
                break;
            }
        }
    }

    // 2. Metal & Cofactor Coordination Sphere Analysis
    let mut metal_cofactor_sites = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            if !matches!(r.class, ResidueClass::Ion | ResidueClass::Ligand) {
                continue;
            }
            let mut center = [0.0; 3];
            for &ai in &r.atom_indices {
                center[0] += mol.atoms[ai].coords[0];
                center[1] += mol.atoms[ai].coords[1];
                center[2] += mol.atoms[ai].coords[2];
            }
            let cnt = r.atom_indices.len().max(1) as f64;
            center[0] /= cnt;
            center[1] /= cnt;
            center[2] /= cnt;

            let cutoff = if r.class == ResidueClass::Ion {
                3.1
            } else {
                4.0
            };
            let mut donors = Vec::new();
            for pch in &mol.chains {
                for pr in &pch.residues {
                    if !matches!(
                        pr.class,
                        ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
                    ) {
                        continue;
                    }
                    let mut min_d = f64::INFINITY;
                    for &ai in &pr.atom_indices {
                        let a = &mol.atoms[ai];
                        if matches!(a.element.as_str(), "N" | "O" | "S") {
                            let d = dist3(a.coords, center);
                            if d < min_d {
                                min_d = d;
                            }
                        }
                    }
                    if min_d <= cutoff {
                        donors.push((
                            pr.chain,
                            pr.res_seq,
                            pr.res_name.clone(),
                            (min_d * 100.0).round() / 100.0,
                        ));
                    }
                }
            }
            donors.sort_by(|a, b| a.3.total_cmp(&b.3));
            let cn = donors.len();
            let geom = match cn {
                0..=2 => "Weak / Surface Contact",
                3 => "Trigonal Planar / Pyramidal",
                4 => "Tetrahedral (Zn/Fe-like)",
                5 => "Trigonal Bipyramidal",
                6 => "Octahedral (Mg/Ca-like)",
                _ => "Extended Cofactor Pocket",
            }
            .to_string();
            metal_cofactor_sites.push(MetalCofactorSite {
                label: format!("{}:{}{}", r.chain, r.res_name, r.res_seq),
                center,
                coordination_number: cn,
                geometry_class: geom,
                coordinating_residues: donors.into_iter().take(8).collect(),
            });
        }
    }

    // 3. 3D Substrate Access Tunnel Finder from Active Site Center -> Bulk Solvent
    let active_origin = catalytic_motifs
        .first()
        .map(|m| m.center)
        .or_else(|| metal_cofactor_sites.first().map(|s| s.center))
        .unwrap_or_else(|| mol.centroid());
    let tunnels = detect_enzyme_tunnels(mol, active_origin);

    // 4. Stabilizing Disulfide Bridge Engineering (Cβ-Cβ 3.4..4.6 A, Cα-Cα 4.2..6.8 A)
    let disulfide_proposals = design_stabilizing_disulfides(mol);

    EnzymeEngineeringReport {
        title: mol.title.clone(),
        catalytic_motifs,
        metal_cofactor_sites,
        tunnels,
        disulfide_proposals,
        native_disulfide_count: mol.disulfides.len(),
    }
}

/// Ray-marching clearance-maximizing 3D substrate access tunnel finder.
#[must_use]
pub fn detect_enzyme_tunnels(mol: &Biomolecule, origin: [f64; 3]) -> Vec<SubstrateTunnel> {
    let heavy: Vec<([f64; 3], char, i32, &str)> = mol
        .atoms
        .iter()
        .filter(|a| {
            matches!(
                a.res_class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            ) && a.element != "H"
        })
        .map(|a| (a.coords, a.chain, a.res_seq, a.res_name.as_str()))
        .collect();
    if heavy.len() < 20 {
        return Vec::new();
    }

    let clearance_at = |p: [f64; 3]| -> f64 {
        let mut min_d = f64::INFINITY;
        for &(c, _, _, _) in &heavy {
            let d = dist3(p, c);
            if d < min_d {
                min_d = d;
            }
        }
        (min_d - 1.55).max(0.0)
    };

    // 14 golden-spiral probe directions outward from active site
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let n_dirs = 14usize;
    let mut candidates: Vec<SubstrateTunnel> = Vec::new();

    for k in 0..n_dirs {
        let y = 1.0 - (k as f64 / ((n_dirs - 1) as f64)) * 2.0;
        let rad = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * (k as f64);
        let mut dir = [rad * theta.cos(), y, rad * theta.sin()];
        let mut pos = origin;
        let mut beads = Vec::new();
        let mut bottleneck = f64::INFINITY;
        let mut length = 0.0_f64;
        let step_a = 1.5_f64;

        for step in 0..16 {
            let clr = clearance_at(pos);
            if step > 0 {
                bottleneck = bottleneck.min(clr);
            }
            beads.push((pos, clr.clamp(0.85, 3.2)));
            if clr > 4.6 && step >= 3 {
                // Reached bulk solvent!
                break;
            }
            // Greedily nudge forward direction toward maximum local clearance
            let mut best_next = [
                pos[0] + dir[0] * step_a,
                pos[1] + dir[1] * step_a,
                pos[2] + dir[2] * step_a,
            ];
            let mut best_score = clearance_at(best_next);
            for &(ox, oy, oz) in &[
                (0.35, 0.0, 0.0),
                (-0.35, 0.0, 0.0),
                (0.0, 0.35, 0.0),
                (0.0, -0.35, 0.0),
                (0.0, 0.0, 0.35),
                (0.0, 0.0, -0.35),
            ] {
                let mut cand_d = [dir[0] + ox, dir[1] + oy, dir[2] + oz];
                let nrm = (cand_d[0] * cand_d[0] + cand_d[1] * cand_d[1] + cand_d[2] * cand_d[2])
                    .sqrt()
                    .max(1e-6);
                cand_d[0] /= nrm;
                cand_d[1] /= nrm;
                cand_d[2] /= nrm;
                let np = [
                    pos[0] + cand_d[0] * step_a,
                    pos[1] + cand_d[1] * step_a,
                    pos[2] + cand_d[2] * step_a,
                ];
                // Must keep moving outward from origin
                if dist3(np, origin) > dist3(pos, origin) + 0.5 {
                    let sc = clearance_at(np);
                    if sc > best_score {
                        best_score = sc;
                        best_next = np;
                        dir = cand_d;
                    }
                }
            }
            length += dist3(pos, best_next);
            pos = best_next;
        }

        if beads.len() >= 4 && bottleneck.is_finite() && bottleneck >= 0.75 {
            // Collect lining residues within 4.5 A of tunnel beads
            let mut lining = Vec::new();
            for &(bp, _) in &beads {
                for &(ac, ch, rseq, rname) in &heavy {
                    if dist3(bp, ac) <= 4.5 {
                        let tag = format!("{ch}:{rname}{rseq}");
                        if !lining.contains(&tag) && lining.len() < 12 {
                            lining.push(tag);
                        }
                    }
                }
            }
            candidates.push(SubstrateTunnel {
                id: 0,
                length_a: (length * 10.0).round() / 10.0,
                bottleneck_radius_a: (bottleneck * 100.0).round() / 100.0,
                beads,
                lining_residues: lining,
            });
        }
    }

    // Deduplicate tunnels with exit points closer than 6.0 A, keeping widest bottleneck
    candidates.sort_by(|a, b| b.bottleneck_radius_a.total_cmp(&a.bottleneck_radius_a));
    let mut kept: Vec<SubstrateTunnel> = Vec::new();
    for mut t in candidates {
        let exit = t.beads.last().map_or([0.0; 3], |b| b.0);
        let duplicate = kept.iter().any(|k| {
            let ke = k.beads.last().map_or([0.0; 3], |b| b.0);
            dist3(exit, ke) < 6.0
        });
        if !duplicate {
            t.id = kept.len() + 1;
            kept.push(t);
            if kept.len() >= 3 {
                break;
            }
        }
    }
    kept
}

/// Scan a protein structure for geometrically ideal thermostabilizing disulfide sites.
#[must_use]
pub fn design_stabilizing_disulfides(mol: &Biomolecule) -> Vec<DisulfideProposal> {
    struct ResGeom {
        chain: char,
        res_seq: i32,
        res_name: String,
        ca: [f64; 3],
        cb: [f64; 3],
    }
    let mut geom = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            if r.class != ResidueClass::Protein || r.code == 'C' || r.code == 'P' || r.code == 'G' {
                continue;
            }
            let Some(ca_idx) = r.guide_idx else { continue };
            let Some(&cb_idx) = r.atom_indices.iter().find(|&&i| mol.atoms[i].name == "CB") else {
                continue;
            };
            geom.push(ResGeom {
                chain: r.chain,
                res_seq: r.res_seq,
                res_name: r.res_name.clone(),
                ca: mol.atoms[ca_idx].coords,
                cb: mol.atoms[cb_idx].coords,
            });
        }
    }

    let mut proposals = Vec::new();
    for i in 0..geom.len() {
        for j in (i + 1)..geom.len() {
            let a = &geom[i];
            let b = &geom[j];
            if a.chain == b.chain && (a.res_seq - b.res_seq).abs() < 6 {
                continue;
            }
            let d_ca = dist3(a.ca, b.ca);
            if !(4.2..=6.8).contains(&d_ca) {
                continue;
            }
            let d_cb = dist3(a.cb, b.cb);
            if !(3.4..=4.6).contains(&d_cb) {
                continue;
            }
            // Ideal disulfide geometry centers around Cβ-Cβ = 3.95 Å and Cα-Cα = 5.4 Å
            let err_cb = (d_cb - 3.95).abs() / 0.65;
            let err_ca = (d_ca - 5.40).abs() / 1.40;
            let score = (100.0 * (1.0 - 0.55 * err_cb - 0.45 * err_ca)).clamp(20.0, 99.0);
            let seq_sep = if a.chain == b.chain {
                (a.res_seq - b.res_seq).abs() as f64
            } else {
                80.0
            };
            // Entropic loop closure stabilization scales with ln(seq_sep)
            let est_dtm = ((1.2 + 0.85 * seq_sep.ln()) * (score / 100.0)).clamp(1.0, 9.5);
            proposals.push(DisulfideProposal {
                chain_a: a.chain,
                res_seq_a: a.res_seq,
                res_name_a: a.res_name.clone(),
                chain_b: b.chain,
                res_seq_b: b.res_seq,
                res_name_b: b.res_name.clone(),
                ca_dist_a: (d_ca * 100.0).round() / 100.0,
                cb_dist_a: (d_cb * 100.0).round() / 100.0,
                score_0_100: (score * 10.0).round() / 10.0,
                est_delta_tm_c: (est_dtm * 10.0).round() / 10.0,
                point_a: a.cb,
                point_b: b.cb,
            });
        }
    }
    proposals.sort_by(|x, y| y.score_0_100.total_cmp(&x.score_0_100));
    proposals.truncate(8);
    proposals
}

/// Fast fold-stability `ΔΔG_fold` heuristic (kcal/mol; negative = stabilizing) for a point mutation.
#[must_use]
pub fn estimate_mutation_ddg_fold(
    mol: &Biomolecule,
    chain: char,
    res_seq: i32,
    new_aa: char,
) -> f64 {
    let sasa = compute_sasa(&mol.atoms, 1.4, 24);
    let Some(ch) = mol.chain(chain) else {
        return 0.0;
    };
    let Some(res) = ch.residues.iter().find(|r| r.res_seq == res_seq) else {
        return 0.0;
    };
    let rsasa: f64 = res.atom_indices.iter().map(|&i| sasa[i]).sum();
    let burial = (1.0 - (rsasa / 160.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let hydro = |aa: char| match aa {
        'I' | 'V' | 'L' | 'F' | 'M' | 'W' => 1.8,
        'A' | 'C' | 'Y' => 0.9,
        'G' | 'T' | 'S' | 'P' => 0.0,
        'H' | 'N' | 'Q' => -0.8,
        'D' | 'E' | 'K' | 'R' => -1.6,
        _ => 0.0,
    };
    let d_hydro = hydro(new_aa.to_ascii_uppercase()) - hydro(res.code);
    // Burying hydrophobic sidechains stabilizes; burying polar charges destabilizes
    let ddg = -burial * d_hydro * 1.15;
    (ddg * 100.0).round() / 100.0
}

/// Build 3D visual overlays (`SphereOverlay` tunnels & active sites + `LineOverlay` disulfide previews).
#[must_use]
pub fn enzyme_visual_overlays(
    report: &EnzymeEngineeringReport,
) -> (Vec<SphereOverlay>, Vec<LineOverlay>) {
    let mut spheres = Vec::new();
    let mut lines = Vec::new();

    for m in &report.catalytic_motifs {
        spheres.push(SphereOverlay {
            label: m.motif_kind.clone(),
            category: "Catalytic".to_string(),
            coords: m.center,
            radius_a: 2.2,
            rgba: [0.96, 0.28, 0.36, 0.72],
        });
    }
    for s in &report.metal_cofactor_sites {
        spheres.push(SphereOverlay {
            label: s.label.clone(),
            category: "Cofactor/Metal".to_string(),
            coords: s.center,
            radius_a: 1.9,
            rgba: [0.68, 0.38, 0.98, 0.75],
        });
    }
    let tunnel_colors = [
        [0.14, 0.86, 0.88, 0.42],
        [0.24, 0.90, 0.56, 0.38],
        [0.98, 0.78, 0.20, 0.35],
    ];
    for (ti, t) in report.tunnels.iter().enumerate() {
        let col = tunnel_colors[ti % tunnel_colors.len()];
        for &(pt, rad) in &t.beads {
            spheres.push(SphereOverlay {
                label: format!("Tunnel #{}", t.id),
                category: "Tunnel".to_string(),
                coords: pt,
                radius_a: rad.clamp(0.9, 2.5),
                rgba: col,
            });
        }
    }
    for d in report.disulfide_proposals.iter().take(4) {
        lines.push(LineOverlay {
            label: format!(
                "S-S {}{}-{}{}",
                d.chain_a, d.res_seq_a, d.chain_b, d.res_seq_b
            ),
            from: d.point_a,
            to: d.point_b,
            dashed: true,
            rgba: [0.98, 0.85, 0.16, 0.95],
        });
    }
    (spheres, lines)
}

/// Render a structured Markdown report for the Enzyme & Protein Engineering module.
#[must_use]
pub fn render_enzyme_markdown(rep: &EnzymeEngineeringReport) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Enzyme & Protein Engineering Audit: {}\n", rep.title);
    let _ = writeln!(
        md,
        "- **Catalytic Motifs Detected:** `{}` | **Metal/Cofactor Sites:** `{}` | **3D Substrate Tunnels:** `{}` | **Native Disulfides:** `{}`",
        rep.catalytic_motifs.len(),
        rep.metal_cofactor_sites.len(),
        rep.tunnels.len(),
        rep.native_disulfide_count
    );
    if !rep.catalytic_motifs.is_empty() {
        md.push_str("\n## Catalytic Triads / Dyads\n\n");
        for m in &rep.catalytic_motifs {
            let rlist: Vec<String> = m
                .residues
                .iter()
                .map(|(c, r, n)| format!("{c}:{n}{r}"))
                .collect();
            let _ = writeln!(
                md,
                "- **{}**: `{}` (distances `{:?} Å`)",
                m.motif_kind,
                rlist.join(" – "),
                m.key_distances_a
            );
        }
    }
    if !rep.metal_cofactor_sites.is_empty() {
        md.push_str("\n## Metal & Cofactor Coordination Spheres\n\n");
        for s in &rep.metal_cofactor_sites {
            let donors: Vec<String> = s
                .coordinating_residues
                .iter()
                .take(6)
                .map(|(c, r, n, d)| format!("{c}:{n}{r} ({d:.2} Å)"))
                .collect();
            let _ = writeln!(
                md,
                "- **{}** — CN=`{}` ({}): {}",
                s.label,
                s.coordination_number,
                s.geometry_class,
                donors.join(", ")
            );
        }
    }
    if !rep.tunnels.is_empty() {
        md.push_str("\n## 3D Substrate Access Tunnels\n\n| Tunnel | Length (Å) | Bottleneck Radius (Å) | Lining Residues |\n|---:|---:|---:|---|\n");
        for t in &rep.tunnels {
            let _ = writeln!(
                md,
                "| #{} | {:.1} | {:.2} | {} |",
                t.id,
                t.length_a,
                t.bottleneck_radius_a,
                t.lining_residues
                    .iter()
                    .take(6)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    if !rep.disulfide_proposals.is_empty() {
        md.push_str("\n## Stabilizing Disulfide Engineering Proposals\n\n| Pair | Cα–Cα (Å) | Cβ–Cβ (Å) | Score | Est. ΔTm |\n|---|---:|---:|---:|---:|\n");
        for d in &rep.disulfide_proposals {
            let _ = writeln!(
                md,
                "| `{}:{}{}C – {}:{}{}C` | {:.2} | {:.2} | {:.1}/100 | +{:.1} °C |",
                d.chain_a,
                d.res_name_a,
                d.res_seq_a,
                d.chain_b,
                d.res_name_b,
                d.res_seq_b,
                d.ca_dist_a,
                d.cb_dist_a,
                d.score_0_100,
                d.est_delta_tm_c
            );
        }
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use craftbio_struct::parse_pdb;

    #[test]
    fn detects_serine_protease_catalytic_triad_and_disulfide_pair()
    -> Result<(), Box<dyn std::error::Error>> {
        let pdb = "\
ATOM      1  CA  SER A  10       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CB  SER A  10       1.200   0.000   0.000  1.00 90.00           C
ATOM      3  OG  SER A  10       2.100   0.500   0.000  1.00 90.00           O
ATOM      4  CA  HIS A  25       5.200   0.400   0.000  1.00 90.00           C
ATOM      5  CB  HIS A  25       5.100   0.200   0.000  1.00 90.00           C
ATOM      6  NE2 HIS A  25       4.800   0.600   0.000  1.00 90.00           N
ATOM      7  CA  ASP A  40       8.800   0.500   0.000  1.00 90.00           C
ATOM      8  OD1 ASP A  40       7.600   0.600   0.000  1.00 90.00           O
HETATM    9 ZN    ZN B 201       2.100   2.700   0.000  1.00 99.00          ZN
END
";
        let mol = parse_pdb("enz_test", pdb)?;
        let rep = analyze_enzyme_engineering(&mol);
        assert_eq!(rep.catalytic_motifs.len(), 1);
        assert!(rep.catalytic_motifs[0].motif_kind.contains("SER-HIS-ASP"));
        assert_eq!(rep.metal_cofactor_sites.len(), 1);
        assert!(!rep.disulfide_proposals.is_empty());
        Ok(())
    }
}
