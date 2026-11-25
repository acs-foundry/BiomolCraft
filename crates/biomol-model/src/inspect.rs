#![forbid(unsafe_code)]

//! PDB/mmCIF Structural Inspection & Teaching Diagnostics (`info chains|residues|polymers|missing|atoms`).
//!
//! Answers the core structure-reading questions of structural biology practicals:
//! - Chain & copy inventory (polymers, ligands, ions, crystallographic waters).
//! - Polymer stretches and internal missing-residue gaps (sequence jumps or `Cα–Cα > 4.5 Å`).
//! - Missing / truncated sidechain heavy atoms compared against canonical amino acid templates.
//! - Sidechain `χ1` / `χ2` dihedral measurements and crystallographic B-factor statistics.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

use craftbio_struct::{Biomolecule, ResidueClass, dist3, measure_dihedral_deg};
use serde::{Deserialize, Serialize};

/// Summary of one chain in a biomolecular structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainInfoSummary {
    pub chain_id: char,
    pub dominant_class: String,
    pub protein_residues: usize,
    pub nucleic_residues: usize,
    pub ligand_names: Vec<(String, usize)>,
    pub ion_count: usize,
    pub water_count: usize,
    pub res_range: Option<(i32, i32)>,
    pub mean_bfactor: f64,
}

/// A detected missing-residue gap inside a polymer chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingGap {
    pub chain: char,
    pub prev_res: i32,
    pub next_res: i32,
    pub missing_count: i32,
    pub ca_distance_a: Option<f64>,
}

/// A residue whose sidechain is missing one or more expected heavy atoms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncatedSidechain {
    pub chain: char,
    pub res_seq: i32,
    pub res_name: String,
    pub missing_atoms: Vec<String>,
    pub mean_bfactor: f64,
}

/// Complete structural inspection report (`info`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureInfoReport {
    pub title: String,
    pub total_atoms: usize,
    pub chains: Vec<ChainInfoSummary>,
    pub modeled_stretches: Vec<(char, Vec<(i32, i32)>)>,
    pub missing_gaps: Vec<MissingGap>,
    pub truncated_sidechains: Vec<TruncatedSidechain>,
    pub bfactor_min: f64,
    pub bfactor_median: f64,
    pub bfactor_max: f64,
}

/// Canonical expected heavy sidechain atoms (beyond `N, CA, C, O`) for standard amino acids.
#[must_use]
pub fn expected_sidechain_heavy_atoms(aa: char) -> &'static [&'static str] {
    match aa.to_ascii_uppercase() {
        'G' => &[],
        'A' => &["CB"],
        'S' => &["CB", "OG"],
        'C' => &["CB", "SG"],
        'T' => &["CB", "OG1", "CG2"],
        'V' => &["CB", "CG1", "CG2"],
        'I' => &["CB", "CG1", "CG2", "CD1"],
        'L' => &["CB", "CG", "CD1", "CD2"],
        'D' => &["CB", "CG", "OD1", "OD2"],
        'N' => &["CB", "CG", "OD1", "ND2"],
        'E' => &["CB", "CG", "CD", "OE1", "OE2"],
        'Q' => &["CB", "CG", "CD", "OE1", "NE2"],
        'M' => &["CB", "CG", "SD", "CE"],
        'K' => &["CB", "CG", "CD", "CE", "NZ"],
        'R' => &["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        'H' => &["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        'F' => &["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        'Y' => &["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        'W' => &[
            "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        ],
        'P' => &["CB", "CG", "CD"],
        _ => &[],
    }
}

fn find_atom_coords(mol: &Biomolecule, atom_indices: &[usize], name: &str) -> Option<[f64; 3]> {
    atom_indices
        .iter()
        .copied()
        .find(|&i| mol.atoms[i].name == name)
        .map(|i| mol.atoms[i].coords)
}

/// Measure IUPAC `χ1` and `χ2` dihedral angles (degrees) for a protein residue if heavy atoms exist.
#[must_use]
pub fn measure_residue_chi1_chi2(
    mol: &Biomolecule,
    chain: char,
    res_seq: i32,
) -> (Option<f64>, Option<f64>) {
    let Some(ch) = mol.chain(chain) else {
        return (None, None);
    };
    let Some(res) = ch
        .residues
        .iter()
        .find(|r| r.res_seq == res_seq && r.class == ResidueClass::Protein)
    else {
        return (None, None);
    };
    let n = find_atom_coords(mol, &res.atom_indices, "N");
    let ca = find_atom_coords(mol, &res.atom_indices, "CA");
    let cb = find_atom_coords(mol, &res.atom_indices, "CB");
    let (Some(n), Some(ca), Some(cb)) = (n, ca, cb) else {
        return (None, None);
    };
    let cg_name = match res.code {
        'S' => "OG",
        'C' => "SG",
        'T' => "OG1",
        'V' | 'I' => "CG1",
        _ => "CG",
    };
    let Some(cg) = find_atom_coords(mol, &res.atom_indices, cg_name) else {
        return (None, None);
    };
    let chi1 = (measure_dihedral_deg(n, ca, cb, cg) * 10.0).round() / 10.0;

    let cd_name = match res.code {
        'I' | 'L' | 'F' | 'Y' | 'W' => Some("CD1"),
        'D' | 'N' => Some("OD1"),
        'E' | 'Q' | 'K' | 'R' | 'P' => Some("CD"),
        'M' => Some("SD"),
        'H' => Some("ND1"),
        _ => None,
    };
    let chi2 = cd_name
        .and_then(|dn| find_atom_coords(mol, &res.atom_indices, dn))
        .map(|cd| (measure_dihedral_deg(ca, cb, cg, cd) * 10.0).round() / 10.0);

    (Some(chi1), chi2)
}

/// Inspect chains, ligands, waters, polymer stretches, missing gaps, truncated sidechains & B-factors.
#[must_use]
pub fn inspect_structure_info(mol: &Biomolecule) -> StructureInfoReport {
    let mut chains = Vec::new();
    let mut modeled_stretches = Vec::new();
    let mut missing_gaps = Vec::new();
    let mut truncated_sidechains = Vec::new();

    for ch in &mol.chains {
        let mut prot_n = 0usize;
        let mut nuc_n = 0usize;
        let mut ion_n = 0usize;
        let mut wat_n = 0usize;
        let mut lig_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut min_r = i32::MAX;
        let mut max_r = i32::MIN;
        let mut bf_sum = 0.0_f64;
        let mut bf_cnt = 0usize;

        let mut poly_res = Vec::new();
        for r in &ch.residues {
            match r.class {
                ResidueClass::Protein => {
                    prot_n += 1;
                    min_r = min_r.min(r.res_seq);
                    max_r = max_r.max(r.res_seq);
                    bf_sum += r.mean_bfactor;
                    bf_cnt += 1;
                    poly_res.push(r);

                    let expected = expected_sidechain_heavy_atoms(r.code);
                    if !expected.is_empty() {
                        let present: HashSet<&str> = r
                            .atom_indices
                            .iter()
                            .map(|&ai| mol.atoms[ai].name.as_str())
                            .collect();
                        let missing: Vec<String> = expected
                            .iter()
                            .filter(|&&a| !present.contains(a))
                            .map(|&a| a.to_string())
                            .collect();
                        if !missing.is_empty() {
                            truncated_sidechains.push(TruncatedSidechain {
                                chain: ch.id,
                                res_seq: r.res_seq,
                                res_name: r.res_name.clone(),
                                missing_atoms: missing,
                                mean_bfactor: (r.mean_bfactor * 10.0).round() / 10.0,
                            });
                        }
                    }
                }
                ResidueClass::Rna | ResidueClass::Dna => {
                    nuc_n += 1;
                    min_r = min_r.min(r.res_seq);
                    max_r = max_r.max(r.res_seq);
                    bf_sum += r.mean_bfactor;
                    bf_cnt += 1;
                    poly_res.push(r);
                }
                ResidueClass::Ligand => {
                    *lig_counts.entry(r.res_name.clone()).or_insert(0) += 1;
                }
                ResidueClass::Ion => ion_n += 1,
                ResidueClass::Water => wat_n += 1,
            }
        }

        // Segment contiguous modeled stretches and missing gaps
        if !poly_res.is_empty() {
            let mut stretches = Vec::new();
            let mut start = poly_res[0].res_seq;
            let mut prev = poly_res[0];
            for cur in poly_res.iter().skip(1) {
                let seq_jump = cur.res_seq - prev.res_seq;
                let ca_dist = find_atom_coords(mol, &prev.atom_indices, "CA")
                    .zip(find_atom_coords(mol, &cur.atom_indices, "CA"))
                    .map(|(a, b)| (dist3(a, b) * 100.0).round() / 100.0);
                let broken = seq_jump > 1 || ca_dist.is_some_and(|d| d > 4.5);
                if broken {
                    stretches.push((start, prev.res_seq));
                    if seq_jump > 1 {
                        missing_gaps.push(MissingGap {
                            chain: ch.id,
                            prev_res: prev.res_seq,
                            next_res: cur.res_seq,
                            missing_count: seq_jump - 1,
                            ca_distance_a: ca_dist,
                        });
                    }
                    start = cur.res_seq;
                }
                prev = cur;
            }
            stretches.push((start, prev.res_seq));
            modeled_stretches.push((ch.id, stretches));
        }

        let dom = match ch.dominant_class {
            ResidueClass::Protein => "Protein",
            ResidueClass::Rna => "RNA",
            ResidueClass::Dna => "DNA",
            ResidueClass::Ligand => "Ligand",
            ResidueClass::Ion => "Ion",
            ResidueClass::Water => "Water",
        };
        chains.push(ChainInfoSummary {
            chain_id: ch.id,
            dominant_class: dom.to_string(),
            protein_residues: prot_n,
            nucleic_residues: nuc_n,
            ligand_names: lig_counts.into_iter().collect(),
            ion_count: ion_n,
            water_count: wat_n,
            res_range: if min_r <= max_r {
                Some((min_r, max_r))
            } else {
                None
            },
            mean_bfactor: if bf_cnt > 0 {
                ((bf_sum / bf_cnt as f64) * 10.0).round() / 10.0
            } else {
                0.0
            },
        });
    }

    let mut bs: Vec<f64> = mol
        .atoms
        .iter()
        .filter(|a| a.res_class != ResidueClass::Water && a.element != "H")
        .map(|a| a.b_factor)
        .filter(|v| v.is_finite())
        .collect();
    bs.sort_by(f64::total_cmp);
    let r1 = |v: f64| (v * 10.0).round() / 10.0;
    let bfactor_min = bs.first().copied().map_or(0.0, r1);
    let bfactor_median = if bs.is_empty() {
        0.0
    } else {
        r1(bs[bs.len() / 2])
    };
    let bfactor_max = bs.last().copied().map_or(0.0, r1);

    StructureInfoReport {
        title: mol.title.clone(),
        total_atoms: mol.atoms.len(),
        chains,
        modeled_stretches,
        missing_gaps,
        truncated_sidechains,
        bfactor_min,
        bfactor_median,
        bfactor_max,
    }
}

/// Render Markdown output for `info <summary|chains|polymers|missing|residues|atoms>`.
#[must_use]
pub fn render_info_markdown(
    mol: &Biomolecule,
    sub: &str,
    mask: Option<&[bool]>,
) -> String {
    let rep = inspect_structure_info(mol);
    let mut md = String::new();
    match sub.to_ascii_lowercase().as_str() {
        "chains" => {
            let _ = writeln!(md, "## Structure Chains (`info chains`) — {}\n", rep.title);
            md.push_str("| Chain | Type | Polymer Res | Range | Ligands | Ions | Waters | Mean B |\n|---|---|---:|---|---|---:|---:|---:|\n");
            for c in &rep.chains {
                let rng = c
                    .res_range
                    .map_or_else(|| "—".to_string(), |(a, b)| format!("{a}–{b}"));
                let ligs = if c.ligand_names.is_empty() {
                    "—".to_string()
                } else {
                    c.ligand_names
                        .iter()
                        .map(|(n, k)| format!("{n}×{k}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let _ = writeln!(
                    md,
                    "| `{}` | {} | {} | {} | {} | {} | {} | {:.1} |",
                    c.chain_id,
                    c.dominant_class,
                    c.protein_residues + c.nucleic_residues,
                    rng,
                    ligs,
                    c.ion_count,
                    c.water_count,
                    c.mean_bfactor
                );
            }
        }
        "polymers" => {
            let _ = writeln!(
                md,
                "## Polymer Stretches & Gaps (`info polymers`) — {}\n",
                rep.title
            );
            for (cid, stretches) in &rep.modeled_stretches {
                let s_txt: Vec<String> = stretches
                    .iter()
                    .map(|(a, b)| format!("{a}–{b}"))
                    .collect();
                let _ = writeln!(md, "- **Chain `{cid}` modeled stretches:** {}", s_txt.join(", "));
            }
            if rep.missing_gaps.is_empty() {
                md.push_str("\nNo internal residue-numbering gaps detected.\n");
            } else {
                md.push_str("\n### Internal Missing Residue Gaps\n\n| Chain | Gap Range | Missing Count | Cα–Cα (Å) |\n|---|---|---:|---:|\n");
                for g in &rep.missing_gaps {
                    let range = if g.missing_count == 1 {
                        format!("{}", g.prev_res + 1)
                    } else {
                        format!("{}–{}", g.prev_res + 1, g.next_res - 1)
                    };
                    let cd = g
                        .ca_distance_a
                        .map_or_else(|| "—".to_string(), |d| format!("{d:.2}"));
                    let _ = writeln!(
                        md,
                        "| `{}` | `{}` | {} | {} |",
                        g.chain, range, g.missing_count, cd
                    );
                }
            }
        }
        "missing" => {
            let _ = writeln!(
                md,
                "## Missing Residues, Truncated Sidechains & B-Factors (`info missing`) — {}\n",
                rep.title
            );
            let _ = writeln!(
                md,
                "- **Internal Missing Gaps:** `{}` | **Truncated Sidechains:** `{}` | **B-Factor (min / med / max):** `{:.1} / {:.1} / {:.1} Å²`\n",
                rep.missing_gaps.len(),
                rep.truncated_sidechains.len(),
                rep.bfactor_min,
                rep.bfactor_median,
                rep.bfactor_max
            );
            if !rep.missing_gaps.is_empty() {
                md.push_str("### Missing Residue Ranges\n");
                for g in &rep.missing_gaps {
                    let range = if g.missing_count == 1 {
                        format!("single residue {}", g.prev_res + 1)
                    } else {
                        format!("{}–{} ({} res)", g.prev_res + 1, g.next_res - 1, g.missing_count)
                    };
                    let _ = writeln!(md, "- Chain `{}`: {range}", g.chain);
                }
            }
            if !rep.truncated_sidechains.is_empty() {
                md.push_str("\n### Residues with Missing Sidechain Heavy Atoms\n\n| Residue | Missing Heavy Atoms | Mean B |\n|---|---|---:|\n");
                for t in rep.truncated_sidechains.iter().take(40) {
                    let _ = writeln!(
                        md,
                        "| `{}:{}{}` | `{}` | {:.1} |",
                        t.chain,
                        t.res_name,
                        t.res_seq,
                        t.missing_atoms.join(", "),
                        t.mean_bfactor
                    );
                }
            }
        }
        "residues" => {
            let _ = writeln!(md, "## Residue Table (`info residues`) — {}\n", rep.title);
            let mut rows = Vec::new();
            for ch in &mol.chains {
                for r in &ch.residues {
                    let hit = match mask {
                        Some(m) => r
                            .atom_indices
                            .iter()
                            .any(|&ai| m.get(ai).copied().unwrap_or(false)),
                        None => r.class != ResidueClass::Water,
                    };
                    if hit {
                        rows.push(r);
                    }
                }
            }
            let _ = writeln!(md, "Matched **{}** residue(s):\n", rows.len());
            md.push_str("| Chain | Residue | Class | Atoms | Mean B | χ1 (°) | χ2 (°) |\n|---|---|---|---:|---:|---:|---:|\n");
            for r in rows.iter().take(80) {
                let (c1, c2) = if r.class == ResidueClass::Protein {
                    measure_residue_chi1_chi2(mol, r.chain, r.res_seq)
                } else {
                    (None, None)
                };
                let s1 = c1.map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
                let s2 = c2.map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
                let _ = writeln!(
                    md,
                    "| `{}` | `{}{}` | {:?} | {} | {:.1} | {} | {} |",
                    r.chain,
                    r.res_name,
                    r.res_seq,
                    r.class,
                    r.atom_indices.len(),
                    r.mean_bfactor,
                    s1,
                    s2
                );
            }
        }
        "atoms" => {
            let _ = writeln!(md, "## Atom Table (`info atoms`) — {}\n", rep.title);
            let mut deg = vec![0usize; mol.atoms.len()];
            for b in &mol.bonds {
                if b.a < deg.len() && b.b < deg.len() {
                    deg[b.a] += 1;
                    deg[b.b] += 1;
                }
            }
            let idxs: Vec<usize> = (0..mol.atoms.len())
                .filter(|&i| match mask {
                    Some(m) => m.get(i).copied().unwrap_or(false),
                    None => mol.atoms[i].res_class != ResidueClass::Water,
                })
                .collect();
            let _ = writeln!(md, "Matched **{}** atom(s):\n", idxs.len());
            md.push_str("| Spec | Elem | Bond Deg | Geom Hint | Occ | B-factor | Coords (Å) |\n|---|---|---:|---|---:|---:|---|\n");
            for &i in idxs.iter().take(80) {
                let a = &mol.atoms[i];
                let d = deg[i];
                let hint = match (a.element.as_str(), d) {
                    ("C", 2) => "C1/sp (linear/triple?)",
                    ("C", 3) => "C2/sp2",
                    ("C", 4) => "C3/sp3",
                    ("N", 1..=2) => "N/sp-sp2",
                    ("N", 3..=4) => "N/sp2-sp3",
                    ("O", 1) => "O=carbonyl/hydroxyl",
                    _ => "heavy",
                };
                let _ = writeln!(
                    md,
                    "| `{}:{}{}.{}` | {} | {} | {} | {:.2} | {:.1} | `({:.2}, {:.2}, {:.2})` |",
                    a.chain,
                    a.res_name,
                    a.res_seq,
                    a.name,
                    a.element,
                    d,
                    hint,
                    a.occupancy,
                    a.b_factor,
                    a.coords[0],
                    a.coords[1],
                    a.coords[2]
                );
            }
        }
        _ => {
            let _ = writeln!(md, "## Structure Summary (`info`) — {}\n", rep.title);
            let _ = writeln!(
                md,
                "- **Total Atoms:** `{}` | **Chains:** `{}` | **Internal Gaps:** `{}` | **Truncated Sidechains:** `{}`",
                rep.total_atoms,
                rep.chains.len(),
                rep.missing_gaps.len(),
                rep.truncated_sidechains.len()
            );
            let _ = writeln!(
                md,
                "- **B-Factor (min / median / max):** `{:.1} / {:.1} / {:.1} Å²` | **AltLocs:** `{}`\n",
                rep.bfactor_min,
                rep.bfactor_median,
                rep.bfactor_max,
                mol.altlocs.len()
            );
            md.push_str("| Chain | Type | Polymer Res | Range | Ligands | Waters |\n|---|---|---:|---|---|---:|\n");
            for c in &rep.chains {
                let rng = c
                    .res_range
                    .map_or_else(|| "—".to_string(), |(a, b)| format!("{a}–{b}"));
                let ligs = if c.ligand_names.is_empty() {
                    "—".to_string()
                } else {
                    c.ligand_names
                        .iter()
                        .map(|(n, k)| format!("{n}×{k}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let _ = writeln!(
                    md,
                    "| `{}` | {} | {} | {} | {} | {} |",
                    c.chain_id,
                    c.dominant_class,
                    c.protein_residues + c.nucleic_residues,
                    rng,
                    ligs,
                    c.water_count
                );
            }
        }
    }
    md
}
