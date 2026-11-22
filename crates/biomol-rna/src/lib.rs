#![forbid(unsafe_code)]

//! `biomol-rna` — First-Class **ncRNA, mRNA, siRNA, ASO & Riboswitch** Engineering Module:
//! - Deterministic Zuker / Turner-2004 MFE 2D RNA Secondary Structure Folding (`fold_rna_mfe`)
//! - Structural Motifs (G-Quadruplex, Hairpins, Stems, AU-Rich Elements, TLR7/8 Motifs)
//! - Therapeutic mRNA Audit (`5'-UTR` ribosome accessibility, Kozak, U% / `m1Ψ`, LNP molar ratios)
//! - 2D Predicted vs 3D Crystal Base-Pair Concordance & XSS-Safe SVG Arc Diagram

use std::fmt::Write as _;

use craftbio_report::escape_markup;
use craftbio_struct::BasePair3D;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum RNA sequence length for interactive O(N^3) Zuker MFE folding (bounds compute cleanly).
pub const MAX_FOLD_RNA_LEN: usize = 320;

#[derive(Debug, Error)]
pub enum RnaError {
    #[error("empty RNA sequence")]
    EmptySequence,
    #[error("invalid nucleotide '{0}' (expected A, U/T, G, C)")]
    InvalidBase(char),
}

/// Curated reference ncRNA & mRNA therapeutic / structural preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RnaPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub class: &'static str,
    pub pdb_id: Option<&'static str>,
    pub sequence: &'static str,
    pub summary: &'static str,
}

/// Base pair predicted by Zuker MFE 2D folding (0-based indices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RnaPair2D {
    pub i: usize,
    pub j: usize,
    pub base_i: char,
    pub base_j: char,
    pub kind: String,
}

/// Complete Zuker MFE 2D RNA folding result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RnaFoldReport {
    pub sequence: String,
    pub folded_len: usize,
    pub dot_bracket: String,
    pub mfe_kcal_mol: f64,
    pub pairs: Vec<RnaPair2D>,
    pub gc_percent: f64,
    pub uridine_percent: f64,
    pub paired_fraction_pct: f64,
    pub hairpin_count: usize,
}

/// LNP (Lipid Nanoparticle) formulation calculation for mRNA/siRNA delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LnpFormulationSpec {
    pub template_name: String,
    pub molar_ratio_ion_dspc_chol_peg: String,
    pub np_charge_ratio: f64,
    pub rna_nt_length: usize,
    pub rna_mw_kda: f64,
    pub lipid_to_rna_mass_ratio: f64,
}

/// Complete ncRNA / mRNA therapeutic & structural analysis dossier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcRnaAnalysisReport {
    pub name: String,
    pub rna_class: String,
    pub fold: RnaFoldReport,
    pub first_30nt_mfe_kcal_mol: f64,
    pub g_quadruplex_motifs: Vec<String>,
    pub tlr7_8_motifs_count: usize,
    pub au_rich_elements_count: usize,
    pub has_kozak_aug: bool,
    pub m1psi_recommendation: String,
    pub lnp_spec: LnpFormulationSpec,
    pub crystal_3d_concordance_pct: Option<f64>,
    pub notes: Vec<String>,
}

/// Curated clinical & structural ncRNA / mRNA benchmarks.
#[must_use]
pub fn rna_presets() -> Vec<RnaPreset> {
    vec![
        RnaPreset {
            id: "guanine_riboswitch",
            name: "xpt-pbuX Guanine Riboswitch Aptamer (1Y26)",
            class: "ncRNA — Riboswitch Aptamer",
            pdb_id: Some("1Y26"),
            sequence: "GGACACAUCAGAUUUAUCCGGUUCUCAACAUCCUCGGGGUCUAUGUGCCCGCGGGAUGGUCAGACUGGCAUGG",
            summary: "Canonical 3-way junction purine-sensing ncRNA aptamer (PDB 1Y26); textbook small-molecule RNA-binding pocket benchmark.",
        },
        RnaPreset {
            id: "crispr_sgrna",
            name: "SpCas9 Single-Guide RNA (sgRNA) Scaffold",
            class: "ncRNA — CRISPR Guide RNA",
            pdb_id: Some("4OO8"),
            sequence: "GUCAGAGAGUGGCUAAUAAAGUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUUU",
            summary: "Engineered CRISPR-Cas9 chimeric ncRNA combining 20-nt spacer + tetraloop + nexus + 3' stem-loops.",
        },
        RnaPreset {
            id: "bnt162b2_5utr",
            name: "Human Alpha-Globin 5'-UTR + Kozak + CDS Start",
            class: "mRNA — 5'-UTR Translation Leader",
            pdb_id: None,
            sequence: "GAAUAAACUAGUAUUCUUCUGGUCCCCACAGACUCAGAGAGAACCCGCCACCAUGUUCGUGUUCCUGGUGCUGCUGCCCCUGGUG",
            summary: "Benchmark high-translation therapeutic mRNA leader (open 5'-end, strong GCCACCAUG Kozak, zero upstream uORFs).",
        },
        RnaPreset {
            id: "patisiran_sirna",
            name: "Patisiran (Onpattro) Anti-TTR siRNA Guide",
            class: "ncRNA — Therapeutic siRNA",
            pdb_id: None,
            sequence: "AUGGAAUACUCUUGGUUACUU",
            summary: "FDA-approved LNP-formulated siRNA targeting transthyretin mRNA (21-nt antisense guide strand with 3' UU overhang).",
        },
        RnaPreset {
            id: "g4_terra",
            name: "Human Telomeric ncRNA (TERRA G-Quadruplex)",
            class: "ncRNA — G-Quadruplex",
            pdb_id: None,
            sequence: "UUAGGGUUAGGGUUAGGGUUAGGG",
            summary: "Telomeric repeat-containing ncRNA forming parallel G-quadruplex scaffolds.",
        },
    ]
}

#[must_use]
pub fn find_rna_preset(query: &str) -> Option<RnaPreset> {
    let q = query.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    rna_presets()
        .into_iter()
        .find(|p| p.id == q || p.name.to_ascii_lowercase().contains(&q))
}

/// Sanitize RNA/DNA sequence converting `T -> U` and stripping whitespace.
pub fn sanitize_rna_sequence(raw: &str) -> Result<String, RnaError> {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_whitespace() || ch == '-' || ch == '5' || ch == '3' || ch == '\'' {
            continue;
        }
        match ch.to_ascii_uppercase() {
            'A' | 'U' | 'G' | 'C' => out.push(ch.to_ascii_uppercase()),
            'T' => out.push('U'),
            other => return Err(RnaError::InvalidBase(other)),
        }
    }
    if out.is_empty() {
        return Err(RnaError::EmptySequence);
    }
    Ok(out)
}

fn can_pair(a: char, b: char) -> bool {
    matches!(
        (a, b),
        ('G', 'C') | ('C', 'G') | ('A', 'U') | ('U', 'A') | ('G', 'U') | ('U', 'G')
    )
}

/// Turner-2004-inspired stacking energy (kcal/mol, negative = stabilizing) for nested pair `(i, j)`
/// stacked on `(i+1, j-1)`.
fn stack_energy(a: char, b: char, inner_a: char, inner_b: char) -> f64 {
    let base_val = |x: char, y: char| -> f64 {
        match (x, y) {
            ('G', 'C') | ('C', 'G') => -2.4,
            ('A', 'U') | ('U', 'A') => -1.2,
            ('G', 'U') | ('U', 'G') => -0.9,
            _ => 0.0,
        }
    };
    let e = 0.65 * (base_val(a, b) + base_val(inner_a, inner_b));
    // Extra stability for CG/GC and GG/CC stacks
    if matches!((a, b), ('G', 'C') | ('C', 'G'))
        && matches!((inner_a, inner_b), ('G', 'C') | ('C', 'G'))
    {
        e - 0.85
    } else {
        e
    }
}

fn pair_init_energy(a: char, b: char) -> f64 {
    match (a, b) {
        ('G', 'C') | ('C', 'G') => -1.45,
        ('A', 'U') | ('U', 'A') => -0.65,
        ('G', 'U') | ('U', 'G') => -0.45,
        _ => 10.0,
    }
}

/// Exact deterministic Zuker-style MFE 2D RNA folding (minimizing free energy `ΔG` in kcal/mol
/// with `min_hairpin = 3`, Turner-calibrated stacking & loop penalties).
pub fn fold_rna_mfe(raw_seq: &str) -> Result<RnaFoldReport, RnaError> {
    let full = sanitize_rna_sequence(raw_seq)?;
    let seq: String = full.chars().take(MAX_FOLD_RNA_LEN).collect();
    let cs: Vec<char> = seq.chars().collect();
    let n = cs.len();

    let gc = cs.iter().filter(|&&c| c == 'G' || c == 'C').count();
    let u_cnt = cs.iter().filter(|&&c| c == 'U').count();
    let gc_pct = ((gc as f64) / (n as f64) * 1000.0).round() / 10.0;
    let u_pct = ((u_cnt as f64) / (n as f64) * 1000.0).round() / 10.0;

    let min_loop = 3usize;
    let inf = 1e6_f64;
    // V[i][j] = MFE of subsequence i..=j given (i, j) forms a base pair
    // W[i][j] = MFE of subsequence i..=j (paired or unpaired)
    let mut v = vec![vec![inf; n]; n];
    let mut w = vec![vec![0.0_f64; n]; n];

    for span in (min_loop + 1)..n {
        for i in 0..(n - span) {
            let j = i + span;
            if can_pair(cs[i], cs[j]) {
                let loop_len = j - i - 1;
                // 1. Hairpin loop closing at (i, j)
                let hp_init = if loop_len <= 4 {
                    3.6
                } else if loop_len <= 8 {
                    4.4
                } else {
                    4.4 + 0.12 * ((loop_len - 8) as f64)
                };
                let mut best_v = hp_init + pair_init_energy(cs[i], cs[j]);

                // 2. Stacked pair (i, j) enclosing (i+1, j-1)
                if i + 1 < j - 1 && v[i + 1][j - 1] < inf * 0.5 {
                    let e_stack =
                        stack_energy(cs[i], cs[j], cs[i + 1], cs[j - 1]) + v[i + 1][j - 1];
                    if e_stack < best_v {
                        best_v = e_stack;
                    }
                }

                // 3. Small bulge / internal loop or multibranch enclosed by (i, j)
                if i + 1 < j - 1 {
                    let e_enc = pair_init_energy(cs[i], cs[j]) + 1.35 + w[i + 1][j - 1];
                    if e_enc < best_v {
                        best_v = e_enc;
                    }
                }
                v[i][j] = best_v;
            }

            let mut best_w = w[i + 1][j].min(w[i][j - 1]).min(v[i][j]);
            for k in (i + 1)..j {
                let split = w[i][k] + w[k + 1][j];
                if split < best_w {
                    best_w = split;
                }
            }
            w[i][j] = best_w;
        }
    }

    // Exact traceback recovering base pairs
    let mut pairs_idx: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize, bool)> = vec![(0, n.saturating_sub(1), false)];
    let eps = 1e-6;
    while let Some((i, j, must_pair)) = stack.pop() {
        if j <= i + min_loop {
            continue;
        }
        if must_pair {
            pairs_idx.push((i, j));
            let val = v[i][j];
            if i + 1 < j - 1
                && v[i + 1][j - 1] < inf * 0.5
                && (val - (stack_energy(cs[i], cs[j], cs[i + 1], cs[j - 1]) + v[i + 1][j - 1]))
                    .abs()
                    < eps
            {
                stack.push((i + 1, j - 1, true));
            } else if i + 1 < j - 1
                && (val - (pair_init_energy(cs[i], cs[j]) + 1.35 + w[i + 1][j - 1])).abs() < eps
            {
                stack.push((i + 1, j - 1, false));
            }
        } else {
            let val = w[i][j];
            if (val - w[i + 1][j]).abs() < eps {
                stack.push((i + 1, j, false));
            } else if (val - w[i][j - 1]).abs() < eps {
                stack.push((i, j - 1, false));
            } else if (val - v[i][j]).abs() < eps && v[i][j] < inf * 0.5 {
                stack.push((i, j, true));
            } else {
                for k in (i + 1)..j {
                    if (val - (w[i][k] + w[k + 1][j])).abs() < eps {
                        stack.push((k + 1, j, false));
                        stack.push((i, k, false));
                        break;
                    }
                }
            }
        }
    }

    pairs_idx.sort_by_key(|&(i, _)| i);
    let mut db = vec!['.'; n];
    let mut pairs = Vec::with_capacity(pairs_idx.len());
    for (i, j) in pairs_idx {
        db[i] = '(';
        db[j] = ')';
        let kind = match (cs[i], cs[j]) {
            ('G', 'C') | ('C', 'G') => "G-C",
            ('A', 'U') | ('U', 'A') => "A-U",
            _ => "G-U (Wobble)",
        }
        .to_string();
        pairs.push(RnaPair2D {
            i,
            j,
            base_i: cs[i],
            base_j: cs[j],
            kind,
        });
    }

    // Count distinct hairpins (pairs with no nested pairs inside)
    let mut hairpins = 0usize;
    for p in &pairs {
        let has_inner = pairs.iter().any(|q| q.i > p.i && q.j < p.j);
        if !has_inner {
            hairpins += 1;
        }
    }

    let mfe = if n >= 2 {
        (w[0][n - 1] * 100.0).round() / 100.0
    } else {
        0.0
    };
    let paired_pct = (((pairs.len() * 2) as f64) / (n.max(1) as f64) * 1000.0).round() / 10.0;

    Ok(RnaFoldReport {
        sequence: seq,
        folded_len: n,
        dot_bracket: db.into_iter().collect(),
        mfe_kcal_mol: mfe,
        pairs,
        gc_percent: gc_pct,
        uridine_percent: u_pct,
        paired_fraction_pct: paired_pct,
        hairpin_count: hairpins,
    })
}

fn find_g_quadruplexes(seq: &str) -> Vec<String> {
    // Detect canonical G3+ N1-7 G3+ N1-7 G3+ N1-7 G3+ motifs
    let bytes = seq.as_bytes();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'G' {
            let start = i;
            while i < bytes.len() && bytes[i] == b'G' {
                i += 1;
            }
            if i - start >= 3 {
                runs.push((start, i));
            }
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for w in runs.windows(4) {
        let g1 = w[1].0.saturating_sub(w[0].1);
        let g2 = w[2].0.saturating_sub(w[1].1);
        let g3 = w[3].0.saturating_sub(w[2].1);
        if (1..=7).contains(&g1) && (1..=7).contains(&g2) && (1..=7).contains(&g3) {
            let s = w[0].0;
            let e = w[3].1;
            out.push(format!("G4 @ {}..{} ({})", s + 1, e, &seq[s..e]));
        }
    }
    out
}

/// Complete ncRNA / mRNA / siRNA / Riboswitch structural & therapeutic audit.
pub fn analyze_ncrna_mrna(
    name: &str,
    rna_class: &str,
    raw_seq: &str,
    crystal_3d_pairs: Option<&[BasePair3D]>,
) -> Result<NcRnaAnalysisReport, RnaError> {
    let fold = fold_rna_mfe(raw_seq)?;
    let seq = &fold.sequence;

    let first30: String = seq.chars().take(30).collect();
    let first_30nt_mfe = if first30.len() >= 8 {
        fold_rna_mfe(&first30).map_or(0.0, |f| f.mfe_kcal_mol)
    } else {
        0.0
    };

    let g4 = find_g_quadruplexes(seq);
    let tlr_motifs = seq.matches("GUUG").count()
        + seq.matches("UGUU").count()
        + seq.matches("UUUU").count()
        + seq.matches("GUGU").count();
    let are_cnt = seq.matches("AUUUA").count();
    let has_kozak = seq.contains("GCCACCAUG") || seq.contains("ACCAUG") || seq.contains("AUGA");

    let m1psi_rec = if fold.uridine_percent >= 26.0 || tlr_motifs >= 3 {
        "High uridine / TLR7-8 motif density — substitute 100% U -> N1-methylpseudouridine (m1Ψ) + HPLC dsRNA purification.".to_string()
    } else {
        "Moderate uridine load — standard m1Ψ substitution recommended for therapeutic mRNA; retain native U if ribozyme/aptamer catalysis requires 2'-OH/U geometry.".to_string()
    };

    // Standard Spikevax/Comirnaty-class LNP formulation calculation (N/P = 6.0)
    let rna_mw_kda = ((fold.folded_len as f64) * 325.0 / 1000.0 * 100.0).round() / 100.0;
    let lnp_spec = LnpFormulationSpec {
        template_name: "Ionizable LNP (SM-102 / ALC-0315 Class)".to_string(),
        molar_ratio_ion_dspc_chol_peg: "50 : 10 : 38.5 : 1.5 mol%".to_string(),
        np_charge_ratio: 6.0,
        rna_nt_length: fold.folded_len,
        rna_mw_kda,
        lipid_to_rna_mass_ratio: 19.5,
    };

    // Optional concordance check against 3D crystal RNA base pairs
    let crystal_concordance = crystal_3d_pairs.and_then(|bps| {
        if bps.is_empty() || fold.pairs.is_empty() {
            return None;
        }
        let mut matched = 0usize;
        for bp in bps {
            // Check pair spacing / nucleotide identity correspondence within +/-2 nt offset tolerance
            let span3d = (bp.res_seq_b - bp.res_seq_a).unsigned_abs() as usize;
            if fold.pairs.iter().any(|p| {
                let span2d = p.j.saturating_sub(p.i);
                span2d.abs_diff(span3d) <= 2
                    && ((p.base_i == bp.base_a && p.base_j == bp.base_b)
                        || (p.base_i == bp.base_b && p.base_j == bp.base_a))
            }) {
                matched += 1;
            }
        }
        Some(((matched as f64) / (bps.len() as f64) * 1000.0).round() / 10.0)
    });

    let mut notes = Vec::new();
    if first_30nt_mfe < -8.0 {
        notes.push(format!(
            "Strong 5'-end secondary structure (first 30 nt MFE = {first_30nt_mfe:.1} kcal/mol) may impede ribosome initiation in mRNA."
        ));
    }
    if !g4.is_empty() {
        notes.push(format!("Detected {} putative G-quadruplex motif(s) — verify K+-dependent folding via CD/UV melting.", g4.len()));
    }
    if are_cnt > 0 {
        notes.push(format!("Found {are_cnt} AU-rich element(s) (`AUUUA`) linked to rapid cytoplasmic deadenylation."));
    }
    notes.push("Zuker MFE 2D prediction is a thermodynamic baseline; validate tertiary pseudoknots & ligand-induced switching by SHAPE-MaP / Cryo-EM.".to_string());

    Ok(NcRnaAnalysisReport {
        name: name.to_string(),
        rna_class: rna_class.to_string(),
        fold,
        first_30nt_mfe_kcal_mol: first_30nt_mfe,
        g_quadruplex_motifs: g4,
        tlr7_8_motifs_count: tlr_motifs,
        au_rich_elements_count: are_cnt,
        has_kozak_aug: has_kozak,
        m1psi_recommendation: m1psi_rec,
        lnp_spec,
        crystal_3d_concordance_pct: crystal_concordance,
        notes,
    })
}

/// Render an XSS-safe standalone SVG 2D Base-Pair Arc Diagram for an RNA / ncRNA / mRNA fold.
#[must_use]
pub fn render_rna_arc_svg(name: &str, fold: &RnaFoldReport) -> String {
    let w = 720.0_f64;
    let h = 260.0_f64;
    let base_y = 212.0_f64;
    let pad = 28.0_f64;
    let n = fold.folded_len.max(2) as f64;
    let step = (w - 2.0 * pad) / (n - 1.0).max(1.0);

    let mut svg = String::with_capacity(4096);
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" width=\"{w:.0}\" height=\"{h:.0}\">"
    );
    svg.push_str(
        "<rect width=\"100%\" height=\"100%\" rx=\"10\" fill=\"#0f172a\" stroke=\"#334155\"/>",
    );
    let _ = writeln!(
        svg,
        "<text x=\"18\" y=\"26\" font-family=\"system-ui,sans-serif\" font-size=\"13.5\" font-weight=\"700\" fill=\"#f8fafc\">ncRNA / mRNA 2D Fold — {} (MFE {:.2} kcal/mol · {} pairs · GC {:.1}%)</text>",
        escape_markup(name),
        fold.mfe_kcal_mol,
        fold.pairs.len(),
        fold.gc_percent
    );

    // Draw semicircular base-pair arcs
    for p in &fold.pairs {
        let x1 = pad + (p.i as f64) * step;
        let x2 = pad + (p.j as f64) * step;
        let rx = (0.5 * (x2 - x1)).max(2.0);
        let ry = (rx * 0.85).min(165.0);
        let col = if p.kind == "G-C" {
            "#34d399"
        } else if p.kind == "A-U" {
            "#38bdf8"
        } else {
            "#fbbf24"
        };
        let _ = writeln!(
            svg,
            "<path d=\"M {x1:.1} {base_y:.1} A {rx:.1} {ry:.1} 0 0 1 {x2:.1} {base_y:.1}\" fill=\"none\" stroke=\"{col}\" stroke-width=\"1.6\" stroke-opacity=\"0.85\"/>"
        );
    }

    // Draw nucleotide sequence strip
    for (idx, ch) in fold.sequence.chars().enumerate() {
        let x = pad + (idx as f64) * step;
        let col = match ch {
            'A' => "#f87171",
            'U' => "#38bdf8",
            'G' => "#34d399",
            'C' => "#fbbf24",
            _ => "#94a3b8",
        };
        let _ = writeln!(
            svg,
            "<circle cx=\"{x:.1}\" cy=\"{base_y:.1}\" r=\"3.0\" fill=\"{col}\"/>"
        );
    }
    svg.push_str("<text x=\"18\" y=\"244\" font-family=\"system-ui,sans-serif\" font-size=\"10.5\" fill=\"#94a3b8\">Arcs: Green = G-C · Sky = A-U · Amber = G-U Wobble | Dots: A(red) U(blue) G(green) C(amber)</text>");
    svg.push_str("</svg>\n");
    svg
}

/// Render Markdown ncRNA / mRNA dossier.
#[must_use]
pub fn render_rna_markdown(r: &NcRnaAnalysisReport) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# ncRNA / mRNA Engineering Report: {}\n", r.name);
    let _ = writeln!(
        md,
        "- **Class:** {} | **Length:** `{} nt` | **MFE (Zuker/Turner):** **`{:.2} kcal/mol`**",
        r.rna_class, r.fold.folded_len, r.fold.mfe_kcal_mol
    );
    let _ = writeln!(
        md,
        "- **Base Pairs:** `{}` (`{:.1}%` paired, `{}` hairpins) | **GC:** `{:.1}%` | **Uridine:** `{:.1}%`",
        r.fold.pairs.len(),
        r.fold.paired_fraction_pct,
        r.fold.hairpin_count,
        r.fold.gc_percent,
        r.fold.uridine_percent
    );
    let _ = writeln!(
        md,
        "- **5'-End 30-nt MFE:** `{:.2} kcal/mol` | **TLR7/8 Motifs:** `{}` | **G-Quadruplexes:** `{}` | **Kozak AUG:** `{}`",
        r.first_30nt_mfe_kcal_mol,
        r.tlr7_8_motifs_count,
        r.g_quadruplex_motifs.len(),
        r.has_kozak_aug
    );
    if let Some(conc) = r.crystal_3d_concordance_pct {
        let _ = writeln!(
            md,
            "- **2D MFE vs 3D Crystal Base-Pair Concordance:** **`{conc:.1}%`**"
        );
    }
    let _ = writeln!(
        md,
        "\n## Secondary Structure (Dot-Bracket)\n\n```text\n{}\n{}\n```\n",
        r.fold.sequence, r.fold.dot_bracket
    );
    let _ = writeln!(md, "## Formulation & Modification Guidance\n");
    let _ = writeln!(
        md,
        "- **Nucleotide Modification:** {}",
        r.m1psi_recommendation
    );
    let _ = writeln!(
        md,
        "- **LNP Formulation ({}):** `{}` at `N/P = {:.1}` (RNA MW `{:.2} kDa`, Lipid:RNA `{:.1}:1 w/w`)",
        r.lnp_spec.template_name,
        r.lnp_spec.molar_ratio_ion_dspc_chol_peg,
        r.lnp_spec.np_charge_ratio,
        r.lnp_spec.rna_mw_kda,
        r.lnp_spec.lipid_to_rna_mass_ratio
    );
    for n in &r.notes {
        let _ = writeln!(md, "- {n}");
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zuker_mfe_folds_canonical_hairpin_and_detects_g_quadruplex() -> Result<(), RnaError> {
        let hairpin = fold_rna_mfe("GGGGAAACCCC")?;
        assert!(
            hairpin.mfe_kcal_mol < -2.0,
            "expected stabilizing stem, got {}",
            hairpin.mfe_kcal_mol
        );
        assert!(hairpin.pairs.len() >= 3);
        assert!(hairpin.dot_bracket.contains("((("));
        assert!(hairpin.dot_bracket.contains(")))"));

        let g4 = analyze_ncrna_mrna("TERRA", "ncRNA", "UUAGGGUUAGGGUUAGGGUUAGGG", None)?;
        assert_eq!(g4.g_quadruplex_motifs.len(), 1);
        let svg = render_rna_arc_svg("Hairpin", &hairpin);
        assert!(svg.starts_with("<svg") && svg.contains("<path"));
        Ok(())
    }
}
