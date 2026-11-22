#![forbid(unsafe_code)]

//! Pure-Rust HLA-DRB1 MHC-II T-Cell Epitope Immunogenicity Predictor (8 global supertypes via
//! TepitoPe P1/P4/P6/P7/P9 pocket profiles) with Human Germline Self-Tolerance Subtraction and
//! Anchor-Residue De-Immunization Suggestions.

use std::collections::HashSet;

use bio_core::{AntibodyAnnotation, human_germline_sequences};
use serde::{Deserialize, Serialize};

/// Flagged non-self 9-mer MHC-II binding hotspot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCellEpitopeHotspot {
    pub chain: String,
    pub start_pos_1based: usize,
    pub region: String,
    pub peptide_9mer: String,
    pub hit_allele_count: usize,
    pub matched_alleles: Vec<String>,
    pub max_binding_score: f64,
    pub deimmunize_suggestion: String,
}

/// Complete MHC-II immunogenicity & de-immunization report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmunogenicityReport {
    pub total_9mers_scanned: usize,
    pub germline_tolerized_hits: usize,
    pub non_self_strong_binders: usize,
    pub risk_tier: String,
    pub hotspots: Vec<TCellEpitopeHotspot>,
}

const HLA_ALLELES: &[&str] = &[
    "DRB1*01:01",
    "DRB1*03:01",
    "DRB1*04:01",
    "DRB1*07:01",
    "DRB1*08:01",
    "DRB1*11:01",
    "DRB1*13:01",
    "DRB1*15:01",
];

/// TepitoPe-class pocket-profile preference for P1 (idx 0), P4 (idx 3), P6 (idx 5), P7 (idx 6), P9 (idx 8).
fn pocket_score(allele_idx: usize, aa: char, pocket: usize) -> f64 {
    match pocket {
        // P1: primary aromatic/aliphatic anchor across all HLA-DRB1 alleles
        0 => match (allele_idx, aa) {
            (_, 'Y' | 'F' | 'W') => 2.2,
            (7, 'I' | 'L' | 'V') => 2.4, // DRB1*15:01 prefers aliphatic P1
            (_, 'L' | 'I' | 'M' | 'V') => 1.7,
            (_, 'A' | 'T') => 0.2,
            (_, 'D' | 'E' | 'K' | 'R' | 'P') => -2.2,
            _ => -0.6,
        },
        // P4: allele-discriminating pocket (acidic vs basic vs hydrophobic)
        3 => match (allele_idx, aa) {
            (1, 'D' | 'E') => 1.9,             // DRB1*03:01 acidic P4
            (2, 'D' | 'E' | 'M' | 'Q') => 1.4, // DRB1*04:01
            (4..=6, 'R' | 'K' | 'H') => 1.6,   // DRB1*08/11/13 basic P4
            (0 | 3 | 7, 'L' | 'I' | 'M' | 'F' | 'Y' | 'A') => 1.3,
            (_, 'P') => -1.5,
            _ => 0.0,
        },
        // P6: small polar / aliphatic / charged preferences
        5 => match (allele_idx, aa) {
            (0 | 2 | 3, 'A' | 'S' | 'T' | 'N' | 'V') => 1.2,
            (4..=6, 'K' | 'R' | 'S' | 'T') => 1.3,
            (1 | 7, 'K' | 'R' | 'I' | 'L' | 'V') => 1.1,
            (_, 'W' | 'F' | 'Y') => -1.0,
            _ => 0.0,
        },
        // P7: secondary hydrophobic/polar groove
        6 => match aa {
            'L' | 'I' | 'V' | 'F' | 'Y' | 'M' => 0.8,
            'S' | 'T' | 'N' | 'Q' => 0.4,
            'D' | 'E' => -0.7,
            _ => 0.0,
        },
        // P9: C-terminal anchor
        8 => match (allele_idx, aa) {
            (0 | 2 | 3, 'L' | 'A' | 'V' | 'M' | 'F') => 1.2,
            (1 | 4 | 5 | 6, 'Y' | 'F' | 'L' | 'I' | 'K') => 1.2,
            (7, 'V' | 'I' | 'L' | 'M') => 1.3,
            (_, 'P' | 'G') => -0.8,
            _ => 0.0,
        },
        _ => 0.0,
    }
}

fn score_9mer_allele(w: &[char], allele_idx: usize) -> f64 {
    pocket_score(allele_idx, w[0], 0)
        + pocket_score(allele_idx, w[3], 3)
        + pocket_score(allele_idx, w[5], 5)
        + pocket_score(allele_idx, w[6], 6)
        + pocket_score(allele_idx, w[8], 8)
}

/// Evaluate HLA-DRB1 T-cell epitope risk across annotated chains with germline self-peptide
/// tolerance subtraction and anchor-residue de-immunization recommendations.
#[must_use]
pub fn evaluate_immunogenicity(annotations: &[AntibodyAnnotation]) -> ImmunogenicityReport {
    // Build human germline self-tolerance 9-mer set (plus classic human IgG Fc Tregitopes)
    let mut self_9mers: HashSet<String> = HashSet::new();
    for gseq in human_germline_sequences() {
        let cs: Vec<char> = gseq.chars().collect();
        for w in cs.windows(9) {
            self_9mers.insert(w.iter().collect());
        }
    }
    // Common conserved human framework junction 9-mers
    for conserved in [
        "WVRQAPGKG",
        "WVRQAPGQG",
        "WYQQKPGKA",
        "WYQQKPGQA",
        "RFTISRDNS",
        "RFTISRDNA",
        "LQMNSLRAE",
        "AEDTAVYYC",
        "PEDFATYYC",
        "WGQGTLVTV",
        "FGQGTKVEI",
    ] {
        self_9mers.insert(conserved.to_string());
    }

    let mut total_scanned = 0usize;
    let mut tolerized_hits = 0usize;
    let mut hotspots = Vec::new();
    let bind_thresh = 4.85_f64;

    for ann in annotations {
        let cs: Vec<char> = ann.sequence.chars().collect();
        if cs.len() < 9 {
            continue;
        }
        for i in 0..=(cs.len() - 9) {
            total_scanned += 1;
            let w = &cs[i..i + 9];
            let pep: String = w.iter().collect();
            let mut matched = Vec::new();
            let mut max_s = f64::NEG_INFINITY;
            for (ai, &aname) in HLA_ALLELES.iter().enumerate() {
                let s = score_9mer_allele(w, ai);
                if s > max_s {
                    max_s = s;
                }
                if s >= bind_thresh {
                    matched.push(aname.to_string());
                }
            }
            if matched.len() < 3 {
                continue;
            }
            if self_9mers.contains(&pep) {
                tolerized_hits += 1;
                continue;
            }
            // Also check 8/9 identity to human germline self-peptides (high central tolerance)
            let near_self = self_9mers
                .iter()
                .any(|s| s.chars().zip(w.iter()).filter(|(a, b)| a == *b).count() >= 8);
            if near_self && matched.len() < 5 {
                tolerized_hits += 1;
                continue;
            }

            let region = ann
                .numbered
                .get(i)
                .map_or("OTHER", |r| r.region.short())
                .to_string();

            // Recommend P1 anchor disruption (prefer non-CDR site if P1 is in FR, else P4/P9)
            let p1_in_cdr = ann.numbered.get(i).is_some_and(|r| r.region.is_cdr());
            let suggestion = if !p1_in_cdr {
                format!(
                    "Disrupt P1 anchor {}{} -> A/S/D (or humanize framework)",
                    w[0],
                    i + 1
                )
            } else {
                format!(
                    "Disrupt P4 ({}{}) or P9 ({}{}) -> A/S/T while retaining CDR contact",
                    w[3],
                    i + 4,
                    w[8],
                    i + 9
                )
            };

            hotspots.push(TCellEpitopeHotspot {
                chain: ann.chain_name.clone(),
                start_pos_1based: i + 1,
                region,
                peptide_9mer: pep,
                hit_allele_count: matched.len(),
                matched_alleles: matched,
                max_binding_score: (max_s * 100.0).round() / 100.0,
                deimmunize_suggestion: suggestion,
            });
        }
    }

    hotspots.sort_by(|a, b| {
        b.hit_allele_count
            .cmp(&a.hit_allele_count)
            .then_with(|| b.max_binding_score.total_cmp(&a.max_binding_score))
    });

    let n_strong = hotspots.len();
    let risk_tier = if n_strong <= 4 {
        "Low (Human-like T-cell epitope burden)".to_string()
    } else if n_strong <= 10 {
        "Moderate (Verify non-self epitopes in MAPPs / PBMC CD4+ T-cell assay)".to_string()
    } else {
        "High (De-immunization / framework humanization strongly recommended)".to_string()
    };

    ImmunogenicityReport {
        total_9mers_scanned: total_scanned,
        germline_tolerized_hits: tolerized_hits,
        non_self_strong_binders: n_strong,
        risk_tier,
        hotspots,
    }
}
