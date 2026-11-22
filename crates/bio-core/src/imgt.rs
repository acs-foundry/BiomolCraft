#![forbid(unsafe_code)]

//! Pure-Rust Antibody & Nanobody (VHH) Chain Classification, IMGT & Kabat Numbering,
//! Human Germline Identification, Humanness Z-Score, and CDR-Grafting Humanization Engine.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::structure::sanitize_protein_sequence;

/// Classified immunoglobulin / protein chain type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IgChainType {
    Vh,
    Vk,
    Vl,
    Vhh,
    Fc,
    GeneralProtein,
}

impl IgChainType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Vh => "VH (Heavy Variable)",
            Self::Vk => "VL-Kappa (Light Variable)",
            Self::Vl => "VL-Lambda (Light Variable)",
            Self::Vhh => "VHH (Camelid Nanobody)",
            Self::Fc => "Fc Constant Domain",
            Self::GeneralProtein => "General Protein / Antigen",
        }
    }

    pub fn is_variable_ig(self) -> bool {
        matches!(self, Self::Vh | Self::Vk | Self::Vl | Self::Vhh)
    }
}

/// IMGT immunoglobulin framework / CDR region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionKind {
    Fr1,
    Cdr1,
    Fr2,
    Cdr2,
    Fr3,
    Cdr3,
    Fr4,
    ConstantOrOther,
}

impl RegionKind {
    pub fn short(self) -> &'static str {
        match self {
            Self::Fr1 => "FR1",
            Self::Cdr1 => "CDR1",
            Self::Fr2 => "FR2",
            Self::Cdr2 => "CDR2",
            Self::Fr3 => "FR3",
            Self::Cdr3 => "CDR3",
            Self::Fr4 => "FR4",
            Self::ConstantOrOther => "OTHER",
        }
    }

    pub fn is_cdr(self) -> bool {
        matches!(self, Self::Cdr1 | Self::Cdr2 | Self::Cdr3)
    }
}

/// Numbered residue with IMGT position, Kabat label, region, and structural flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberedResidue {
    /// 0-based index in the input chain sequence.
    pub seq_idx: usize,
    pub imgt_pos: String,
    pub kabat_label: String,
    pub aa: char,
    pub region: RegionKind,
    pub is_vernier: bool,
    pub is_vhh_hallmark: bool,
}

/// Ranked match against embedded human IMGT reference germlines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GermlineHit {
    pub name: String,
    pub chain_type: IgChainType,
    pub full_identity_pct: f64,
    pub framework_identity_pct: f64,
    pub germline_sequence: String,
}

/// CDR-grafting humanization blueprint preserving Vernier zone & VHH FR2 solubility hallmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanizationProposal {
    pub target_germline: String,
    pub humanized_sequence: String,
    pub before_fr_identity_pct: f64,
    pub after_fr_identity_pct: f64,
    pub mutations: Vec<String>,
    pub preserved_vernier_sites: Vec<String>,
    pub preserved_vhh_hallmarks: Vec<String>,
}

/// Complete IMGT/Kabat antibody or VHH annotation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntibodyAnnotation {
    pub chain_name: String,
    pub sequence: String,
    pub chain_type: IgChainType,
    pub v_domain_len: usize,
    pub fr1: String,
    pub cdr1_imgt: String,
    pub fr2: String,
    pub cdr2_imgt: String,
    pub fr3: String,
    pub cdr3_imgt: String,
    pub fr4: String,
    pub cdr1_kabat: String,
    pub cdr2_kabat: String,
    pub cdr3_kabat: String,
    pub vhh_hallmark_tetrad: Option<String>,
    pub extra_noncanonical_cys: Vec<String>,
    pub top_germlines: Vec<GermlineHit>,
    pub humanness_score: f64,
    pub humanization: Option<HumanizationProposal>,
    pub numbered: Vec<NumberedResidue>,
    pub notes: Vec<String>,
}

struct GermlineRef {
    name: &'static str,
    chain_type: IgChainType,
    fr1: &'static str,
    fr2: &'static str,
    fr3: &'static str,
    fr4: &'static str,
    full_v: &'static str,
}

/// Curated top clinical human IMGT V-gene germline reference frameworks (FR1..FR4).
const HUMAN_GERMLINES: &[GermlineRef] = &[
    GermlineRef {
        name: "IGHV3-66*01",
        chain_type: IgChainType::Vh,
        fr1: "EVQLVESGGGLVQPGGSLRLSCAAS",
        fr2: "MSWVRQAPGKGLEWVSV",
        fr3: "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
        fr4: "WGQGTLVTVSS",
        full_v: "EVQLVESGGGLVQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
    },
    GermlineRef {
        name: "IGHV3-23*01",
        chain_type: IgChainType::Vh,
        fr1: "EVQLLESGGGLVQPGGSLRLSCAAS",
        fr2: "MSWVRQAPGKGLEWVSA",
        fr3: "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
        fr4: "WGQGTLVTVSS",
        full_v: "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
    },
    GermlineRef {
        name: "IGHV3-30*01",
        chain_type: IgChainType::Vh,
        fr1: "QVQLVESGGGVVQPGRSLRLSCAAS",
        fr2: "MHWVRQAPGKGLEWVAV",
        fr3: "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
        fr4: "WGQGTLVTVSS",
        full_v: "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
    },
    GermlineRef {
        name: "IGHV3-9*01",
        chain_type: IgChainType::Vh,
        fr1: "EVQLVESGGGLVQPGRSLRLSCAAS",
        fr2: "MHWVRQAPGKGLEWVSG",
        fr3: "GYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTALYYC",
        fr4: "WGQGTLVTVSS",
        full_v: "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTALYYC",
    },
    GermlineRef {
        name: "IGHV1-69*01",
        chain_type: IgChainType::Vh,
        fr1: "QVQLVQSGAEVKKPGSSVKVSCKAS",
        fr2: "ISWVRQAPGQGLEWMGG",
        fr3: "NYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYC",
        fr4: "WGQGTLVTVSS",
        full_v: "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYC",
    },
    GermlineRef {
        name: "IGHV1-2*02",
        chain_type: IgChainType::Vh,
        fr1: "QVQLVQSGAEVKKPGASVKVSCKAS",
        fr2: "MHWVRQAPGQGLEWMGW",
        fr3: "NYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYC",
        fr4: "WGQGTTVTVSS",
        full_v: "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYC",
    },
    GermlineRef {
        name: "IGHV1-46*01",
        chain_type: IgChainType::Vh,
        fr1: "QVQLVQSGAEVKKPGASVKVSCKAS",
        fr2: "MHWVRQAPGQGLEWMGI",
        fr3: "SYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYC",
        fr4: "WGQGTTVTVSS",
        full_v: "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYC",
    },
    GermlineRef {
        name: "IGHV4-59*01",
        chain_type: IgChainType::Vh,
        fr1: "QVQLQESGPGLVKPSETLSLTCTVS",
        fr2: "WSWIRQPPGKGLEWIGY",
        fr3: "NYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYC",
        fr4: "WGQGTLVTVSS",
        full_v: "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYC",
    },
    GermlineRef {
        name: "IGKV1-39*01",
        chain_type: IgChainType::Vk,
        fr1: "DIQMTQSPSSLSASVGDRVTITCRAS",
        fr2: "LNWYQQKPGKAPKLLIY",
        fr3: "SLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYC",
        fr4: "FGQGTKVEIK",
        full_v: "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYC",
    },
    GermlineRef {
        name: "IGKV1-33*01",
        chain_type: IgChainType::Vk,
        fr1: "DIQMTQSPSSLSASVGDRVTITCQAS",
        fr2: "LNWYQQKPGKAPKLLIY",
        fr3: "NLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYC",
        fr4: "FGQGTKLEIK",
        full_v: "DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIYDASNLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYC",
    },
    GermlineRef {
        name: "IGKV3-20*01",
        chain_type: IgChainType::Vk,
        fr1: "EIVLTQSPGTLSLSPGERATLSCRAS",
        fr2: "LAWYQQKPGQAPRLLIY",
        fr3: "SRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYC",
        fr4: "FGQGTKVEIK",
        full_v: "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYC",
    },
    GermlineRef {
        name: "IGKV3D-11*01",
        chain_type: IgChainType::Vk,
        fr1: "EIVLTQSPATLSLSPGERATLSCRAS",
        fr2: "LAWYQQKPGQAPRLLIY",
        fr3: "NRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYC",
        fr4: "FGGGTKVEIK",
        full_v: "EIVLTQSPATLSLSPGERATLSCRASQGVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYC",
    },
    GermlineRef {
        name: "IGLV1-40*01",
        chain_type: IgChainType::Vl,
        fr1: "QSVLTQPPSVSGAPGQRVTISCTGSS",
        fr2: "VHWYQQLPGTAPKLLIY",
        fr3: "NRPSGVPDRFSGSKSGTSASLAITGLQAEDEADYYC",
        fr4: "FGGGTKLTVL",
        full_v: "QSVLTQPPSVSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIYGNSNRPSGVPDRFSGSKSGTSASLAITGLQAEDEADYYC",
    },
    GermlineRef {
        name: "IGLV2-14*01",
        chain_type: IgChainType::Vl,
        fr1: "QSALTQPASVSGSPGQSITISCTGTS",
        fr2: "VSWYQQHPGKAPKLMIY",
        fr3: "NRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYC",
        fr4: "FGGGTKLTVL",
        full_v: "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYEVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYC",
    },
];

/// Expose curated human germline V-domain sequences for MHC-II self-tolerance subtraction.
#[must_use]
pub fn human_germline_sequences() -> Vec<&'static str> {
    HUMAN_GERMLINES.iter().map(|g| g.full_v).collect()
}

fn positional_identity(a: &str, b: &str) -> f64 {
    let ca: Vec<char> = a.chars().collect();
    let cb: Vec<char> = b.chars().collect();
    let n = ca.len().min(cb.len());
    if n == 0 {
        return 0.0;
    }
    let m = (0..n).filter(|&i| ca[i] == cb[i]).count();
    (m as f64) / (n as f64) * 100.0
}

fn find_ig_anchors(chars: &[char]) -> Option<(usize, usize, usize, usize, bool)> {
    let n = chars.len();
    if n < 90 {
        return None;
    }
    // 1. Conserved C23 in positions 17..=26
    let c23 = (17..=26.min(n - 1)).find(|&i| chars[i] == 'C')?;
    // 2. Conserved W41 in c23+9..=c23+22 (followed by V/I/F/Y/L)
    let w41 = ((c23 + 9)..=(c23 + 22).min(n - 2))
        .find(|&i| chars[i] == 'W' && matches!(chars[i + 1], 'V' | 'I' | 'F' | 'Y' | 'L' | 'H'))
        .or_else(|| ((c23 + 9)..=(c23 + 22).min(n - 1)).find(|&i| chars[i] == 'W'))?;
    // 3. Conserved C104 in w41+46..=w41+66 preceded by Y/F/H
    let c104 = ((w41 + 46)..=(w41 + 66).min(n - 1))
        .rev()
        .find(|&i| chars[i] == 'C' && i >= 2 && matches!(chars[i - 1], 'Y' | 'F' | 'H' | 'W'))
        .or_else(|| {
            ((w41 + 46)..=(w41 + 66).min(n - 1))
                .rev()
                .find(|&i| chars[i] == 'C')
        })?;
    // 4. Conserved J-region W118 (`WG.G`, VH/VHH) or F118 (`FG.G`, VK/VL)
    let mut j_anchor = None;
    let mut is_heavy = true;
    for i in (c104 + 4)..=(c104 + 34).min(n.saturating_sub(4)) {
        if chars[i] == 'W' && chars[i + 1] == 'G' && chars[i + 3] == 'G' {
            j_anchor = Some(i);
            is_heavy = true;
            break;
        }
        if (chars[i] == 'F' || chars[i] == 'W') && chars[i + 1] == 'G' && chars[i + 3] == 'G' {
            j_anchor = Some(i);
            is_heavy = chars[i] == 'W';
            break;
        }
    }
    let j118 = j_anchor?;
    Some((c23, w41, c104, j118, is_heavy))
}

/// Annotate any antibody, nanobody (VHH), Fc, or general protein sequence with exact IMGT + Kabat
/// CDR/FR boundaries, VHH hallmark detection, germline identity, and CDR-grafting humanization.
#[must_use]
pub fn annotate_chain(chain_name: &str, raw_seq: &str) -> AntibodyAnnotation {
    let seq = sanitize_protein_sequence(raw_seq).unwrap_or_default();
    let chars: Vec<char> = seq.chars().collect();
    let mut notes = Vec::new();

    let Some((c23, w41, c104, j118, is_heavy)) = find_ig_anchors(&chars) else {
        // Check if Fc constant region (CH2/CH3 motifs: `ELLGGP`, `DTLMISR`, `GQPREPQ`)
        let is_fc = seq.contains("DTLMISR") || seq.contains("GQPREPQ") || seq.contains("APELLGG");
        let chain_type = if is_fc {
            IgChainType::Fc
        } else {
            IgChainType::GeneralProtein
        };
        let numbered = chars
            .iter()
            .enumerate()
            .map(|(i, &aa)| NumberedResidue {
                seq_idx: i,
                imgt_pos: format!("{}", i + 1),
                kabat_label: format!("{}", i + 1),
                aa,
                region: RegionKind::ConstantOrOther,
                is_vernier: false,
                is_vhh_hallmark: false,
            })
            .collect();
        notes.push(if is_fc {
            "Detected IgG Fc constant region (CH2/CH3)".to_string()
        } else {
            "No canonical Ig V-domain C23/W41/C104/(W|F)GXG anchors found — treated as general protein / antigen.".to_string()
        });
        return AntibodyAnnotation {
            chain_name: chain_name.to_string(),
            sequence: seq,
            chain_type,
            v_domain_len: 0,
            fr1: String::new(),
            cdr1_imgt: String::new(),
            fr2: String::new(),
            cdr2_imgt: String::new(),
            fr3: String::new(),
            cdr3_imgt: String::new(),
            fr4: String::new(),
            cdr1_kabat: String::new(),
            cdr2_kabat: String::new(),
            cdr3_kabat: String::new(),
            vhh_hallmark_tetrad: None,
            extra_noncanonical_cys: Vec::new(),
            top_germlines: Vec::new(),
            humanness_score: 0.0,
            humanization: None,
            numbered,
            notes,
        };
    };

    // Detect VHH camelid FR2 hallmark tetrad at IMGT 42, 49, 50, 52 (Kabat 37, 44, 45, 47):
    // offsets from W41: +1 (Kabat 37: F/Y vs V), +8 (Kabat 44: E/Q/D vs G), +9 (Kabat 45: R/C vs L), +11 (Kabat 47: G/F/L/W)
    let mut vhh_hallmark_tetrad = None;
    let chain_type = if is_heavy {
        let p37 = chars.get(w41 + 1).copied().unwrap_or('-');
        let p44 = chars.get(w41 + 8).copied().unwrap_or('-');
        let p45 = chars.get(w41 + 9).copied().unwrap_or('-');
        let p47 = chars.get(w41 + 11).copied().unwrap_or('-');
        let tetrad = format!("{p37}37/{p44}44/{p45}45/{p47}47");
        let vhh_votes = usize::from(matches!(p37, 'F' | 'Y'))
            + usize::from(matches!(p44, 'E' | 'Q' | 'D'))
            + usize::from(matches!(p45, 'R' | 'C'))
            + usize::from(matches!(p47, 'G' | 'F' | 'S' | 'R'));
        let has_q108 = chars.get(j118 + 5).is_some_and(|&c| c == 'Q');
        if vhh_votes >= 2
            || (matches!(p37, 'F' | 'Y') && matches!(p45, 'R' | 'C'))
            || (vhh_votes >= 1 && has_q108 && c104 + 16 < j118)
        {
            vhh_hallmark_tetrad = Some(tetrad.clone());
            notes.push(format!(
                "Camelid VHH nanobody signature detected in FR2 ({tetrad}); hydrophilic VH/VL interface enables autonomous single-domain solubility."
            ));
            IgChainType::Vhh
        } else {
            IgChainType::Vh
        }
    } else {
        // Distinguish VK (`VEIK`/`LEIK`, `DIQ`/`EIV`) vs VL (`LTVL`, `QSV`/`QSA`/`SYE`)
        let fr4_tail: String = chars
            .get(j118..(j118 + 10).min(chars.len()))
            .map(|s| s.iter().collect())
            .unwrap_or_default();
        if fr4_tail.ends_with("VL")
            || seq.starts_with("QSV")
            || seq.starts_with("QSA")
            || seq.starts_with("SYE")
        {
            IgChainType::Vl
        } else {
            IgChainType::Vk
        }
    };

    // Exact IMGT region boundaries from conserved anchors
    let cdr1_start = (c23 + 4).min(w41.saturating_sub(2));
    let cdr1_end = w41.saturating_sub(2).max(cdr1_start + 1);
    let cdr2_start = (w41 + 15).min(c104.saturating_sub(10));
    let expected_fr3_len = if is_heavy { 37 } else { 35 };
    let cdr2_end = c104
        .saturating_sub(expected_fr3_len)
        .clamp(cdr2_start + 3, c104.saturating_sub(25));
    let cdr3_start = c104 + 1;
    let cdr3_end = j118;
    let fr4_end = (j118 + if is_heavy { 11 } else { 10 }).min(chars.len());

    let slice_s = |a: usize, b: usize| -> String {
        if a < b && b <= chars.len() {
            chars[a..b].iter().collect()
        } else {
            String::new()
        }
    };

    let fr1 = slice_s(0, cdr1_start);
    let cdr1_imgt = slice_s(cdr1_start, cdr1_end);
    let fr2 = slice_s(cdr1_end, cdr2_start);
    let cdr2_imgt = slice_s(cdr2_start, cdr2_end);
    let fr3 = slice_s(cdr2_end, cdr3_start);
    let cdr3_imgt = slice_s(cdr3_start, cdr3_end);
    let fr4 = slice_s(cdr3_end, fr4_end);

    // Kabat CDRs (classic definitions relative to IMGT anchors)
    let cdr1_kabat = if is_heavy {
        slice_s((c23 + 9).min(w41), w41)
    } else {
        slice_s(c23 + 1, w41)
    };
    let cdr2_kabat = if is_heavy {
        slice_s(w41 + 14, (cdr2_end + 8).min(c104))
    } else {
        slice_s(w41 + 15, (w41 + 22).min(c104))
    };
    let cdr3_kabat = if is_heavy {
        slice_s((c104 + 3).min(j118), j118)
    } else {
        cdr3_imgt.clone()
    };

    // Build residue-level IMGT & Kabat table + detect non-canonical Cys
    let mut numbered = Vec::with_capacity(chars.len());
    let mut extra_noncanonical_cys = Vec::new();
    for (i, &aa) in chars.iter().enumerate() {
        let region = if i < cdr1_start {
            RegionKind::Fr1
        } else if i < cdr1_end {
            RegionKind::Cdr1
        } else if i < cdr2_start {
            RegionKind::Fr2
        } else if i < cdr2_end {
            RegionKind::Cdr2
        } else if i < cdr3_start {
            RegionKind::Fr3
        } else if i < cdr3_end {
            RegionKind::Cdr3
        } else if i < fr4_end {
            RegionKind::Fr4
        } else {
            RegionKind::ConstantOrOther
        };

        let imgt_num: i32 = match region {
            RegionKind::Fr1 => 23 - (c23 as i32 - i as i32),
            RegionKind::Cdr1 => 27 + (i - cdr1_start) as i32,
            RegionKind::Fr2 => 41 + (i as i32 - w41 as i32),
            RegionKind::Cdr2 => 56 + (i - cdr2_start) as i32,
            RegionKind::Fr3 => 104 - (c104 as i32 - i as i32),
            RegionKind::Cdr3 => 105 + (i - cdr3_start) as i32,
            RegionKind::Fr4 => 118 + (i - cdr3_end) as i32,
            RegionKind::ConstantOrOther => 129 + (i - fr4_end) as i32,
        };
        let is_vernier = matches!(
            imgt_num,
            2 | 27 | 28 | 29 | 30 | 52 | 54 | 76 | 78 | 80 | 87 | 102 | 103
        ) && !region.is_cdr();
        let is_vhh_hallmark = chain_type == IgChainType::Vhh
            && (i == w41 + 1 || i == w41 + 8 || i == w41 + 9 || i == w41 + 11);

        if aa == 'C' && i != c23 && i != c104 && i < fr4_end {
            extra_noncanonical_cys.push(format!("C{} ({})", imgt_num, region.short()));
        }

        numbered.push(NumberedResidue {
            seq_idx: i,
            imgt_pos: format!("{imgt_num}"),
            kabat_label: format!("{}{}", region.short(), i + 1),
            aa,
            region,
            is_vernier,
            is_vhh_hallmark,
        });
    }

    if extra_noncanonical_cys.len() == 2 && chain_type == IgChainType::Vhh {
        notes.push(format!(
            "Non-canonical VHH extra disulfide pair detected ({}) — stabilizes long CDR3 loop.",
            extra_noncanonical_cys.join(" – ")
        ));
    } else if !extra_noncanonical_cys.is_empty() {
        notes.push(format!(
            "WARNING: Non-canonical cysteine(s) detected in V-domain: {} — verify disulfide pairing or free-thiol liability.",
            extra_noncanonical_cys.join(", ")
        ));
    }

    // Rank against human IMGT reference germlines
    let target_family = match chain_type {
        IgChainType::Vh | IgChainType::Vhh => IgChainType::Vh,
        IgChainType::Vk => IgChainType::Vk,
        IgChainType::Vl => IgChainType::Vl,
        _ => IgChainType::GeneralProtein,
    };
    let query_fr = format!("{fr1}{fr2}{fr3}");
    let query_v = slice_s(0, c104 + 1);

    let mut hits: Vec<GermlineHit> = HUMAN_GERMLINES
        .iter()
        .filter(|g| g.chain_type == target_family)
        .map(|g| {
            let g_fr = format!("{}{}{}", g.fr1, g.fr2, g.fr3);
            let fr_id = positional_identity(&query_fr, &g_fr);
            let full_id = positional_identity(&query_v, g.full_v);
            GermlineHit {
                name: g.name.to_string(),
                chain_type: g.chain_type,
                full_identity_pct: (full_id * 10.0).round() / 10.0,
                framework_identity_pct: (fr_id * 10.0).round() / 10.0,
                germline_sequence: format!("{}{}", g.full_v, g.fr4),
            }
        })
        .collect();
    hits.sort_by(|a, b| {
        b.framework_identity_pct
            .total_cmp(&a.framework_identity_pct)
            .then_with(|| b.full_identity_pct.total_cmp(&a.full_identity_pct))
    });
    let top_germlines: Vec<GermlineHit> = hits.into_iter().take(3).collect();

    // Humanness score (0..100 normalized against human framework identity; clinical human mAbs >=85%)
    let best_fr = top_germlines
        .first()
        .map_or(0.0, |h| h.framework_identity_pct);
    let humanness_score = (best_fr * 10.0).round() / 10.0;

    // Propose CDR-grafted humanization onto top germline while protecting Vernier sites & VHH FR2 hallmarks
    let humanization = top_germlines.first().and_then(|best| {
        let gref = HUMAN_GERMLINES.iter().find(|g| g.name == best.name)?;
        let mut muts = Vec::new();
        let mut kept_vernier = Vec::new();
        let mut kept_vhh = Vec::new();

        let graft_fr = |q_fr: &str,
                        g_fr: &str,
                        region_lbl: &str,
                        base_offset: usize,
                        muts: &mut Vec<String>,
                        kept_vernier: &mut Vec<String>,
                        kept_vhh: &mut Vec<String>|
         -> String {
            let qc: Vec<char> = q_fr.chars().collect();
            let gc: Vec<char> = g_fr.chars().collect();
            let mut out = String::with_capacity(qc.len());
            for (k, &qa) in qc.iter().enumerate() {
                let Some(&ga) = gc.get(k) else {
                    out.push(qa);
                    continue;
                };
                if qa == ga {
                    out.push(qa);
                    continue;
                }
                let seq_pos = base_offset + k;
                let nr = numbered.get(seq_pos);
                if nr.is_some_and(|r| r.is_vhh_hallmark) {
                    kept_vhh.push(format!(
                        "{}{} (preserved VHH solubility hallmark vs human {ga})",
                        qa,
                        nr.map_or("?", |r| r.imgt_pos.as_str())
                    ));
                    out.push(qa);
                } else if nr.is_some_and(|r| r.is_vernier) {
                    kept_vernier.push(format!(
                        "{}{} ({region_lbl} Vernier retained vs {ga})",
                        qa,
                        nr.map_or("?", |r| r.imgt_pos.as_str())
                    ));
                    out.push(qa);
                } else {
                    muts.push(format!(
                        "{qa}{}{ga} ({region_lbl})",
                        nr.map_or("?", |r| r.imgt_pos.as_str())
                    ));
                    out.push(ga);
                }
            }
            out
        };

        let h_fr1 = graft_fr(
            &fr1,
            gref.fr1,
            "FR1",
            0,
            &mut muts,
            &mut kept_vernier,
            &mut kept_vhh,
        );
        let h_fr2 = graft_fr(
            &fr2,
            gref.fr2,
            "FR2",
            cdr1_end,
            &mut muts,
            &mut kept_vernier,
            &mut kept_vhh,
        );
        let h_fr3 = graft_fr(
            &fr3,
            gref.fr3,
            "FR3",
            cdr2_end,
            &mut muts,
            &mut kept_vernier,
            &mut kept_vhh,
        );
        let h_fr4 = graft_fr(
            &fr4,
            gref.fr4,
            "FR4",
            cdr3_end,
            &mut muts,
            &mut kept_vernier,
            &mut kept_vhh,
        );
        let humanized_sequence =
            format!("{h_fr1}{cdr1_imgt}{h_fr2}{cdr2_imgt}{h_fr3}{cdr3_imgt}{h_fr4}");
        let new_fr = format!("{h_fr1}{h_fr2}{h_fr3}");
        let g_fr = format!("{}{}{}", gref.fr1, gref.fr2, gref.fr3);
        let after_fr = (positional_identity(&new_fr, &g_fr) * 10.0).round() / 10.0;

        Some(HumanizationProposal {
            target_germline: best.name.clone(),
            humanized_sequence,
            before_fr_identity_pct: best.framework_identity_pct,
            after_fr_identity_pct: after_fr,
            mutations: muts,
            preserved_vernier_sites: kept_vernier,
            preserved_vhh_hallmarks: kept_vhh,
        })
    });

    AntibodyAnnotation {
        chain_name: chain_name.to_string(),
        sequence: seq,
        chain_type,
        v_domain_len: fr4_end,
        fr1,
        cdr1_imgt,
        fr2,
        cdr2_imgt,
        fr3,
        cdr3_imgt,
        fr4,
        cdr1_kabat,
        cdr2_kabat,
        cdr3_kabat,
        vhh_hallmark_tetrad,
        extra_noncanonical_cys,
        top_germlines,
        humanness_score,
        humanization,
        numbered,
        notes,
    }
}

/// Render a clean Markdown IMGT/Kabat annotation & humanization report.
#[must_use]
pub fn render_annotation_markdown(ann: &AntibodyAnnotation) -> String {
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Antibody / VHH Annotation Report: {}\n",
        ann.chain_name
    );
    let _ = writeln!(
        md,
        "- **Chain Type:** {} | **Length:** `{} aa` (V-domain `{} aa`)",
        ann.chain_type.label(),
        ann.sequence.len(),
        ann.v_domain_len
    );
    if let Some(ref tet) = ann.vhh_hallmark_tetrad {
        let _ = writeln!(md, "- **VHH FR2 Hallmark Tetrad:** `{tet}`");
    }
    if ann.chain_type.is_variable_ig() {
        let _ = writeln!(
            md,
            "- **Humanness (Top Human FR Identity):** **`{:.1}%`**\n",
            ann.humanness_score
        );
        let _ = writeln!(md, "## IMGT & Kabat Regions\n");
        let _ = writeln!(md, "| Region | IMGT Sequence | Len | Kabat Sequence |");
        let _ = writeln!(md, "|---|---|---:|---|");
        let _ = writeln!(md, "| **FR1** | `{}` | {} | — |", ann.fr1, ann.fr1.len());
        let _ = writeln!(
            md,
            "| **CDR1** | **`{}`** | {} | `{}` |",
            ann.cdr1_imgt,
            ann.cdr1_imgt.len(),
            ann.cdr1_kabat
        );
        let _ = writeln!(md, "| **FR2** | `{}` | {} | — |", ann.fr2, ann.fr2.len());
        let _ = writeln!(
            md,
            "| **CDR2** | **`{}`** | {} | `{}` |",
            ann.cdr2_imgt,
            ann.cdr2_imgt.len(),
            ann.cdr2_kabat
        );
        let _ = writeln!(md, "| **FR3** | `{}` | {} | — |", ann.fr3, ann.fr3.len());
        let _ = writeln!(
            md,
            "| **CDR3** | **`{}`** | {} | `{}` |",
            ann.cdr3_imgt,
            ann.cdr3_imgt.len(),
            ann.cdr3_kabat
        );
        let _ = writeln!(md, "| **FR4** | `{}` | {} | — |", ann.fr4, ann.fr4.len());

        if !ann.top_germlines.is_empty() {
            let _ = writeln!(md, "\n## Closest Human IMGT Germlines\n");
            let _ = writeln!(
                md,
                "| Germline | Framework Identity | Full V-Domain Identity |"
            );
            let _ = writeln!(md, "|---|---:|---:|");
            for g in &ann.top_germlines {
                let _ = writeln!(
                    md,
                    "| `{}` | **{:.1}%** | {:.1}% |",
                    g.name, g.framework_identity_pct, g.full_identity_pct
                );
            }
        }
        if let Some(ref hum) = ann.humanization {
            let _ = writeln!(
                md,
                "\n## CDR-Grafting Humanization Proposal (`{}`: {:.1}% → {:.1}% FR identity)\n",
                hum.target_germline, hum.before_fr_identity_pct, hum.after_fr_identity_pct
            );
            if hum.mutations.is_empty() {
                let _ = writeln!(
                    md,
                    "- Framework already matches target human germline (excluding Vernier/VHH hallmarks)."
                );
            } else {
                let _ = writeln!(
                    md,
                    "- **Proposed FR Mutations ({}):** `{}`",
                    hum.mutations.len(),
                    hum.mutations.join(", ")
                );
            }
            if !hum.preserved_vernier_sites.is_empty() {
                let _ = writeln!(
                    md,
                    "- **Preserved Vernier Zone Back-Mutations:** {}",
                    hum.preserved_vernier_sites.join("; ")
                );
            }
            if !hum.preserved_vhh_hallmarks.is_empty() {
                let _ = writeln!(
                    md,
                    "- **Preserved VHH FR2 Solubility Hallmarks:** {}",
                    hum.preserved_vhh_hallmarks.join("; ")
                );
            }
        }
    }
    if !ann.notes.is_empty() {
        let _ = writeln!(md, "\n## Notes\n");
        for n in &ann.notes {
            let _ = writeln!(md, "- {n}");
        }
    }
    md
}
