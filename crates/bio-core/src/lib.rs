#![forbid(unsafe_code)]

//! Layer 0 (`bio-core`): Clean-room FASTA/PDB/mmCIF parsers, Shrake-Rupley 3D SASA,
//! IMGT/Kabat antibody & VHH numbering, human germline alignment, CDR-grafting humanization,
//! and curated clinical biologics presets.

pub mod imgt;
pub mod presets;
pub mod structure;

pub use imgt::{
    AntibodyAnnotation, GermlineHit, HumanizationProposal, IgChainType, NumberedResidue,
    RegionKind, annotate_chain, human_germline_sequences, render_annotation_markdown,
};
pub use presets::{ClinicalBiologicPreset, clinical_biologic_presets, find_biologic_preset};
pub use structure::{
    Atom3D, BioError, Chain3D, DisulfideBond, MAX_BIO_ATOMS, ProteinStructure, Residue3D,
    aa1_to_aa3, aa3_to_aa1, black_mould_sap_hydro, compute_shrake_rupley_sasa, dist3,
    kyte_doolittle, max_residue_sasa, parse_fasta, parse_mmcif, parse_pdb, parse_structure_auto,
    residue_avg_mw, sanitize_protein_sequence, vdw_radius,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trastuzumab_exact_imgt_cdrs_and_vhh_hallmark_detection() -> Result<(), BioError> {
        let trast = find_biologic_preset("trastuzumab").ok_or(BioError::EmptyInput)?;
        let vh = annotate_chain("Trastuzumab_VH", trast.vh_or_vhh_seq);
        let vl = annotate_chain("Trastuzumab_VL", trast.vl_seq.ok_or(BioError::EmptyInput)?);
        assert_eq!(vh.chain_type, IgChainType::Vh);
        assert_eq!(vh.cdr1_imgt, "GFNIKDTY");
        assert_eq!(vh.cdr2_imgt, "IYPTNGYT");
        assert_eq!(vh.cdr3_imgt, "SRWGGDGFYAMDY");
        assert_eq!(vh.top_germlines[0].name, "IGHV3-66*01");
        assert!(vh.humanness_score >= 90.0);

        assert_eq!(vl.chain_type, IgChainType::Vk);
        assert_eq!(vl.cdr1_imgt, "QDVNTA");
        assert_eq!(vl.cdr2_imgt, "SAS");
        assert_eq!(vl.cdr3_imgt, "QQHYTTPPT");
        assert_eq!(vl.top_germlines[0].name, "IGKV1-39*01");

        // VHH nanobodies must be detected automatically via FR2 hallmark tetrad
        for id in ["envafolimab", "caplacizumab"] {
            let p = find_biologic_preset(id).ok_or(BioError::EmptyInput)?;
            let vhh = annotate_chain(p.name, p.vh_or_vhh_seq);
            assert_eq!(vhh.chain_type, IgChainType::Vhh, "{id} should be VHH");
            assert!(vhh.vhh_hallmark_tetrad.is_some());
            if id == "envafolimab" {
                assert_eq!(vhh.extra_noncanonical_cys.len(), 2);
                assert_eq!(vhh.cdr3_imgt, "AADSFEDPTCTLVTSSGAFQY");
            }
            // Humanization must preserve VHH FR2 solubility hallmarks!
            let hum = vhh.humanization.as_ref().ok_or(BioError::EmptyInput)?;
            assert!(!hum.preserved_vhh_hallmarks.is_empty());
        }
        Ok(())
    }

    #[test]
    fn shrake_rupley_sasa_matches_isolated_sphere_and_buries_dimer() {
        let single = vec![Atom3D {
            serial: 1,
            name: "CA".to_string(),
            element: "C".to_string(),
            res_name: "ALA".to_string(),
            res_aa1: 'A',
            chain: 'A',
            res_seq: 1,
            icode: ' ',
            coords: [0.0, 0.0, 0.0],
            occupancy: 1.0,
            b_factor: 10.0,
            is_hetatm: false,
        }];
        let sasa1 = compute_shrake_rupley_sasa(&single, 1.4, 92);
        let expected = 4.0 * std::f64::consts::PI * (1.70 + 1.40_f64).powi(2);
        assert!((sasa1[0] - expected).abs() < 0.01);

        let mut pair = single;
        pair.push(Atom3D {
            serial: 2,
            coords: [1.54, 0.0, 0.0],
            ..pair[0].clone()
        });
        let sasa2 = compute_shrake_rupley_sasa(&pair, 1.4, 92);
        assert!(sasa2[0] < sasa1[0] * 0.85);
    }
}
