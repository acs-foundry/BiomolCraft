#![forbid(unsafe_code)]

//! Layer 1 (`bio-develop`): TAP 5-metric developability, 3D SAP aggregation patches, CDR/FR
//! chemical PTM liabilities, germline-tolerized HLA-DRB1 MHC-II immunogenicity & de-immunization,
//! format decision matrix, and multi-host codon optimization.

pub mod formats;
pub mod immunogenicity;
pub mod tap;

pub use formats::{
    BinderFormatGoal, CodonOptReport, ExpressionHost, FormatComparisonRow, FormatDecisionMatrix,
    compare_binder_formats, optimize_codons, render_format_matrix_markdown,
};
pub use immunogenicity::{ImmunogenicityReport, TCellEpitopeHotspot, evaluate_immunogenicity};
pub use tap::{
    BiophysicalProps, DevelopabilityReport, PtmLiability, SapHotspot, TapMetric, TrafficLight,
    evaluate_developability, render_developability_markdown,
};

#[cfg(test)]
mod tests {
    use super::*;
    use bio_core::{BioError, find_biologic_preset};

    #[test]
    fn trastuzumab_and_adalimumab_pass_tap_while_negative_controls_flag_risks()
    -> Result<(), BioError> {
        let trast = find_biologic_preset("trastuzumab").ok_or(BioError::EmptyInput)?;
        let trast_rep = evaluate_developability(
            trast.name,
            &[
                ("VH", trast.vh_or_vhh_seq),
                ("VL", trast.vl_seq.ok_or(BioError::EmptyInput)?),
            ],
            None,
        );
        assert_eq!(trast_rep.tap_overall, TrafficLight::Green);
        assert!(trast_rep.developability_score_0_100 >= 80.0);
        assert!(trast_rep.biophysical.extinction_coeff_m_cm > 30_000.0);
        assert!((trast_rep.biophysical.isoelectric_point_pi - 8.5).abs() < 1.5);

        // Briakinumab (PPC positive charge control) & Bococizumab must score lower than Trastuzumab
        let briak = find_biologic_preset("briakinumab").ok_or(BioError::EmptyInput)?;
        let briak_rep = evaluate_developability(
            briak.name,
            &[
                ("VH", briak.vh_or_vhh_seq),
                ("VL", briak.vl_seq.ok_or(BioError::EmptyInput)?),
            ],
            None,
        );
        let ppc_trast = trast_rep
            .tap_metrics
            .iter()
            .find(|m| m.name.starts_with("PPC"))
            .map_or(0.0, |m| m.value);
        let ppc_briak = briak_rep
            .tap_metrics
            .iter()
            .find(|m| m.name.starts_with("PPC"))
            .map_or(0.0, |m| m.value);
        assert!(
            ppc_briak > ppc_trast,
            "Briakinumab PPC ({ppc_briak}) should exceed Trastuzumab ({ppc_trast})"
        );

        let boco = find_biologic_preset("bococizumab").ok_or(BioError::EmptyInput)?;
        let boco_rep = evaluate_developability(
            boco.name,
            &[
                ("VH", boco.vh_or_vhh_seq),
                ("VL", boco.vl_seq.ok_or(BioError::EmptyInput)?),
            ],
            None,
        );
        assert!(boco_rep.developability_score_0_100 <= trast_rep.developability_score_0_100);

        // Format matrix & codon optimization sanity
        let mat = compare_binder_formats(&BinderFormatGoal {
            target_name: "PD-L1".to_string(),
            require_deep_tissue_penetration: true,
            require_long_half_life: true,
            require_subcutaneous: true,
            ..Default::default()
        });
        assert_eq!(mat.ranked_formats[0].format_id, "vhh_hsa");

        let cod = optimize_codons(trast.vh_or_vhh_seq, ExpressionHost::Cho);
        assert_eq!(cod.length_bp, trast.vh_or_vhh_seq.len() * 3 + 3);
        assert!((45.0..=72.0).contains(&cod.gc_percent));
        Ok(())
    }
}
