#![forbid(unsafe_code)]

//! Layer 2 (`bio-adc`): Antibody-Drug Conjugate (ADC) Architect & Human Biologics
//! Minimal-PBPK + FcRn Recycling + Nonlinear QSS TMDD Simulator.

pub mod adc;
pub mod biopk;

pub use adc::{
    AdcDesignReport, AdcDesignRequest, AdcError, ConjugationStrategy, LinkerSpec, PayloadSpec,
    design_adc, linker_catalog, payload_catalog, render_adc_markdown,
};
pub use biopk::{
    BiologicsFormat, BiologicsPkReport, BiologicsPkRequest, DosingRoute, PkTimePoint,
    render_biopk_markdown, simulate_biologics_pk_tmdd,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tdxd_vs_tdm1_vs_dar8_mmae_differentiation() -> Result<(), AdcError> {
        let tdxd = design_adc(&AdcDesignRequest {
            name: "T-DXd".to_string(),
            antibody_mw_da: 148_000.0,
            linker_id: "ggfg".to_string(),
            payload_id: "dxd".to_string(),
            custom_payload_smiles: None,
            dar: 8.0,
            conjugation: ConjugationStrategy::InterchainCys,
            admetcraft_payload_summary: None,
        })?;
        let tdm1 = design_adc(&AdcDesignRequest {
            name: "T-DM1".to_string(),
            antibody_mw_da: 148_000.0,
            linker_id: "smcc".to_string(),
            payload_id: "dm1".to_string(),
            custom_payload_smiles: None,
            dar: 3.5,
            conjugation: ConjugationStrategy::RandomLys,
            admetcraft_payload_summary: None,
        })?;
        let bad_dar8_mmae = design_adc(&AdcDesignRequest {
            name: "DAR8-vcMMAE".to_string(),
            antibody_mw_da: 148_000.0,
            linker_id: "vc_pab".to_string(),
            payload_id: "mmae".to_string(),
            custom_payload_smiles: None,
            dar: 8.0,
            conjugation: ConjugationStrategy::InterchainCys,
            admetcraft_payload_summary: None,
        })?;

        // T-DXd has high bystander killing, while non-cleavable T-DM1 has low bystander killing
        assert!(tdxd.bystander_score_0_100 >= 75.0);
        assert!(tdm1.bystander_score_0_100 <= 25.0);
        // Hydrophilic GGFG keeps T-DXd DAR 8 HDI well below unmasked DAR 8 vc-MMAE
        assert!(tdxd.hydrophobic_drug_load_index < 8.0);
        assert!(bad_dar8_mmae.hydrophobic_drug_load_index > 18.0);
        assert!(bad_dar8_mmae.design_verdict.contains("HIGH RISK"));
        Ok(())
    }

    #[test]
    fn biologics_pk_tmdd_respects_fcrn_format_half_life_hierarchy() -> Result<(), AdcError> {
        let base = BiologicsPkRequest::default();
        let igg1 = simulate_biologics_pk_tmdd(&base)?;
        let yte = simulate_biologics_pk_tmdd(&BiologicsPkRequest {
            format: BiologicsFormat::Igg1YteLs,
            ..base.clone()
        })?;
        let vhh_fc = simulate_biologics_pk_tmdd(&BiologicsPkRequest {
            format: BiologicsFormat::VhhFcOrHsa,
            ..base.clone()
        })?;
        let vhh = simulate_biologics_pk_tmdd(&BiologicsPkRequest {
            format: BiologicsFormat::VhhMonomer,
            ..base
        })?;

        assert!(yte.linear_t_half_days > igg1.linear_t_half_days * 2.5);
        assert!(igg1.linear_t_half_days > vhh_fc.linear_t_half_days);
        assert!(vhh_fc.linear_t_half_days > vhh.linear_t_half_days * 20.0);
        assert!(
            igg1.ro_trough_ss_pct >= 95.0,
            "Trastuzumab 420mg Q3W should saturate HER2"
        );
        assert!(vhh.ro_trough_ss_pct < 5.0, "Naked VHH Q3W clears rapidly");
        Ok(())
    }
}
