#![forbid(unsafe_code)]

//! Layer 3 (`bio-engine`): Kernel-confined `openat2` PathSandbox, CraftBio Ecosystem IPC,
//! and unified `.biocraft.json` workspace project persistence.

pub mod ecosystem;
pub mod sandbox;

pub use ecosystem::{
    EcosystemStatus, audit_payload_with_admetcraft, detect_ecosystem, send_to_biomolcraft,
    spawn_companion_app,
};
pub use sandbox::{DEFAULT_MAX_FILE_BYTES, PathSandbox, SandboxError};

use bio_adc::{AdcDesignReport, BiologicsPkReport};
use bio_develop::{DevelopabilityReport, FormatDecisionMatrix};
use bio_interface::{AlaScanReport, InterfaceReport};
use serde::{Deserialize, Serialize};

/// Portable BioCraft project snapshot persisted as `.biocraft.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioProjectSnapshot {
    pub schema_version: String,
    pub project_name: String,
    pub preset_id: Option<String>,
    pub developability: Option<DevelopabilityReport>,
    pub interface: Option<InterfaceReport>,
    pub alascan: Option<AlaScanReport>,
    pub adc: Option<AdcDesignReport>,
    pub format_matrix: Option<FormatDecisionMatrix>,
    pub biopk: Option<BiologicsPkReport>,
}
