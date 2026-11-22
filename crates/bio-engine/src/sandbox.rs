#![forbid(unsafe_code)]

//! Unified Kernel-Confined `openat2` PathSandbox re-exported directly from shared `craftbio-libs::craftbio-sandbox`.

pub use craftbio_sandbox::{DEFAULT_MAX_FILE_BYTES, PathSandbox, SandboxError};
