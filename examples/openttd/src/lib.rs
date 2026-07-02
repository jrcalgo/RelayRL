pub mod environment;
pub mod host;
pub mod openttd;
pub mod subsystems;
pub mod training;

pub const OPENTTD_REPOSITORY: &str = "https://github.com/OpenTTD/OpenTTD";
pub const OPENTTD_LATEST_STABLE: &str = "15.3";
pub const OPENTTD_LATEST_STABLE_PUBLISHED: &str = "2026-04-04";
pub const OPENTTD_LATEST_STABLE_URL: &str = "https://github.com/OpenTTD/OpenTTD/releases/tag/15.3";
pub const OPENTTD_NEWEST_PRERELEASE: &str = "16.0-beta1";
pub const OPENTTD_NEWEST_PRERELEASE_URL: &str =
    "https://github.com/OpenTTD/OpenTTD/releases/tag/16.0-beta1";

pub const OBSERVATION_DIM: usize = 32;
pub const ACTION_DIM: usize = 8;
