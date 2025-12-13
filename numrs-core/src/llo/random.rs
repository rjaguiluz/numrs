use serde::{Serialize, Deserialize};

/// Random-related LLO kinds (rand, randn, randint, seed)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RandomKind {
    Rand,
    Randn,
    RandInt,
    Seed,
}

impl Default for RandomKind {
    fn default() -> Self { RandomKind::Rand }
}

// Future: add RNG state descriptors or distribution parameters
