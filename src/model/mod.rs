use crate::config::{SaTConfig, SaTConfigError};
use ort::execution_providers::cpu::CPU;
use ort::session::Session;
use tokenizers::Tokenizer;
const TOKENIZER_ID: &str = "facebookAI/xlm-roberta-base";
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelLoadError {
    #[error("Failed to load tokenizer: {0}")]
    TokenizerLoadError(String),

    #[error("Failed to load ort session: {0}")]
    OrtError(#[from] ort::Error),

    #[error("Failed to load model from file: {0}")]
    SessionError(#[from] ort::Error<ort::session::builder::SessionBuilder>),

    #[error("Configuration error: {0}")]
    ConfigError(#[from] SaTConfigError),
}

pub fn load_tokenizer() -> Result<Tokenizer, ModelLoadError> {
    Tokenizer::from_pretrained(TOKENIZER_ID, None).map_err(|e| ModelLoadError::TokenizerLoadError(e.to_string()))
}

pub fn load_ort_model(config: &SaTConfig) -> Result<Session, ModelLoadError> {
    let mut builder = Session::builder()?
        .with_execution_providers([CPU::default().build()])?
        .with_optimization_level(match config.optimization_level {
            0 => ort::session::builder::GraphOptimizationLevel::Disable,
            1 => ort::session::builder::GraphOptimizationLevel::Level1,
            2 => ort::session::builder::GraphOptimizationLevel::Level2,
            3 => ort::session::builder::GraphOptimizationLevel::Level3,
            _ => {
                let config_err = SaTConfigError::InvalidOptimizationLevel(config.optimization_level);
                return Err(ModelLoadError::ConfigError(config_err));
            }
        })?;

    let session = builder.commit_from_file(&config.model_path)?;

    Ok(session)
}
