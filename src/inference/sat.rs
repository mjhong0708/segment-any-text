use crate::config::{SaTConfig, WeightingScheme};
use crate::model::{ModelLoadError, load_ort_model, load_tokenizer};
use half::f16;
use ort::session::Session;
use ort::tensor::Shape;
use ort::value::TensorRef;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokenizers::{Encoding, Tokenizer};

#[derive(Error, Debug)]
pub enum SaTModelError {
    #[error("Failed to acquire lock on ONNX session: {0}")]
    SessionLock(String),

    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Failed to run tokenizer: {0}")]
    TokenizerRun(String),
}
pub struct SaT {
    tokenizer: Arc<Tokenizer>,
    ort_session: Arc<Mutex<Session>>,
    pub optimization_level: usize,
    pub threshold: f32,
    pub batch_size: usize,
    pub block_size: usize,
    pub stride: usize,
    pub weighting: WeightingScheme,
    pub strip_whitespace: bool,
    pub split_on_newline: bool,
}

impl SaT {
    pub fn new(config: SaTConfig) -> Result<Self, ModelLoadError> {
        let tokenizer = load_tokenizer()?;
        let session = load_ort_model(&config)?;
        Ok(SaT {
            tokenizer: Arc::new(tokenizer),
            ort_session: Arc::new(Mutex::new(session)),
            optimization_level: config.optimization_level,
            threshold: config.threshold,
            batch_size: config.batch_size,
            block_size: config.block_size,
            stride: config.stride,
            weighting: config.weighting,
            strip_whitespace: config.strip_whitespace,
            split_on_newline: config.split_on_newline,
        })
    }

    /// Encode the input text into token IDs and attention masks.
    /// The special tokens are not added; CLS and SEP are added per-chunk.
    pub fn encode_text(&self, text: &str) -> Result<Encoding, SaTModelError> {
        self.tokenizer
            .encode(text, false)
            .map_err(|e| SaTModelError::TokenizerRun(e.to_string()))
    }

    /// Run the onnx session with the given inputs.
    pub fn run_ort_session(
        &self,
        input_ids: TensorRef<i64>,
        attention_mask: TensorRef<f16>,
    ) -> Result<(Shape, Vec<f16>), SaTModelError> {
        let mut session = self
            .ort_session
            .lock()
            .map_err(|e| SaTModelError::SessionLock(e.to_string()))?;
        let outputs = session.run(ort::inputs![input_ids, attention_mask])?;
        let (dims, logits) = outputs[0].try_extract_tensor::<f16>()?;

        Ok((dims.to_owned(), logits.to_vec()))
    }
}
