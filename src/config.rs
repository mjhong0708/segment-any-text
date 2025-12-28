use hf_hub::api::sync::Api;
use std::path::PathBuf;
use thiserror::Error;

const MODEL_REPO: &str = "segment-any-text";
const ONNX_MODEL_FILE: &str = "model_optimized.onnx";

#[derive(Error, Debug)]
pub enum SaTConfigError {
    #[error("Invalid optimization level: {0}. Valid levels are 0, 1, 2, or 3.")]
    InvalidOptimizationLevel(usize),

    #[error("Hugging Face Hub error: {0}")]
    HuggingFaceHubError(#[from] hf_hub::api::sync::ApiError),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

#[derive(Clone, Copy, Debug)]
pub enum WeightingScheme {
    Uniform,
    Hat,
}

#[derive(Clone, Debug)]
pub struct SaTConfig {
    pub model_name: Option<String>,
    pub model_path: PathBuf,
    pub optimization_level: usize,
    pub threshold: f32,
    pub batch_size: usize,
    pub block_size: usize,
    pub stride: usize,
    pub weighting: WeightingScheme,
    pub strip_whitespace: bool,
    pub split_on_newline: bool,
}

impl SaTConfig {
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::new()
    }
}
impl Default for SaTConfig {
    fn default() -> Self {
        SaTConfig {
            model_name: None,
            model_path: std::path::PathBuf::new(),
            optimization_level: 3,
            threshold: 0.25,
            batch_size: 32,
            block_size: 512,
            stride: 64,
            weighting: WeightingScheme::Uniform,
            strip_whitespace: true,
            split_on_newline: true,
        }
    }
}

#[derive(Default)]
pub struct ConfigBuilder {
    config: SaTConfig,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        ConfigBuilder {
            config: SaTConfig::default(),
        }
    }

    /// The model name to use in https://huggingface.co/segment-any-text.
    /// It will automatically download the model if not present locally.
    /// Also, the default sentence splitting threshold will be set based on the model name.
    /// - `sm (supervised mixture)` models: 0.25
    /// - `no-limited-lookahead` models without sm: 0.01
    /// - other models: 0.025
    pub fn model_name(mut self, model_name: &str) -> Self {
        self.config.model_name = Some(model_name.to_string());
        self.config.threshold = default_theshold(model_name);
        self
    }

    /// The local path to the model file.
    pub fn model_path<P>(mut self, model_path: P) -> Self
    where
        P: AsRef<PathBuf>,
    {
        self.config.model_path = model_path.as_ref().to_path_buf();
        self
    }

    pub fn optimization_level(mut self, level: usize) -> Result<Self, SaTConfigError> {
        if level > 3 {
            return Err(SaTConfigError::InvalidOptimizationLevel(level));
        }
        self.config.optimization_level = level;
        Ok(self)
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.threshold = threshold;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn block_size(mut self, block_size: usize) -> Self {
        self.config.block_size = block_size;
        self
    }

    pub fn stride(mut self, stride: usize) -> Self {
        self.config.stride = stride;
        self
    }

    pub fn weighting_scheme(mut self, scheme: WeightingScheme) -> Self {
        self.config.weighting = scheme;
        self
    }

    pub fn strip_whitespace(mut self, strip: bool) -> Self {
        self.config.strip_whitespace = strip;
        self
    }

    pub fn split_on_newline(mut self, split: bool) -> Self {
        self.config.split_on_newline = split;
        self
    }

    pub fn build(mut self) -> Result<SaTConfig, SaTConfigError> {
        let config_err = |msg: &str| Err(SaTConfigError::InvalidConfiguration(msg.to_string()));

        if let Some(model_name) = &self.config.model_name {
            if !self.config.model_path.as_os_str().is_empty() {
                return config_err("Both model_name and model_path are set. Please set only one of them.");
            }

            let api = Api::new()?;
            let repo = api.model(format!("{}/{}", MODEL_REPO, model_name));
            self.config.model_path = repo.get(ONNX_MODEL_FILE)?;
        } else if self.config.model_path.as_os_str().is_empty() {
            return config_err("Could not locate the model. Please provide a model name or local path.");
        }
        Ok(self.config)
    }
}

fn default_theshold(model_name: &str) -> f32 {
    match model_name.contains("sm") {
        true => 0.25,
        false => match model_name.contains("no-limited-lookahead") {
            true => 0.01,
            false => 0.025,
        },
    }
}
