use super::sat::{SaT, SaTModelError};
use super::utils::{chunk_with_overlap, generate_weights, indices_to_sentences, sigmoid, token_to_char_logits};
use crate::constants::{CLS_TOKEN_ID, SEP_TOKEN_ID};
use half::f16;
use ndarray::{Array1, Array2, Array3};
use ort::value::TensorRef;
use thiserror::Error;
use tokenizers::Encoding;

const MAX_BLOCK_SIZE_INTERNAL: usize = 510;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model inference error: {0}")]
    ModelError(#[from] SaTModelError),

    #[error("Failed to process tensor data: {0}")]
    TensorDataError(String),

    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug)]
pub struct SaTOutput {
    split_on_newline: bool,
    strip_whitespace: bool,
    pub token_logits: Vec<f32>,
    pub char_logits: Vec<f32>,
    pub sentence_probs: Vec<f32>,
    pub sentence_boundaries: Vec<usize>,
}

impl SaTOutput {
    pub fn sentences<'a>(&'a self, text: &'a str) -> impl Iterator<Item = String> + 'a {
        let sentences_iter = indices_to_sentences(text, self.sentence_boundaries.iter().cloned());

        let split_sentence = move |s: String| -> Vec<String> {
            if self.split_on_newline {
                s.split('\n').map(str::to_string).collect::<Vec<_>>()
            } else {
                vec![s]
            }
        };
        let trim_sentence = move |s: String| -> String { if self.strip_whitespace { s.trim().to_string() } else { s } };
        let retain_sentence = move |s: &String| -> bool { !self.strip_whitespace || !s.is_empty() };

        sentences_iter
            .flat_map(move |s| split_sentence(s).into_iter())
            .map(trim_sentence)
            .filter(retain_sentence)
    }
}

impl SaT {
    pub fn run_model(&self, text: &str) -> Result<SaTOutput, InferenceError> {
        let encoding = self.encode_text(text)?;
        let token_logits = self.predict_token_logits(&encoding)?;
        let char_logits = token_to_char_logits(text, &encoding, &token_logits);
        let sentence_probs = sigmoid(&char_logits);
        let boundary_idx: Vec<usize> = sentence_probs
            .iter()
            .enumerate()
            .filter_map(|(idx, &p)| (p > self.threshold).then_some(idx))
            .collect();

        let output = SaTOutput {
            split_on_newline: self.split_on_newline,
            strip_whitespace: self.strip_whitespace,
            token_logits: token_logits.to_vec(),
            char_logits: char_logits.to_vec(),
            sentence_probs: sentence_probs.to_vec(),
            sentence_boundaries: boundary_idx,
        };
        Ok(output)
    }

    fn predict_token_logits(&self, encoding: &Encoding) -> Result<Array1<f32>, InferenceError> {
        let token_ids = encoding.get_ids();
        let num_tokens = token_ids.len();

        if num_tokens == 0 {
            return Ok(Array1::from(Vec::new()));
        }

        let block_size = num_tokens.min(self.block_size).min(MAX_BLOCK_SIZE_INTERNAL);

        let weights = generate_weights(block_size, self.weighting);
        let mut all_logits = vec![0f32; num_tokens];
        let mut all_counts = vec![0f32; num_tokens];

        let chunk_slice_indices: Vec<(usize, usize)> =
            chunk_with_overlap(num_tokens, block_size, self.stride).collect();
        for batch_slice_indices in chunk_slice_indices.chunks(self.batch_size) {
            let actual_batch_size = batch_slice_indices.len();
            let input_shape = (actual_batch_size, block_size + 2); // +2 for CLS and SEP tokens

            let mut input_ids: Array2<i64> = Array2::zeros(input_shape);
            let mut attention_mask: Array2<f16> = Array2::<f32>::zeros(input_shape).mapv(f16::from_f32);

            for (row, &(start, end)) in batch_slice_indices.iter().enumerate() {
                let chunk_iter = std::iter::once(CLS_TOKEN_ID as i64)
                    .chain(token_ids[start..end].iter().map(|id| *id as i64))
                    .chain(std::iter::once(SEP_TOKEN_ID as i64));

                for (col, id) in chunk_iter.enumerate() {
                    input_ids[(row, col)] = id;
                    attention_mask[(row, col)] = f16::from_f32(1.0);
                }
            }

            let (_, logits_half) = self.run_ort_session(
                TensorRef::from_array_view(input_ids.view())
                    .map_err(|e| InferenceError::TensorDataError(format!("Failed to create input_ids tensor: {e}")))?,
                TensorRef::from_array_view(attention_mask.view()).map_err(|e| {
                    InferenceError::TensorDataError(format!("Failed to create attention_mask tensor: {e}"))
                })?,
            )?;
            let logits = Array3::from_shape_vec(
                (actual_batch_size, block_size + 2, 1),
                logits_half.into_iter().map(f32::from).collect(),
            )
            .map_err(|e| InferenceError::TensorDataError(format!("Failed to reshape logits: {e}")))?;

            for (row, &(start, end)) in batch_slice_indices.iter().enumerate() {
                let n = end - start;
                for offset in 0..n {
                    let logit = logits[[row, offset + 1, 0]]; // skip CLS
                    let weight = weights[offset];
                    all_logits[start + offset] += weight * logit;
                    all_counts[start + offset] += weight;
                }
            }
        }

        for i in 0..num_tokens {
            if all_counts[i] != 0.0 {
                all_logits[i] /= all_counts[i];
            }
        }

        Array1::from_shape_vec(num_tokens, all_logits)
            .map_err(|e| InferenceError::TensorDataError(format!("Failed to create token logits array: {e}")))
    }
}
