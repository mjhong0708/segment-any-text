use crate::config::WeightingScheme;
use crate::constants::{CLS_TOKEN_ID, PAD_TOKEN_ID, SEP_TOKEN_ID};
use ndarray::Array1;
use tokenizers::Encoding;

const SPECIAL_TOKENS: [u32; 3] = [CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID];

pub fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn indices_to_sentences<'a, I>(text: &'a str, indices: I) -> impl Iterator<Item = String> + 'a
where
    I: IntoIterator<Item = usize> + 'a,
    I::IntoIter: 'a,
{
    let mut boundaries = indices.into_iter();
    let mut start = 0;
    let mut tail_emitted = false;

    std::iter::from_fn(move || {
        loop {
            if let Some(boundary) = boundaries.next() {
                let mut i = boundary + 1;
                while i < text.len() && text.as_bytes()[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i > start {
                    let sentence = text[start..i].to_string();
                    start = i;
                    return Some(sentence);
                }
                start = i;
            } else if !tail_emitted && start < text.len() {
                tail_emitted = true;
                let sentence = text[start..].to_string();
                start = text.len();
                return Some(sentence);
            } else {
                return None;
            }
        }
    })
}

pub fn token_to_char_logits(text: &str, encoding: &Encoding, token_logits: &Array1<f32>) -> Array1<f32> {
    let token_ids = encoding.get_ids();
    let offsets = encoding.get_offsets();

    let mut token_indices = Vec::new();
    let mut char_indices = Vec::new();
    for (idx, (&token_id, offset)) in token_ids.iter().zip(offsets.iter()).enumerate() {
        if SPECIAL_TOKENS.contains(&token_id) {
            continue;
        }
        if offset.1 == 0 {
            continue;
        }
        token_indices.push(idx);
        char_indices.push(offset.1 - 1);
    }

    let mut char_logits = vec![-f32::INFINITY; text.len()];
    if token_indices.is_empty() {
        return Array1::from_vec(char_logits);
    }

    let token_values = token_logits.clone();

    for (&token_idx, &char_idx) in token_indices.iter().zip(char_indices.iter()) {
        if token_idx < token_values.len() && char_idx < char_logits.len() {
            char_logits[char_idx] = token_values[token_idx];
        }
    }

    Array1::from_vec(char_logits)
}

pub fn generate_weights(length: usize, scheme: WeightingScheme) -> Array1<f32> {
    match scheme {
        WeightingScheme::Uniform => Array1::ones(length),
        WeightingScheme::Hat => {
            let max = 1.0 - 1.0 / length as f32;
            Array1::linspace(-max, max, length).mapv(|x| 1.0 - x.abs())
        }
    }
}
pub fn chunk_with_overlap(length: usize, block_size: usize, stride: usize) -> impl Iterator<Item = (usize, usize)> {
    // Final start index to ensure last chunk covers the end
    let final_start = length.saturating_sub(block_size);
    // Stride over indices to yield (start, end) pairs
    (0..final_start)
        .step_by(stride)
        .map(move |i| (i, i + block_size))
        .chain(std::iter::once((final_start, length))) // last chunk
}

#[cfg(test)]
mod tests {
    use super::chunk_with_overlap;
    #[test]
    fn test_chunk_with_overlap() {
        let length = 10;
        let block_size = 8;
        let stride = 4;
        let chunks: Vec<(usize, usize)> = chunk_with_overlap(length, block_size, stride).collect();
        let expected_chunks = vec![(0, 8), (2, 10)];
        assert_eq!(chunks, expected_chunks);
    }
}
