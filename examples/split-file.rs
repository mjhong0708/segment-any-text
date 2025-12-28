use anyhow::Result;

use clap::Parser;
use segment_any_text::{config::SaTConfig, inference::SaT};

/// An example to split a text file into sentences using SaT.
/// For models, see the huggingface page: https://huggingface.co/segment-any-text
#[derive(Debug, Parser)]
struct Opt {
    /// The name of the input text to split
    #[clap(short, long)]
    input_text: String,

    /// The name of model to use, default: sat-3l-sm
    #[clap(short, long, default_value = "sat-3l-sm")]
    model_name: String,

    /// To print more debug information
    #[clap(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let opt = Opt::parse();

    let time = std::time::Instant::now();
    let model = load_model(&opt.model_name)?;
    println!("Model loaded in {:?}", time.elapsed());
    let text = std::fs::read_to_string(&opt.input_text).expect("Failed to read input text file");

    if opt.verbose {
        println!("Using model: {}", opt.model_name);
        println!("Original text: {}", text);
    }

    let time = std::time::Instant::now();
    let output = model.run_model(&text)?;
    if opt.verbose {
        println!("Per-char boundary probs: {:?}", output.sentence_probs);
        println!("Sentence boundaries: {:?}", output.sentence_boundaries);
    }

    for (i, sentence) in output.sentences(&text).enumerate() {
        println!("Sentence {}: '{}'", i + 1, sentence);
    }

    let t_duration = time.elapsed();
    println!("Splitting took: {:?}", t_duration);
    Ok(())
}

pub fn load_model(model_name: &str) -> Result<SaT> {
    let config = SaTConfig::builder()
        .model_name(model_name)
        .optimization_level(3)?
        .strip_whitespace(true)
        .build()?;

    SaT::new(config).map_err(|e| e.into())
}
