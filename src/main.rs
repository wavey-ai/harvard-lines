use harvard_lines::AudioProcessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = AudioProcessor::new(true, 10, 16_000, 500);
    let extracted =
        processor.process_pcm_numbered("Harvard.txt", "HARVARD_raw", "HARVARD_lines_pcm")?;
    println!("Extracted {extracted} PCM clips into HARVARD_lines_pcm");
    Ok(())
}
