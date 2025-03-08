use frame_header::{EncodingFlag, Endianness, FrameHeader};
use image::ImageBuffer;
use mel_spec::mel::MelSpectrogram;
use mel_spec::stft::Spectrogram;
use mel_spec::vad::{DetectionSettings, VoiceActivityDetector, as_image, vad_boundaries};
use ndarray::{Array1, Array2, Axis};
use regex::Regex;
use rubato::Resampler;
use soundkit::audio_bytes::{f32le_to_i16, s24le_to_i16};
use soundkit::audio_packet::Encoder;
use soundkit::wav::WavStreamProcessor;
use soundkit_opus::OpusEncoder;
use std::cmp;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;

#[derive(Debug)]
pub struct LineAudio {
    pub line_name: String,
    pub opus_frames: Vec<Vec<u8>>,
}

pub struct AudioProcessor {
    pub has_intro: bool,
    pub lines_in_file: usize,
    pub output_sample_rate: u32,
    pub pre_vad_padding_ms: u32,
}

impl AudioProcessor {
    pub fn new(
        has_intro: bool,
        lines_in_file: usize,
        output_sample_rate: u32,
        pre_vad_padding_ms: u32,
    ) -> Self {
        Self {
            has_intro,
            lines_in_file,
            output_sample_rate,
            pre_vad_padding_ms,
        }
    }

    fn read_harvard_file_lines(file_path: &str) -> Result<Vec<String>, io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let header_re = Regex::new(r"^H\d+").unwrap();
        let number_re = Regex::new(r"^\s*\d+\.\s*").unwrap();
        let mut sentences = Vec::new();
        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() || header_re.is_match(trimmed) {
                continue;
            }
            let sentence = number_re.replace(trimmed, "").to_string();
            sentences.push(sentence);
        }
        Ok(sentences)
    }

    fn preprocess_file(&self, path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer)?;
        let mut processor = WavStreamProcessor::new();
        let audio_data = processor
            .add(&file_buffer)?
            .ok_or("No audio data returned from WAV processing")?;
        let samples: Vec<f32> = match audio_data.bits_per_sample() {
            8 => {
                let u8_samples: &[u8] = audio_data.data();
                u8_samples
                    .iter()
                    .map(|&s| (s as f32 - 128.0) / 128.0)
                    .collect()
            }
            16 => {
                let sample_count = audio_data.data().len() / 2;
                let i16_samples: &[i16] = unsafe {
                    std::slice::from_raw_parts(
                        audio_data.data().as_ptr() as *const i16,
                        sample_count,
                    )
                };
                i16_samples.iter().map(|&s| s as f32 / 32767.0).collect()
            }
            24 => {
                let i16_samples = s24le_to_i16(audio_data.data());
                i16_samples.iter().map(|&s| s as f32 / 32767.0).collect()
            }
            32 => {
                if audio_data.audio_format() == EncodingFlag::PCMFloat {
                    unsafe {
                        std::slice::from_raw_parts(
                            audio_data.data().as_ptr() as *const f32,
                            audio_data.data().len() / 4,
                        )
                        .to_vec()
                    }
                } else {
                    let sample_count = audio_data.data().len() / 4;
                    let i32_samples: &[i32] = unsafe {
                        std::slice::from_raw_parts(
                            audio_data.data().as_ptr() as *const i32,
                            sample_count,
                        )
                    };
                    i32_samples
                        .iter()
                        .map(|&s| s as f32 / i32::MAX as f32)
                        .collect()
                }
            }
            _ => return Err("Unsupported bits per sample".into()),
        };

        let encoder_sample_rate = self.output_sample_rate;
        let original_sample_rate = audio_data.sampling_rate() as u32;
        if original_sample_rate == encoder_sample_rate {
            return Ok(samples);
        }
        let ratio = encoder_sample_rate as f64 / original_sample_rate as f64;
        let samples_f64: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
        let channels = 1;
        let input_data = vec![samples_f64];
        let sinc_len = 128;
        let oversampling_factor = 2048;
        let interpolation = rubato::SincInterpolationType::Linear;
        let window = rubato::WindowFunction::Blackman2;
        let f_cutoff = rubato::calculate_cutoff(sinc_len, window);
        let params = rubato::SincInterpolationParameters {
            sinc_len,
            f_cutoff,
            interpolation,
            oversampling_factor,
            window,
        };
        let chunksize = 1024;
        let max_ratio = 1.1;
        let mut resampler =
            rubato::SincFixedOut::<f64>::new(ratio, max_ratio, params, chunksize, channels)
                .map_err(|e| format!("Failed to create resampler: {:?}", e))?;
        let mut resampled_data: Vec<f64> = Vec::new();
        let mut pos = 0;
        let input_channel = &input_data[0];
        while pos < input_channel.len() {
            let input_frames = resampler.input_frames_next();
            if pos + input_frames > input_channel.len() {
                break;
            }
            let chunk = vec![input_channel[pos..pos + input_frames].to_vec()];
            let output = resampler
                .process(&chunk, None)
                .map_err(|e| format!("Resampling failed: {:?}", e))?;
            resampled_data.extend_from_slice(&output[0]);
            pos += input_frames;
        }
        Ok(resampled_data.into_iter().map(|s| s as f32).collect())
    }

    fn compute_mel_frames(&self, samples: &[f32], sample_rate: u32) -> Vec<Array2<f64>> {
        let fft_size = 512;
        let hop_size = 160;
        let n_mels = 80;
        let mut spect = Spectrogram::new(fft_size, hop_size);
        let mut mel_spec = MelSpectrogram::new(fft_size, sample_rate as f64, n_mels);
        let mut mel_frames = Vec::new();
        let mut pos = 0;
        while pos < samples.len() {
            let end = cmp::min(pos + hop_size, samples.len());
            if let Some(fft_result) = spect.add(&samples[pos..end]) {
                let fft_result_conv: Array1<_> = Array1::from_shape_vec(
                    fft_result.raw_dim(),
                    fft_result.iter().cloned().collect(),
                )
                .expect("Conversion failed");
                let mel_frame = mel_spec.add(&fft_result_conv);
                mel_frames.push(mel_frame);
            }
            pos += hop_size;
        }
        mel_frames
    }

    fn save_vad_image(
        &self,
        mel_frames: &[Array2<f64>],
        output_path: &str,
        settings: &DetectionSettings,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let edge_info = vad_boundaries(mel_frames, settings);
        let img = as_image(
            mel_frames,
            &edge_info.non_intersected(),
            &edge_info.gradient_positions(),
        );
        img.save(output_path)?;
        Ok(())
    }

    fn compute_segments(
        &self,
        decisions: &[bool],
        contiguous_threshold: usize,
        hop_size: usize,
        total_samples: usize,
    ) -> Vec<(usize, usize)> {
        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut speech_start: Option<usize> = None;
        for i in 0..decisions.len() {
            if !in_speech && decisions[i] {
                if i + contiguous_threshold <= decisions.len()
                    && decisions[i..i + contiguous_threshold].iter().all(|&d| d)
                {
                    speech_start = Some(i);
                    in_speech = true;
                }
            } else if in_speech && !decisions[i] {
                if i + contiguous_threshold <= decisions.len()
                    && decisions[i..i + contiguous_threshold].iter().all(|&d| !d)
                {
                    if let Some(start) = speech_start {
                        segments.push((start * hop_size, i * hop_size));
                    }
                    speech_start = None;
                    in_speech = false;
                }
            }
        }
        if in_speech {
            if let Some(start) = speech_start {
                segments.push((start * hop_size, total_samples));
            }
        }
        segments
    }

    fn run_segmentation_vad(
        &self,
        resampled_samples: &[f32],
        sample_rate: u32,
    ) -> Result<(Vec<Vec<Vec<u8>>>, Vec<Vec<i16>>, usize), Box<dyn std::error::Error>> {
        let hop_size = 160;
        let mel_frames = self.compute_mel_frames(resampled_samples, sample_rate);
        let default_settings = DetectionSettings::new(1.0, 10, 10, 0);
        let re = Regex::new(r"[^\w]+").unwrap();
        let basename = "vad_output";
        let sanitized = re
            .replace_all(basename, "_")
            .into_owned()
            .trim_matches('_')
            .to_string();
        let vad_image_path = format!("vad_{}.png", sanitized);
        self.save_vad_image(&mel_frames, &vad_image_path, &default_settings)?;
        let mut vad = VoiceActivityDetector::new(&default_settings);
        let mut decisions = Vec::new();
        for frame in &mel_frames {
            decisions.push(vad.add(&frame.to_owned()).unwrap_or(false));
        }
        let total_samples = resampled_samples.len();
        let min_length = self.output_sample_rate as usize;
        let mut best_candidate = None;
        let mut best_diff = usize::MAX;
        let mut chosen_threshold = 0;
        for candidate in (5..=30).rev() {
            let mut segments =
                self.compute_segments(&decisions, candidate, hop_size, total_samples);
            if !segments.is_empty() && self.has_intro {
                segments.remove(0);
            }
            segments.retain(|&(start, end)| (end - start) >= min_length);
            let seg_count = segments.len();
            println!(
                "Candidate threshold {}: produced {} segments",
                candidate, seg_count
            );
            if seg_count == self.lines_in_file {
                best_candidate = Some(segments);
                chosen_threshold = candidate;
                println!("Exact match found with threshold {}", candidate);
                break;
            } else {
                let diff = if seg_count > self.lines_in_file {
                    seg_count - self.lines_in_file
                } else {
                    self.lines_in_file - seg_count
                };
                if diff < best_diff {
                    best_diff = diff;
                    best_candidate = Some(segments);
                    chosen_threshold = candidate;
                }
            }
        }
        let final_segments = best_candidate.ok_or("No segments found")?;
        println!("Chosen contiguous threshold: {}", chosen_threshold);

        let pre_vad_padding_samples =
            (self.pre_vad_padding_ms as usize * self.output_sample_rate as usize) / 1000;
        let adjusted_segments: Vec<(usize, usize)> = final_segments
            .into_iter()
            .map(|(start, end)| {
                let new_start = if start > pre_vad_padding_samples {
                    start - pre_vad_padding_samples
                } else {
                    0
                };
                (new_start, end)
            })
            .collect();

        let opus_frame_size = 320; // 20ms at 16kHz
        let encoder_sample_rate = self.output_sample_rate;
        let encoder_bits_per_sample = 16;
        let bitrate = 96_000;
        let mut file_audio_clips = Vec::new();
        let mut file_pcm_clips = Vec::new();
        for (start, end) in adjusted_segments {
            let clip_samples = &resampled_samples[start..end];
            let clip_bytes = unsafe {
                std::slice::from_raw_parts(
                    clip_samples.as_ptr() as *const u8,
                    clip_samples.len() * std::mem::size_of::<f32>(),
                )
            };
            let i16_clip_samples = f32le_to_i16(clip_bytes);
            file_pcm_clips.push(i16_clip_samples.clone());
            let mut encoder = OpusEncoder::new(
                encoder_sample_rate,
                encoder_bits_per_sample,
                1,
                opus_frame_size,
                bitrate,
            );
            encoder
                .init()
                .map_err(|e| format!("Opus encoder initialization failed: {:?}", e))?;
            let chunk_size = (opus_frame_size * 1) as usize;
            let mut opus_frames = Vec::new();
            for chunk in i16_clip_samples.chunks(chunk_size) {
                let mut frame = chunk.to_vec();
                if frame.len() < chunk_size {
                    frame.resize(chunk_size, 0);
                }
                let mut output_buffer = vec![0u8; frame.len() * 2];
                let encoded_len = encoder
                    .encode_i16(&frame, &mut output_buffer)
                    .map_err(|e| format!("Opus encoding failed: {:?}", e))?;
                output_buffer.truncate(encoded_len);
                opus_frames.push(output_buffer);
            }
            file_audio_clips.push(opus_frames);
        }
        let clip_count = file_audio_clips.len();
        Ok((file_audio_clips, file_pcm_clips, clip_count))
    }

    /// Process each WAV file: preprocess, compute mel frames, save a VAD image,
    /// run segmentation with iterative contiguous thresholds until the desired number of segments is reached,
    /// encode segments to Opus, and write the PCM/Opus files using sentence names from the Harvard file.
    pub fn process(
        &self,
        text_file: &str,
        src_dir: &str,
        output_dir: &str,
    ) -> Result<Vec<LineAudio>, Box<dyn std::error::Error>> {
        let sentences = Self::read_harvard_file_lines(text_file)?;
        let mut file_paths: Vec<PathBuf> = fs::read_dir(src_dir)?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("wav") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        file_paths.sort();
        let re = Regex::new(r"[^\w]+").unwrap();
        let mut line_audio_entries = Vec::new();
        let mut sentence_index = 0;
        for path in file_paths.iter() {
            let resampled_samples = self.preprocess_file(path)?;
            let mel_frames = self.compute_mel_frames(&resampled_samples, self.output_sample_rate);
            let basename = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let sanitized = re.replace_all(basename, "_").into_owned();
            let vad_image_path = format!("{}/vad_{}.png", output_dir, sanitized);
            self.save_vad_image(
                &mel_frames,
                &vad_image_path,
                &DetectionSettings::new(1.0, 10, 10, 0),
            )?;
            let (file_audio_clips, file_pcm_clips, clip_count) =
                self.run_segmentation_vad(&resampled_samples, self.output_sample_rate)?;
            println!(
                "VAD segmentation for file {:?} produced {} clips",
                path, clip_count
            );
            for (pcm, opus_frames) in file_pcm_clips.into_iter().zip(file_audio_clips.into_iter()) {
                let sentence = if sentence_index < sentences.len() {
                    sentences[sentence_index].clone()
                } else {
                    format!("clip_{}", sentence_index)
                };

                // Compute corpus and line number.
                let corpus_num = (sentence_index / 10) + 1;
                let line_num = (sentence_index % 10) + 1;
                let prefix = format!("H{:02}_{:02}_", corpus_num, line_num);

                let sanitized_sentence = re
                    .replace_all(&sentence, "_")
                    .into_owned()
                    .trim_matches('_')
                    .to_string();
                let filename = format!("{}{}", prefix, sanitized_sentence);
                let pcm_file_path = format!("{}/{}.pcm", output_dir, filename);
                let pcm_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        pcm.as_ptr() as *const u8,
                        pcm.len() * std::mem::size_of::<i16>(),
                    )
                };
                fs::write(&pcm_file_path, pcm_bytes)?;
                println!("Wrote PCM data to {}", pcm_file_path);
                let opus_file_path = format!("{}/{}.opus", output_dir, filename);
                write_opus_file_with_headers(
                    &opus_file_path,
                    &opus_frames,
                    self.output_sample_rate,
                    1,
                    16,
                )?;
                println!("Wrote Opus data to {}", opus_file_path);
                line_audio_entries.push(LineAudio {
                    line_name: sentence,
                    opus_frames,
                });
                sentence_index += 1;
            }
        }
        Ok(line_audio_entries)
    }
}

/// Writes opus frames to a file using a FrameHeader to delimit each frame.
/// The file begins with a 4-byte big-endian frame count; then, for each frame,
/// the header (serialized via FrameHeader::encode) is written followed by the frame bytes.
pub fn write_opus_file_with_headers(
    path: &str,
    frames: &[Vec<u8>],
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    let frame_count = frames.len() as u32;
    file.write_all(&frame_count.to_be_bytes())?;
    for frame in frames {
        let header = FrameHeader::new(
            EncodingFlag::Opus,
            frame.len() as u16, // Use frame length as sample_size
            sample_rate,
            channels,
            bits_per_sample,
            Endianness::LittleEndian,
            None,
            None,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        header.encode(&mut file)?;
        file.write_all(frame)?;
    }
    Ok(())
}

/// Reads a file produced by `write_opus_file_with_headers` and returns a vector of opus frames.
pub fn read_opus_file_with_headers(path: &str) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut count_buf = [0u8; 4];
    file.read_exact(&mut count_buf)?;
    let frame_count = u32::from_be_bytes(count_buf);
    let mut frames = Vec::with_capacity(frame_count as usize);
    for _ in 0..frame_count {
        let header = FrameHeader::decode(&mut file)?;
        let frame_size = header.sample_size() as usize;
        let mut frame = vec![0u8; frame_size];
        file.read_exact(&mut frame)?;
        frames.push(frame);
    }
    Ok(frames)
}

/// Reconstructs LineAudio entries from opus files in the given directory.
pub fn reconstruct_lines(opus_dir: &str) -> Result<Vec<LineAudio>, Box<dyn std::error::Error>> {
    let mut lines = Vec::new();
    for entry in fs::read_dir(opus_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("opus") {
            let file_stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let frames = read_opus_file_with_headers(path.to_str().unwrap())?;
            lines.push(LineAudio {
                line_name: file_stem,
                opus_frames: frames,
            });
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate_output() -> Result<(), Box<dyn std::error::Error>> {
        let lines_in_file = 10;
        // Create the processor with a 500ms pre-VAD padding.
        let processor = AudioProcessor::new(false, lines_in_file, 16000, 500);
        let text_file = "Harvard.txt";
        let src_dir = "output2";
        let output_dir = "testdata/uk";
        let line_audios = processor.process(text_file, src_dir, output_dir)?;
        println!("Generated {} LineAudio entries.", line_audios.len());
        for la in line_audios.iter() {
            println!(
                "Line: {}, Opus frames: {}",
                la.line_name,
                la.opus_frames.len()
            );
        }
        let reconstructed = reconstruct_lines(output_dir)?;
        println!(
            "Reconstructed {} LineAudio entries from cached opus files.",
            reconstructed.len()
        );
        Ok(())
    }
}
