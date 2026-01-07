use base64::{Engine, prelude::BASE64_STANDARD};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

const TEXT_MODEL: &str = "gemini-2.0-flash";
const IMAGE_MODEL: &str = "gemini-2.0-flash-exp-image-generation";

#[derive(Parser)]
#[command(name = "nano-banana-cli")]
#[command(about = "CLI for Google Gemini text and image generation")]
struct Cli {
    /// API key (defaults to GOOGLE_AI_STUDIO_API_KEY env var)
    #[arg(long, env = "GOOGLE_AI_STUDIO_API_KEY")]
    api_key: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text using Gemini
    Text {
        /// The prompt to send to the model
        prompt: String,
    },
    /// Generate an image using Nano Banana Pro
    Image {
        /// The prompt describing the image to generate
        prompt: String,

        /// Output file path (defaults to output.png)
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,
    },
}

#[derive(Serialize)]
struct TextRequest {
    contents: Vec<Content>,
}

#[derive(Serialize)]
struct ImageRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: ImageGenerationConfig,
}

#[derive(Serialize)]
struct ImageGenerationConfig {
    #[serde(rename = "responseModalities")]
    response_modalities: Vec<String>,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct Response {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    content: CandidateContent,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Vec<ResponsePart>,
}

#[derive(Deserialize)]
struct ResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "inlineData")]
    inline_data: Option<InlineData>,
}

#[derive(Deserialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Text { prompt } => generate_text(&cli.api_key, &prompt)?,
        Commands::Image { prompt, output } => generate_image(&cli.api_key, &prompt, &output)?,
    }

    Ok(())
}

fn generate_text(api_key: &str, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        TEXT_MODEL, api_key
    );

    let request = TextRequest {
        contents: vec![Content {
            parts: vec![Part {
                text: prompt.to_string(),
            }],
        }],
    };

    let response: Response = ureq::post(&url)
        .header("Content-Type", "application/json")
        .send_json(&request)?
        .body_mut()
        .read_json()?;

    if let Some(candidate) = response.candidates.first()
        && let Some(part) = candidate.content.parts.first()
        && let Some(text) = &part.text
    {
        println!("{}", text);
    }

    Ok(())
}

fn generate_image(
    api_key: &str,
    prompt: &str,
    output: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        IMAGE_MODEL, api_key
    );

    let request = ImageRequest {
        contents: vec![Content {
            parts: vec![Part {
                text: prompt.to_string(),
            }],
        }],
        generation_config: ImageGenerationConfig {
            response_modalities: vec!["TEXT".to_string(), "IMAGE".to_string()],
        },
    };

    let response: Response = ureq::post(&url)
        .header("Content-Type", "application/json")
        .send_json(&request)?
        .body_mut()
        .read_json()?;

    for candidate in &response.candidates {
        for part in &candidate.content.parts {
            if let Some(inline_data) = &part.inline_data {
                let image_data = BASE64_STANDARD.decode(&inline_data.data)?;
                fs::write(output, &image_data)?;
                println!("Image saved to: {}", output.display());
                println!("Mime type: {}", inline_data.mime_type);
                return Ok(());
            }
        }
    }

    Err("No image data in response".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_request_serialization() {
        let request = TextRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: "Hello".to_string(),
                }],
            }],
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Hello"));
        assert!(json.contains("contents"));
        assert!(json.contains("parts"));
        assert!(json.contains("text"));
    }

    #[test]
    fn test_image_request_serialization() {
        let request = ImageRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: "A cat".to_string(),
                }],
            }],
            generation_config: ImageGenerationConfig {
                response_modalities: vec!["TEXT".to_string(), "IMAGE".to_string()],
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("A cat"));
        assert!(json.contains("generationConfig"));
        assert!(json.contains("responseModalities"));
        assert!(json.contains("IMAGE"));
    }

    #[test]
    fn test_text_response_deserialization() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello back!"}]
                }
            }]
        }"#;

        let response: Response = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(
            response.candidates[0].content.parts[0].text,
            Some("Hello back!".to_string())
        );
    }

    #[test]
    fn test_image_response_deserialization() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "iVBORw0KGgo="
                        }
                    }]
                }
            }]
        }"#;

        let response: Response = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        let inline_data = response.candidates[0].content.parts[0]
            .inline_data
            .as_ref()
            .unwrap();
        assert_eq!(inline_data.mime_type, "image/png");
        assert_eq!(inline_data.data, "iVBORw0KGgo=");
    }
}
