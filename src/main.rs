use base64::{Engine, prelude::BASE64_STANDARD};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const TEXT_MODEL: &str = "gemini-2.0-flash";

#[derive(clap::ValueEnum, Clone, Debug, Default)]
enum ImageModel {
    /// Nano Banana 2 - gemini-3.1-flash-image-preview (default)
    #[default]
    NanoBanana2,
    /// Nano Banana 1 - gemini-2.0-flash-exp-image-generation (legacy)
    NanoBanana1,
    /// Nano Banana Pro - gemini-3-pro-image
    NanoBananaPro,
}

impl ImageModel {
    fn api_name(&self) -> &'static str {
        match self {
            ImageModel::NanoBanana2 => "gemini-3.1-flash-image-preview",
            ImageModel::NanoBanana1 => "gemini-2.0-flash-exp-image-generation",
            ImageModel::NanoBananaPro => "gemini-3-pro-image",
        }
    }

    /// The backend that knows how to build requests and parse responses
    /// for this model.
    fn provider(&self) -> Box<dyn ImageProvider> {
        // All current models are served by Google; adding a second provider
        // only requires a new arm here, not changes to GoogleImageProvider.
        Box::new(GoogleImageProvider)
    }
}

/// Secret name in mull/1Password for Google AI Studio credentials
const MULL_SECRET_NAME: &str = "google-ai-studio";

#[derive(Parser)]
#[command(name = "nano-banana-cli")]
#[command(about = "CLI for Google Gemini text and image generation")]
struct Cli {
    /// API key (defaults to GOOGLE_AI_STUDIO_API_KEY env var, then mull secrets)
    #[arg(long, env = "GOOGLE_AI_STUDIO_API_KEY")]
    api_key: Option<String>,

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
    /// Generate an image using Nano Banana
    Image {
        /// The prompt describing the image to generate
        prompt: String,

        /// Output file path (defaults to output.png)
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Image model to use (default: nano-banana2)
        #[arg(long, default_value = "nano-banana2")]
        model: ImageModel,
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

/// A provider-built request, ready to be sent over HTTP.
///
/// `headers` must not contain `Content-Type` — the generic HTTP layer uses
/// `ureq`'s `send_json`, which sets it.
struct ProviderRequest {
    url: String,
    headers: Vec<(String, String)>,
    body: serde_json::Value,
}

/// Decoded image data returned by a provider's response parser.
struct GeneratedImage {
    mime_type: String,
    data: Vec<u8>,
}

/// Abstraction over image generation backends.
///
/// Each provider knows how to build an HTTP request (URL, headers, body) and
/// parse the response into image data for its specific API shape. Adding a new
/// provider requires implementing this trait and wiring it into
/// `ImageModel::provider()` — no existing provider code changes.
trait ImageProvider {
    /// Build the full request for a text-to-image generation call.
    fn build_request(&self, api_key: &str, model_api_name: &str, prompt: &str) -> ProviderRequest;

    /// Parse the provider's JSON response into decoded image data.
    fn parse_response(
        &self,
        body: serde_json::Value,
    ) -> Result<GeneratedImage, Box<dyn std::error::Error>>;
}

/// Google Gemini image generation provider.
struct GoogleImageProvider;

impl ImageProvider for GoogleImageProvider {
    fn build_request(&self, api_key: &str, model_api_name: &str, prompt: &str) -> ProviderRequest {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model_api_name
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

        ProviderRequest {
            url,
            // The key goes in a header, never the URL — query parameters leak
            // credentials into shell history, proxies, and access logs.
            headers: vec![("x-goog-api-key".to_string(), api_key.to_string())],
            body: serde_json::to_value(&request).expect("ImageRequest is always serializable"),
        }
    }

    fn parse_response(
        &self,
        body: serde_json::Value,
    ) -> Result<GeneratedImage, Box<dyn std::error::Error>> {
        let response: Response = serde_json::from_value(body)?;

        for candidate in &response.candidates {
            for part in &candidate.content.parts {
                if let Some(inline_data) = &part.inline_data {
                    let image_data = BASE64_STANDARD.decode(&inline_data.data)?;
                    return Ok(GeneratedImage {
                        mime_type: inline_data.mime_type.clone(),
                        data: image_data,
                    });
                }
            }
        }

        Err("No image data in response".into())
    }
}

/// Fetch the API key from mull secrets manager.
///
/// Expects a secret named `google-ai-studio` containing the API key.
fn api_key_from_mull() -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("mull")
        .args(["secrets", "get", MULL_SECRET_NAME])
        .output()
        .map_err(|e| format!("Failed to run mull: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "mull secrets failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Resolve the API key from CLI arg, env var, or mull secrets (in that order).
fn resolve_api_key(cli_api_key: Option<String>) -> Result<String, Box<dyn std::error::Error>> {
    // CLI arg or env var already handled by clap
    if let Some(key) = cli_api_key {
        return Ok(key);
    }

    // Fall back to mull secrets
    api_key_from_mull()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let api_key = resolve_api_key(cli.api_key)?;

    match cli.command {
        Commands::Text { prompt } => generate_text(&api_key, &prompt)?,
        Commands::Image {
            prompt,
            output,
            model,
        } => generate_image(&api_key, &prompt, &output, &model)?,
    }

    Ok(())
}

fn generate_text(api_key: &str, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        TEXT_MODEL
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
        .header("x-goog-api-key", api_key)
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
    model: &ImageModel,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = model.provider();
    let req = provider.build_request(api_key, model.api_name(), prompt);

    let mut http = ureq::post(&req.url);
    for (key, value) in &req.headers {
        http = http.header(key, value);
    }

    let response_body: serde_json::Value = http.send_json(&req.body)?.body_mut().read_json()?;

    let image = provider.parse_response(response_body)?;

    fs::write(output, &image.data)?;
    println!("Image saved to: {}", output.display());
    println!("Mime type: {}", image.mime_type);

    Ok(())
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

    #[test]
    fn test_image_model_api_names() {
        assert_eq!(
            ImageModel::NanoBanana2.api_name(),
            "gemini-3.1-flash-image-preview"
        );
        assert_eq!(
            ImageModel::NanoBanana1.api_name(),
            "gemini-2.0-flash-exp-image-generation"
        );
        assert_eq!(ImageModel::NanoBananaPro.api_name(), "gemini-3-pro-image");
    }

    #[test]
    fn test_api_key_in_header_not_url() {
        let provider = GoogleImageProvider;
        let req = provider.build_request("test-secret-key-12345", "gemini-3-pro-image", "a cat");

        let auth = req
            .headers
            .iter()
            .find(|(k, _)| k == "x-goog-api-key")
            .expect("x-goog-api-key header missing from provider request");
        assert_eq!(auth.1, "test-secret-key-12345");

        // The key must never reach the URL — query params leak into logs.
        assert!(!req.url.contains("key="));
        assert!(!req.url.contains("test-secret-key-12345"));
    }

    #[test]
    fn test_provider_request_url_has_no_query_string() {
        let provider = GoogleImageProvider;
        let req = provider.build_request("key123", "gemini-3-pro-image", "a dog");
        assert!(
            !req.url.contains('?'),
            "URL must not contain query parameters"
        );
    }

    #[test]
    fn test_default_model_is_nano_banana_2() {
        assert!(matches!(ImageModel::default(), ImageModel::NanoBanana2));
    }

    #[test]
    fn test_cli_image_default_model_parses() {
        // The image subcommand must parse when --model is omitted. This catches
        // a regression where default_value doesn't match a valid clap ValueEnum
        // value (the old "nano-banana-2" vs the correct "nano-banana2").
        let cli = Cli::try_parse_from(["nano-banana-cli", "image", "a prompt"]);
        assert!(
            cli.is_ok(),
            "CLI with default model should parse: {:?}",
            cli.err()
        );
    }

    #[test]
    fn test_provider_request_body_is_valid_json() {
        let provider = GoogleImageProvider;
        let req = provider.build_request("key", "gemini-3-pro-image", "a cat");

        assert_eq!(
            req.body["contents"][0]["parts"][0]["text"].as_str(),
            Some("a cat")
        );
        assert!(req.body["generationConfig"]["responseModalities"].is_array());
    }

    #[test]
    fn test_provider_parse_response_extracts_image() {
        let provider = GoogleImageProvider;
        let body = serde_json::json!({
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
        });

        let image = provider.parse_response(body).expect("should parse");
        assert_eq!(image.mime_type, "image/png");
        assert!(!image.data.is_empty());
    }
}
