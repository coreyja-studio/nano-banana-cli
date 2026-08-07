use base64::{Engine, prelude::BASE64_STANDARD};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

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
    /// GPT Image 2 - OpenAI's image model (best for text rendered in images)
    // heck's kebab-case does not insert a hyphen before a trailing digit
    // (`GptImage2` would become `gpt-image2`), so the name is spelled out.
    #[value(name = "gpt-image-2")]
    GptImage2,
}

impl ImageModel {
    fn api_name(&self) -> &'static str {
        match self {
            ImageModel::NanoBanana2 => "gemini-3.1-flash-image-preview",
            ImageModel::NanoBanana1 => "gemini-2.0-flash-exp-image-generation",
            ImageModel::NanoBananaPro => "gemini-3-pro-image",
            ImageModel::GptImage2 => "gpt-image-2",
        }
    }
}

/// Image quality tier. Consumed by OpenAI models only; Google models ignore it.
#[derive(clap::ValueEnum, Clone, Debug, Default, PartialEq, Eq)]
enum Quality {
    Low,
    #[default]
    Medium,
    High,
}

impl Quality {
    fn api_value(&self) -> &'static str {
        match self {
            Quality::Low => "low",
            Quality::Medium => "medium",
            Quality::High => "high",
        }
    }
}

/// Output aspect ratio. Consumed by OpenAI models only; Google models ignore it.
#[derive(clap::ValueEnum, Clone, Debug, Default, PartialEq, Eq)]
enum AspectRatio {
    #[default]
    Square,
    Portrait,
    Landscape,
}

impl AspectRatio {
    fn size_string(&self) -> &'static str {
        match self {
            AspectRatio::Square => "1024x1024",
            AspectRatio::Portrait => "1024x1536",
            AspectRatio::Landscape => "1536x1024",
        }
    }
}

/// Secret name in mull/1Password for Google AI Studio credentials
const MULL_SECRET_NAME: &str = "google-ai-studio";

/// Secret name in mull/1Password for OpenAI credentials
const OPENAI_MULL_SECRET_NAME: &str = "openai";

#[derive(Parser)]
#[command(name = "nano-banana-cli")]
#[command(about = "CLI for Google Gemini and OpenAI text and image generation")]
struct Cli {
    /// Google API key (defaults to GOOGLE_AI_STUDIO_API_KEY env var, then mull secrets)
    #[arg(long, env = "GOOGLE_AI_STUDIO_API_KEY")]
    api_key: Option<String>,

    /// OpenAI API key (defaults to OPENAI_API_KEY env var, then mull secrets)
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

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
    /// Generate an image
    Image {
        /// The prompt describing the image to generate
        prompt: String,

        /// Output file path (defaults to output.png)
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Image model to use (default: nano-banana2)
        #[arg(long, default_value = "nano-banana2")]
        model: ImageModel,

        /// Image quality tier - OpenAI models only (default: medium)
        #[arg(long, default_value = "medium")]
        quality: Quality,

        /// Aspect ratio - OpenAI models only (default: square)
        #[arg(long, default_value = "square")]
        aspect_ratio: AspectRatio,
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

#[derive(Serialize)]
struct OpenAIImageRequest {
    model: String,
    prompt: String,
    size: String,
    quality: String,
}

/// OpenAI's images/generations response. `created` and `usage` are also
/// returned by the API but aren't needed here, so serde ignores them.
#[derive(Deserialize)]
struct OpenAIImageResponse {
    data: Vec<OpenAIImageData>,
}

#[derive(Deserialize)]
struct OpenAIImageData {
    b64_json: String,
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
/// provider requires implementing this trait and wiring it into the match in
/// `generate_image` — no existing provider code changes.
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

/// OpenAI GPT Image 2 provider.
///
/// Quality and aspect ratio live here rather than on `build_request` because
/// the trait signature is shared with providers that have no such options.
struct OpenAIImageProvider {
    quality: Quality,
    aspect_ratio: AspectRatio,
}

impl ImageProvider for OpenAIImageProvider {
    fn build_request(&self, api_key: &str, model_api_name: &str, prompt: &str) -> ProviderRequest {
        let request = OpenAIImageRequest {
            model: model_api_name.to_string(),
            prompt: prompt.to_string(),
            size: self.aspect_ratio.size_string().to_string(),
            quality: self.quality.api_value().to_string(),
        };

        ProviderRequest {
            url: "https://api.openai.com/v1/images/generations".to_string(),
            headers: vec![("Authorization".to_string(), format!("Bearer {}", api_key))],
            body: serde_json::to_value(&request)
                .expect("OpenAIImageRequest is always serializable"),
        }
    }

    fn parse_response(
        &self,
        body: serde_json::Value,
    ) -> Result<GeneratedImage, Box<dyn std::error::Error>> {
        let response: OpenAIImageResponse = serde_json::from_value(body)?;

        let image_data = response
            .data
            .first()
            .ok_or("OpenAI response contained no image data")?;

        // Trim before decoding — API responses can pick up stray whitespace.
        let data = BASE64_STANDARD.decode(image_data.b64_json.trim())?;

        Ok(GeneratedImage {
            // GPT Image 2 returns PNG unless another output format is requested,
            // and this CLI never requests one.
            mime_type: "image/png".to_string(),
            data,
        })
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

/// Fetch the OpenAI API key from mull secrets manager.
///
/// Expects a secret named `openai` containing the API key.
fn openai_api_key_from_mull() -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("mull")
        .args(["secrets", "get", OPENAI_MULL_SECRET_NAME])
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

/// Resolve the OpenAI API key from CLI arg, env var, or mull secrets.
///
/// The mull failure is replaced with an actionable message: a raw
/// "could not read secret" from 1Password doesn't tell the user what to do.
fn resolve_openai_api_key(
    cli_api_key: Option<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    // CLI arg or env var already handled by clap
    if let Some(key) = cli_api_key {
        return Ok(key);
    }

    openai_api_key_from_mull().map_err(|_| {
        "No OpenAI API key found. Provide one via:\n\
         \x20 1. --openai-api-key flag\n\
         \x20 2. OPENAI_API_KEY environment variable\n\
         \x20 3. mull secrets (1Password secret named 'openai')"
            .into()
    })
}

fn main() -> ExitCode {
    // Returning Result from main would print the error via Debug, which escapes
    // newlines — the multi-line credential help would arrive as literal "\n".
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        // Text is Google-only, so its key can be resolved eagerly.
        Commands::Text { prompt } => {
            let api_key = resolve_api_key(cli.api_key)?;
            generate_text(&api_key, &prompt)?;
        }
        // Image defers both keys to generate_image: picking a Google model must
        // not require an OpenAI key to be configured, and vice versa.
        Commands::Image {
            prompt,
            output,
            model,
            quality,
            aspect_ratio,
        } => generate_image(
            cli.api_key,
            cli.openai_api_key,
            &prompt,
            &output,
            &model,
            &quality,
            &aspect_ratio,
        )?,
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
    google_api_key: Option<String>,
    openai_api_key: Option<String>,
    prompt: &str,
    output: &PathBuf,
    model: &ImageModel,
    quality: &Quality,
    aspect_ratio: &AspectRatio,
) -> Result<(), Box<dyn std::error::Error>> {
    // Credentials resolve inside the arm for the selected provider, so an
    // unused provider's missing key never blocks a run.
    let (api_key, provider): (String, Box<dyn ImageProvider>) = match model {
        ImageModel::GptImage2 => (
            resolve_openai_api_key(openai_api_key)?,
            Box::new(OpenAIImageProvider {
                quality: quality.clone(),
                aspect_ratio: aspect_ratio.clone(),
            }),
        ),
        // Quality and aspect ratio are intentionally ignored for Google models
        // rather than rejected, so the flags are safe to always pass.
        _ => (
            resolve_api_key(google_api_key)?,
            Box::new(GoogleImageProvider),
        ),
    };

    let req = provider.build_request(&api_key, model.api_name(), prompt);

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
    use clap::ValueEnum;

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
        assert_eq!(ImageModel::GptImage2.api_name(), "gpt-image-2");
    }

    #[test]
    fn test_gpt_image_2_clap_value_name() {
        // heck would kebab-case this to "gpt-image2"; the explicit
        // #[value(name = ...)] is what keeps the hyphen before the digit.
        let name = ImageModel::GptImage2
            .to_possible_value()
            .unwrap()
            .get_name()
            .to_string();
        assert_eq!(name, "gpt-image-2");
    }

    #[test]
    fn test_cli_gpt_image_2_parses() {
        let cli = Cli::try_parse_from([
            "nano-banana-cli",
            "image",
            "a prompt",
            "--model",
            "gpt-image-2",
        ]);
        assert!(
            cli.is_ok(),
            "--model gpt-image-2 should parse: {:?}",
            cli.err()
        );
    }

    #[test]
    fn test_quality_default_and_values() {
        assert_eq!(Quality::default(), Quality::Medium);
        assert_eq!(Quality::Low.api_value(), "low");
        assert_eq!(Quality::Medium.api_value(), "medium");
        assert_eq!(Quality::High.api_value(), "high");
    }

    #[test]
    fn test_aspect_ratio_default_and_sizes() {
        assert_eq!(AspectRatio::default(), AspectRatio::Square);
        assert_eq!(AspectRatio::Square.size_string(), "1024x1024");
        assert_eq!(AspectRatio::Portrait.size_string(), "1024x1536");
        assert_eq!(AspectRatio::Landscape.size_string(), "1536x1024");
    }

    #[test]
    fn test_openai_request_serialization() {
        let request = OpenAIImageRequest {
            model: "gpt-image-2".to_string(),
            prompt: "A cat holding a sign that says HELLO".to_string(),
            size: "1024x1024".to_string(),
            quality: "medium".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-image-2"));
        assert!(json.contains("A cat holding a sign that says HELLO"));
        assert!(json.contains("1024x1024"));
        assert!(json.contains("medium"));
    }

    #[test]
    fn test_openai_response_deserialization() {
        let json = r#"{"created": 1234567890, "data": [{"b64_json": "iVBORw0KGgo="}]}"#;

        let response: OpenAIImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json, "iVBORw0KGgo=");
    }

    #[test]
    fn test_openai_response_with_usage() {
        // The real API also returns `usage`; parsing must not choke on it.
        let json = r#"{
            "created": 1234567890,
            "data": [{"b64_json": "iVBORw0KGgo="}],
            "usage": {"total_tokens": 1234, "input_tokens": 10, "output_tokens": 1224}
        }"#;

        let response: OpenAIImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data[0].b64_json, "iVBORw0KGgo=");
    }

    #[test]
    fn test_openai_provider_build_request() {
        let provider = OpenAIImageProvider {
            quality: Quality::Medium,
            aspect_ratio: AspectRatio::Square,
        };
        let req = provider.build_request("test-key", "gpt-image-2", "A cat");

        assert_eq!(req.url, "https://api.openai.com/v1/images/generations");
        assert_eq!(req.body["model"].as_str(), Some("gpt-image-2"));
        assert_eq!(req.body["prompt"].as_str(), Some("A cat"));
        assert_eq!(req.body["size"].as_str(), Some("1024x1024"));
        assert_eq!(req.body["quality"].as_str(), Some("medium"));

        let auth = req
            .headers
            .iter()
            .find(|(k, _)| k == "Authorization")
            .expect("Authorization header missing from provider request");
        assert_eq!(auth.1, "Bearer test-key");

        // send_json sets Content-Type; a duplicate here would conflict.
        assert!(req.headers.iter().all(|(k, _)| k != "Content-Type"));
        assert!(!req.url.contains("test-key"));
    }

    #[test]
    fn test_openai_provider_options_reach_request_body() {
        let provider = OpenAIImageProvider {
            quality: Quality::High,
            aspect_ratio: AspectRatio::Landscape,
        };
        let req = provider.build_request("k", "gpt-image-2", "a dog");

        assert_eq!(req.body["quality"].as_str(), Some("high"));
        assert_eq!(req.body["size"].as_str(), Some("1536x1024"));
    }

    #[test]
    fn test_openai_provider_parse_response() {
        let provider = OpenAIImageProvider {
            quality: Quality::Medium,
            aspect_ratio: AspectRatio::Square,
        };
        let body = serde_json::json!({
            "created": 1234567890,
            "data": [{"b64_json": "iVBORw0KGgo="}]
        });

        let image = provider.parse_response(body).expect("should parse");
        assert_eq!(image.mime_type, "image/png");
        assert!(!image.data.is_empty());
    }

    #[test]
    fn test_openai_provider_parse_response_empty_data() {
        let provider = OpenAIImageProvider {
            quality: Quality::Medium,
            aspect_ratio: AspectRatio::Square,
        };
        let body = serde_json::json!({"created": 123, "data": []});

        // `let else` rather than unwrap_err: GeneratedImage holds raw image
        // bytes and deliberately doesn't derive Debug.
        let Err(err) = provider.parse_response(body) else {
            panic!("an empty data array must not parse as a successful image");
        };
        assert!(
            err.to_string().contains("no image data"),
            "empty data should produce a specific error: {err}"
        );
    }

    #[test]
    fn test_openai_key_missing_error() {
        // Only meaningful when no 'openai' secret is configured — otherwise
        // resolution legitimately succeeds.
        if openai_api_key_from_mull().is_ok() {
            eprintln!("Skipping test_openai_key_missing_error: OpenAI key is configured");
            return;
        }

        let msg = resolve_openai_api_key(None).unwrap_err().to_string();
        assert!(
            msg.contains("--openai-api-key"),
            "error must mention the --openai-api-key flag: {msg}"
        );
        assert!(
            msg.contains("OPENAI_API_KEY"),
            "error must mention the OPENAI_API_KEY env var: {msg}"
        );
        assert!(
            msg.contains("openai"),
            "error must mention the mull secret name: {msg}"
        );
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
