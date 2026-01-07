use clap::Parser;
use serde::{Deserialize, Serialize};

const MODEL: &str = "gemini-2.0-flash-lite-preview-02-05";

#[derive(Parser)]
#[command(name = "nano-banana-cli")]
#[command(about = "CLI for interacting with Google Gemini Nano Banana Pro")]
struct Cli {
    /// The prompt to send to the model
    prompt: String,

    /// API key (defaults to GOOGLE_AI_STUDIO_API_KEY env var)
    #[arg(long, env = "GOOGLE_AI_STUDIO_API_KEY")]
    api_key: String,
}

#[derive(Serialize)]
struct Request {
    contents: Vec<Content>,
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
    text: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        MODEL, cli.api_key
    );

    let request = Request {
        contents: vec![Content {
            parts: vec![Part { text: cli.prompt }],
        }],
    };

    let response: Response = ureq::post(&url)
        .header("Content-Type", "application/json")
        .send_json(&request)?
        .body_mut()
        .read_json()?;

    if let Some(candidate) = response.candidates.first()
        && let Some(part) = candidate.content.parts.first()
    {
        println!("{}", part.text);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let request = Request {
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
    fn test_response_deserialization() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello back!"}]
                }
            }]
        }"#;

        let response: Response = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(response.candidates[0].content.parts[0].text, "Hello back!");
    }
}
