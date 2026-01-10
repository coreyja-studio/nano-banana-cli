![Nano Banana CLI Header](header.png)

# nano-banana-cli

CLI for interacting with Google Gemini Nano Banana Pro for text and image generation.

## Features

- Text generation using Gemini 2.0 Flash
- Image generation using Gemini 2.0 Flash Exp (Image Generation)

## Installation

```bash
cargo install --path .
```

## Usage

### Generate Text

```bash
nano-banana-cli text "Your prompt here"
```

### Generate Images

```bash
nano-banana-cli image "Your image prompt here" --output image.png
```

## Configuration

The CLI reads the API key from:
1. `--api-key` flag
2. `GOOGLE_AI_STUDIO_API_KEY` environment variable
3. `google-ai-studio` secret via `mnemon secrets`

## License

MIT
