![Nano Banana CLI Header](header.png)

# nano-banana-cli

CLI for text and image generation with Google Gemini and OpenAI.

## Features

- Text generation using Gemini 2.0 Flash
- Image generation across multiple models and providers

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

Pick a model with `--model` (defaults to `nano-banana2`):

| Value | Model | Provider |
|---|---|---|
| `nano-banana2` | `gemini-3.1-flash-image-preview` | Google |
| `nano-banana1` | `gemini-2.0-flash-exp-image-generation` | Google |
| `nano-banana-pro` | `gemini-3-pro-image` | Google |
| `gpt-image-2` | `gpt-image-2` | OpenAI |

GPT Image 2 is the strongest option for images containing rendered text. It
also accepts `--quality` (`low`, `medium`, `high`; default `medium`) and
`--aspect-ratio` (`square`, `portrait`, `landscape`; default `square`). Both
flags are ignored by the Google models.

```bash
nano-banana-cli image "a poster reading GRAND OPENING" \
  --model gpt-image-2 --quality high --aspect-ratio portrait -o poster.png
```

## Configuration

Each provider's key resolves independently, so you only need a key for the
model you actually use.

Google (`text`, and all `nano-banana*` models):
1. `--api-key` flag
2. `GOOGLE_AI_STUDIO_API_KEY` environment variable
3. `google-ai-studio` secret via `mull secrets`

OpenAI (`--model gpt-image-2`):
1. `--openai-api-key` flag
2. `OPENAI_API_KEY` environment variable
3. `openai` secret via `mull secrets`

## License

MIT
