# LLama Image Recognition

A TypeScript application that uses LM Studio with a vision-capable model for image recognition.

## Requirements

- [LM Studio](https://lmstudio.ai/) installed and running
- A Vision-Language Model (VLM) loaded in LM Studio
  - **Recommended model**: `qwen2-vl-2b-instruct`
  - You can download this model from LM Studio's Model Catalog under "Models" tab
  - Note: While Llama 4 Scout has vision capabilities, it may not be properly configured in LM Studio to support image input yet

## Setup

1. Clone this repository
2. Install dependencies:
```bash
npm install
```

## Usage

1. Make sure LM Studio is running with the `qwen2-vl-2b-instruct` model (or another VLM) loaded
2. Run the project:
```bash
npm start
```

The application will:
1. Take a screenshot of Facebook's homepage
2. Send the screenshot to the VLM model
3. Display the model's analysis of what it sees in the image

For development with auto-reload:
```bash
npm run dev
```

## Build

To compile TypeScript to JavaScript:
```bash
npm run build
```

The compiled output will be in the `dist` directory. 