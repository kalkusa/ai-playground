# Langchain Ollama Example

This project demonstrates using Langchain with a local LLM served by Ollama.

## Setup

1.  **Install Dependencies:**
    ```bash
    npm install
    ```

2.  **Install and Run Ollama:**
    - Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    - Ensure the Ollama application/server is running in the background.

3.  **Pull the Required Ollama Model:**
    - Open your terminal (Command Prompt, PowerShell, etc.) and run the following command to download the default model used by this example:
      ```bash
      ollama pull llama3:instruct
      ```
    - If you wish to use a different model, pull it using `ollama pull <model_name>` and update the `model` parameter in `src/index.ts`.

## Running the Example

```bash
npm start
```

This will run the `src/index.ts` script, which initializes `ChatOllama`, connects to your running Ollama instance, and sends a sample prompt to the specified model. 