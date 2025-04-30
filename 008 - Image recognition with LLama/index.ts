import { LMStudioClient } from "@lmstudio/sdk";

async function main() {
  const client = new LMStudioClient();

  const model = await client.llm.model("llama-4-scout-17b-16e-instruct");
  const result = await model.respond("What is capital of Poland?");

  console.info(result.content);
}

main().catch(console.error); 