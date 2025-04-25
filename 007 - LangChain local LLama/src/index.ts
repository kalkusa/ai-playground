import { ChatOllama } from "@langchain/ollama";
import { z } from "zod";

const countrySchema = z.object({
    Capital: z.string().describe("The capital city of the country"),
    Language: z.string().describe("The official language of the country. Only one main language is allowed."),
});

type Country = z.infer<typeof countrySchema>;

async function main() {
  let structuredLlm;
  try {
    console.log("Attempting to initialize ChatOllama...");
    const llm = new ChatOllama({
      baseUrl: "http://localhost:11434",
      model: "llama3:instruct",
      temperature: 0.0
    });
    console.log("ChatOllama initialized successfully.");

    structuredLlm = llm.withStructuredOutput(countrySchema);
    console.log("Structured output chain created.");

  } catch (error) {
    console.error("Error during ChatOllama initialization or structuring:", error);
    process.exit(1);
  }

  const countryName = "Poland"; 
  console.log(`Fetching information for ${countryName}...`);
  const promptString = `Provide information about the country: ${countryName}. Output should conform to the required schema.`;

  try {
    const response: Country = await structuredLlm.invoke(promptString);
    
    console.log(`\n--- ${countryName} ---`);
    console.log(`Capital: ${response.Capital}`);
    console.log(`Language: ${response.Language}`);
    console.log('--------------------');
  } catch (error) {
    console.error("Error invoking structured output chain:", error);
  }
}

main().catch(console.error); 