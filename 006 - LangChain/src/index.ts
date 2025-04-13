import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import * as dotenv from 'dotenv';
import { z } from "zod";

dotenv.config();

const countrySchema = z.object({
    Capital: z.string().describe("The capital city of the country"),
    Language: z.string().describe("The primary official language of the country"),
});

type Country = z.infer<typeof countrySchema>;

const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o",
    temperature: 0
});

const chain = model.withStructuredOutput(countrySchema);

async function getCountryInfo(countryName: string): Promise<Country> {
    console.log(`Fetching information for ${countryName}...`);

    const prompt = `Provide information about the country: ${countryName}.`;

    try {
        const response = await chain.invoke([new HumanMessage(prompt)]);
        return response;
    } catch (error) {
        console.error("Error fetching country information:", error);
        throw new Error("Failed to retrieve or parse country information.");
    }
}

async function main() {
    const countryName = "France"; 
    try {
        const countryInfo = await getCountryInfo(countryName);
        console.log(`\n--- ${countryName} ---`);
        console.log(`Capital: ${countryInfo.Capital}`);
        console.log(`Language: ${countryInfo.Language}`);
        console.log('--------------------');
    } catch (error) {
        console.error(error);
    }

    const countryName2 = "Japan";
    try {
      const countryInfo2 = await getCountryInfo(countryName2);
      console.log(`\n--- ${countryName2} ---`);
      console.log(`Capital: ${countryInfo2.Capital}`);
      console.log(`Language: ${countryInfo2.Language}`);
      console.log('--------------------');
  } catch (error) {
      console.error(error);
  }
}

main(); 