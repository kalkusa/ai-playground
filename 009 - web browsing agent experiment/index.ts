import { LMStudioClient } from "@lmstudio/sdk";
import { getFacebookScreenshot } from "./screenshot";
import path from "path";

async function main() {
  try {
    // Take screenshot of Facebook
    const screenshotPath = await getFacebookScreenshot("./screenshot.png");
    
    // Connect to LM Studio and use Llama model
    console.log('Connecting to LM Studio...');
    const client = new LMStudioClient();
    
    // Get a model that supports vision (VLM - Vision-Language Model)
    console.log('Loading vision-capable model...');
    const model = await client.llm.model("qwen2-vl-2b-instruct");
    
    // Prepare the image file
    console.log('Preparing image...');
    const image = await client.files.prepareImage(path.resolve("./screenshot.png"));
    
    // Ask the model about the screenshot
    console.log('Asking model about the screenshot...');
    const result = await model.respond([
      { 
        role: "user", 
        content: "What do you see on this screenshot?", 
        images: [image] 
      }
    ]);
    
    console.info('Model response:');
    console.info(result.content);
  } catch (error) {
    console.error('Error in main function:', error);
  }
}

main().catch(console.error); 