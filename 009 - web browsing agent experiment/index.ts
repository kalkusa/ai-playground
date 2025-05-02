import { LMStudioClient } from "@lmstudio/sdk";
import WebAgent from "./web-agent";
import path from "path";
import fs from "fs";
import { RunnableSequence } from "@langchain/core/runnables";
import { StructuredOutputParser } from "langchain/output_parsers";
import {
  NavigateToTool,
  ClickElementTool,
  TypeTextTool,
  PressKeyTool,
  WaitForElementTool,
  GetElementTextTool,
  ClickAtCoordinatesTool,
  delay
} from "./tools";
import { LMStudioChatModel } from "./models/lm-studio-chat-model";
import { ActionSchema, ActionType } from "./schemas";
import { createSystemMessage, userMessageTemplate } from "./prompts";
import { getInteractiveElementList, getSimplifiedHtml, trimHtmlContent } from "./html-parser";

// Maximum token count to aim for (leave margin for safety)
const MAX_TOKEN_COUNT = 8000;
// Assuming approximately 4 characters per token for rough estimation
const MAX_CHAR_LENGTH = MAX_TOKEN_COUNT * 4;

async function main() {
  const webAgent = new WebAgent();
  let currentStep = 1;
  let lmStudioModel: any = null;
  let modelCleanupAttempted = false;
  
  // Function to safely clean up resources
  const cleanupResources = async () => {
    if (modelCleanupAttempted) return;
    modelCleanupAttempted = true;
    
    console.log('Cleaning up resources...');
    
    // Clean up WebAgent resources
    try {
      await webAgent.cleanup();
      console.log('WebAgent cleaned up successfully');
    } catch (error) {
      console.error('Error cleaning up WebAgent:', error);
    }
    
    // Unload the model if it was loaded
    if (lmStudioModel) {
      try {
        console.log('Unloading language model...');
        await lmStudioModel.unload();
        console.log('Language model unloaded successfully');
      } catch (error) {
        console.error('Error unloading language model:', error);
        // Try a more aggressive termination if unload fails
        try {
          console.log('Attempting to force model termination...');
          // This is a last resort - only do this if regular unloading fails
          await lmStudioModel.terminate();
          console.log('Model terminated successfully');
        } catch (termError) {
          console.error('Failed to terminate model:', termError);
        }
      }
    }
  };
  
  try {
    // Initialize the WebAgent with visible browser
    await webAgent.initialize(false);
    
    // Connect to LM Studio
    console.log('Connecting to LM Studio...');
    const client = new LMStudioClient();
    
    //const modelName = "gemma-3-27b-it-qat";
    //const modelName = "gemma-3-12b-it-qat";
    //const modelName = "llama-4-scout-17b-16e-instruct";
    const modelName = "mistral-nemo-instruct-2407";

    // Get the model with larger context length
    console.log(`Loading ${modelName} model...`);
    //const lmStudioModel = await client.llm.model(modelName);
    lmStudioModel = await client.llm.model(modelName, {
      config: {
        contextLength: 10000,
        gpu: {
          ratio: 1.0,
        },
      },
    });
   
    // Create a LangChain model wrapper with the same increased context length
    const model = new LMStudioChatModel(modelName);
    await model.init();
    
    // Set up the Zod parser
    const parser = StructuredOutputParser.fromZodSchema(ActionSchema);
    
    // Take an initial screenshot of blank page
    const initialScreenshotPath = await webAgent.takeScreenshot('step_0_initial_state');
    
    // Get initial page source (should be about:blank)
    let pageSource = await webAgent.getPageSource();
    
    // Save initial HTML source
    fs.writeFileSync(path.join('./screenshots', 'step_0_initial_state_source.html'), pageSource);
    
    // Prepare the image file
    console.log('Preparing initial image...');
    let currentImage = await client.files.prepareImage(path.resolve(initialScreenshotPath));
    
    // Variables to track state across iterations
    let screenshotPath = initialScreenshotPath;
    let actionResult = "Starting web agent...";
    let actionHistory = "";
    
    // Main interaction loop
    const MAX_STEPS = 50;
    while (currentStep <= MAX_STEPS) {
      console.log(`Step ${currentStep}...`);
      
      // Create tools for the current step
      const tools = [
        new NavigateToTool(webAgent, currentStep),
        new ClickElementTool(webAgent, currentStep),
        new ClickAtCoordinatesTool(webAgent, currentStep),
        new TypeTextTool(webAgent, currentStep),
        new PressKeyTool(webAgent, currentStep),
        new WaitForElementTool(webAgent, currentStep),
        new GetElementTextTool(webAgent, currentStep)
      ];
      
      // Use the createSystemMessage function to generate the system message
      const systemMessage = createSystemMessage(currentStep, MAX_STEPS, actionHistory, actionResult);
      
      // Set up the chain
      const chain = RunnableSequence.from([
        {
          //html_source: () => getInteractiveElementList(pageSource)
          html_source: () => {
            // First simplify the HTML to remove unwanted elements
            const simplified = getSimplifiedHtml(pageSource, true, true);
            
            // Check if we're still in danger of exceeding context limits
            // if (simplified.length > MAX_CHAR_LENGTH) {
            //   console.log(`HTML content is still large (${simplified.length} chars), trimming to fit token limit...`);
            //   // Use our new function to trim content to fit in context
            //   return trimHtmlContent(simplified, MAX_CHAR_LENGTH);
            // }
            
            return simplified;
          }
        },
        async (input) => {
          // Send to LM Studio
          const content = userMessageTemplate.content.replace("{html_source}", input.html_source)
          console.log(`Sending prompt to LM Studio (HTML size: ${input.html_source.length} chars)...`);
          return await lmStudioModel.respond([
            {
              ...systemMessage
            },
            {
              ...userMessageTemplate,
              content
            }
          ]);
        },
        async (response) => {
          try {
            const content = response.content;
            console.log("Model response:", content);
            
            // Extract just the JSON part of the response
            const jsonMatch = content.match(/(\{[\s\S]*\})/);
            const jsonContent = jsonMatch ? jsonMatch[0] : content;
            
            // Parse the response using the Zod schema
            const parsedResponse = await parser.parse(jsonContent);
            console.log("Parsed response:", JSON.stringify(parsedResponse, null, 2));
            
            // Save the description to a file
            fs.writeFileSync(
              path.join('./screenshots', `step_${currentStep}_description.txt`), 
              parsedResponse.description
            );
            
            return parsedResponse;
          } catch (error) {
            console.error("Error parsing model response:", error);
            // Fall back to a default action if parsing fails
            if (currentStep === 1) {
              return {
                description: "Failed to parse response, defaulting to navigating to Google",
                action: "navigateTo" as const,
                parameters: { url: "https://www.google.com" }
              } satisfies ActionType;
            } else {
              // If we can't parse the response in later steps, assume we've reached the goal
              return {
                description: "Failed to parse response, assuming goal achieved",
                action: "GOAL_ACHIEVED" as const,
                parameters: {}
              } satisfies ActionType;
            }
          }
        }
      ]);
      
      // Execute the chain for this step
      console.log(`Executing chain for step ${currentStep}...`);
      const result = await chain.invoke({}) as ActionType;
      
      console.log(`Step ${currentStep} action: ${result.action}`);
      
      // Check if the goal has been achieved
      if (result.action === "GOAL_ACHIEVED") {
        console.log("Goal achieved! Task completed successfully.");
        await webAgent.takeScreenshot(`step_${currentStep}_goal_achieved`);
        break;
      }
      
      // Execute the action based on the parsed response
      try {
        // Execute the appropriate tool based on the action
        const tool = tools.find(t => t.name === result.action);
        if (!tool) {
          throw new Error(`Unknown action: ${result.action}`);
        }
        
        // Prepare the tool arguments based on the parameters
        let toolArgs = "";
        switch (result.action) {
          case "navigateTo":
            toolArgs = result.parameters.url || "";
            break;
          case "clickElement":
          case "waitForElement":
          case "getElementText":
            toolArgs = result.parameters.selector || "";
            break;
          case "clickAtCoordinates":
            toolArgs = JSON.stringify({
              x: result.parameters.x || 0,
              y: result.parameters.y || 0
            });
            break;
          case "typeText":
            toolArgs = JSON.stringify({
              selector: result.parameters.selector || "",
              text: result.parameters.text || ""
            });
            break;
          case "pressKey":
            toolArgs = result.parameters.key || "";
            break;
        }
        
        // Call the tool
        actionResult = await tool._call(toolArgs);
        console.log("Action result:", actionResult);
        
        // Take a screenshot of the current state
        screenshotPath = await webAgent.takeScreenshot(`step_${currentStep}_after_action`);
        
        // Get the page source
        pageSource = await webAgent.getPageSource();
        
        // Save the HTML source to a file
        const htmlFileName = `step_${currentStep}_after_action_source.html`;
        const htmlFilePath = path.join('./screenshots', htmlFileName);
        fs.writeFileSync(htmlFilePath, pageSource);
        
        // Update action history
        actionHistory += `Step ${currentStep}: ${result.action}`;
        if (result.parameters.url) {
          actionHistory += ` - URL: ${result.parameters.url}`;
        } else if (result.parameters.selector) {
          actionHistory += ` - Selector: ${result.parameters.selector}`;
          if (result.parameters.text) {
            actionHistory += `, Text: ${result.parameters.text}`;
          } else if (result.parameters.key) {
            actionHistory += ` - Key: ${result.parameters.key}`;
          }
          actionHistory += ` (${result.description.substring(0, 100)}${result.description.length > 100 ? '...' : ''})\n\n`;
        } else if (result.parameters.x !== undefined && result.parameters.y !== undefined) {
          actionHistory += ` - Coordinates: (${result.parameters.x}, ${result.parameters.y})`;
          actionHistory += ` (${result.description.substring(0, 100)}${result.description.length > 100 ? '...' : ''})\n\n`;
        }
      } catch (error) {
        console.error(`Error executing action:`, error);
        actionResult = `Error: ${error instanceof Error ? error.message : String(error)}`;
        
        // Take an error screenshot
        screenshotPath = await webAgent.takeScreenshot(`step_${currentStep}_error`);
        
        // Try to get the page source even after an error
        try {
          pageSource = await webAgent.getPageSource();
          const htmlFileName = `step_${currentStep}_error_source.html`;
          const htmlFilePath = path.join('./screenshots', htmlFileName);
          fs.writeFileSync(htmlFilePath, pageSource);
        } catch (sourceError) {
          console.error('Could not get page source after error:', sourceError);
        }
      }
      
      // Prepare for the next step
      console.log('Preparing next image...');
      currentImage = await client.files.prepareImage(path.resolve(screenshotPath));
      
      // Increment the step counter
      currentStep++;
    }
    
    if (currentStep > MAX_STEPS) {
      console.log("Maximum number of steps reached. Task did not complete.");
      // Take a final screenshot
      await webAgent.takeScreenshot(`step_${currentStep}_max_steps_reached`);
    }
    
  } catch (error) {
    console.error('Error in main function:', error);
  } finally {
    await cleanupResources();
  }
}

// Setup handlers for graceful shutdown
process.on('SIGINT', async () => {
  console.log('\nReceived SIGINT. Shutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\nReceived SIGTERM. Shutting down gracefully...');
  process.exit(0);
});

// Handle uncaught exceptions to ensure cleanup
process.on('uncaughtException', async (error) => {
  console.error('Uncaught exception:', error);
  process.exit(1);
});

main().catch(async (error) => {
  console.error('Unhandled error in main:', error);
  process.exit(1);
});