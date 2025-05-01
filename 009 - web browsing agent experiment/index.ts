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
import { formatInstructionsString, createSystemMessage, userMessageTemplate } from "./prompts";
import { getInteractiveElementList } from "./html-parser";

async function main() {
  const webAgent = new WebAgent();
  let currentStep = 1;
  
  try {
    // Initialize the WebAgent with visible browser
    await webAgent.initialize(false);
    
    // Connect to LM Studio
    console.log('Connecting to LM Studio...');
    const client = new LMStudioClient();
    
    //const modelName = "gemma-3-27b-it-qat";
    //const modelName = "gemma-3-12b-it-qat";
    const modelName = "deepseek-r1-distill-qwen-7b";
    // Get the model with larger context length
    console.log(`Loading ${modelName} model...`);
    //const lmStudioModel = await client.llm.model(modelName);
    const lmStudioModel = await client.llm.model(modelName, {
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
    const MAX_STEPS = 10;
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
          html_source: () => getInteractiveElementList(pageSource)
        },
        async (input) => {
          // Send to LM Studio
          return await lmStudioModel.respond([
            {
              ...systemMessage
            },
            {
              ...userMessageTemplate,
              content: userMessageTemplate.content.replace("{html_source}", input.html_source)
            }
          ]);
        },
        async (response) => {
          try {
            const content = response.content;
            console.log("Model response:", content);
            
            // Parse the response using the Zod schema
            const parsedResponse = await parser.parse(content);
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
    // Always clean up WebAgent resources
    await webAgent.cleanup();
  }
}

main().catch(console.error);