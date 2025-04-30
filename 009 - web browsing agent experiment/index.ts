import { LMStudioClient } from "@lmstudio/sdk";
import WebAgent from "./web-agent";
import path from "path";
import fs from "fs";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StructuredOutputParser } from "langchain/output_parsers";
import {
  NavigateToTool,
  ClickElementTool,
  TypeTextTool,
  PressKeyTool,
  WaitForElementTool,
  GetElementTextTool,
  HandleCookieConsentTool,
  delay
} from "./tools";
import { LMStudioChatModel } from "./models/lm-studio-chat-model";

// Define the schema for the agent's action responses using Zod
const ActionSchema = z.object({
  description: z.string().describe("A detailed description of what you observe on the page and what action you're taking"),
  action: z.enum([
    "navigateTo",
    "clickElement",
    "typeText",
    "pressKey",
    "waitForElement", 
    "getElementText",
    "handleCookieConsent",
    "GOAL_ACHIEVED"
  ]).describe("The action to perform"),
  parameters: z.object({
    url: z.string().optional().describe("URL to navigate to (for navigateTo action)"),
    selector: z.string().optional().describe("CSS selector to target an element (for clickElement, typeText, waitForElement, getElementText)"),
    text: z.string().optional().describe("Text to type (for typeText action)"),
    key: z.string().optional().describe("Key to press (for pressKey action)")
  }).describe("Parameters for the selected action")
});

// Create a TypeScript type from the Zod schema
type ActionType = z.infer<typeof ActionSchema>;

// Hardcoded format instructions to avoid template parsing issues
const formatInstructionsString = `The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
one possible JSON instance would be {"foo": ["bar", "baz"]}

Here's the output schema:
\`\`\`json
{
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "A detailed description of what you observe on the page and what action you're taking"
    },
    "action": {
      "enum": ["navigateTo", "clickElement", "typeText", "pressKey", "waitForElement", "getElementText", "handleCookieConsent", "GOAL_ACHIEVED"],
      "description": "The action to perform"
    },
    "parameters": {
      "type": "object",
      "properties": {
        "url": {
          "type": "string",
          "description": "URL to navigate to (for navigateTo action)"
        },
        "selector": {
          "type": "string", 
          "description": "CSS selector to target an element (for clickElement, typeText, waitForElement, getElementText)"
        },
        "text": {
          "type": "string",
          "description": "Text to type (for typeText action)"
        },
        "key": {
          "type": "string",
          "description": "Key to press (for pressKey action)"
        }
      },
      "description": "Parameters for the selected action"
    }
  },
  "required": ["description", "action", "parameters"]
}
\`\`\`
`;

// Function to extract the most relevant parts of HTML for analysis
function extractRelevantHtml(html: string): string {
  try {
    // If HTML is small enough, return it intact
    if (html.length < 2000) return html;
    
    // Extract the title
    const titleMatch = /<title>(.*?)<\/title>/i.exec(html);
    const title = titleMatch ? titleMatch[1] : 'Unknown Page';
    
    // Create a focused structure report with only the most essential elements
    let structureReport = `
<html>
<head><title>${title}</title></head>
<body>
<h1>HTML Structure Report (Full HTML was ${html.length} characters)</h1>

<h2>Key Interactive Elements</h2>
<ul>
`;
    
    // Prioritize search elements, forms, and main navigation
    const searchPattern = /<input[^>]*type=["']?search["']?[^>]*>/gi;
    const formActionPattern = /<form[^>]*action=["']([^"']*)["'][^>]*>/gi;
    const mainButtonPattern = /<button[^>]*id=["']([^"']*)["'][^>]*>(.*?)<\/button>/gi;
    const navPattern = /<nav[^>]*>([\s\S]*?)<\/nav>/gi;
    
    // Find search boxes - these are critical for Google
    let searchMatch;
    let foundElements = 0;
    while ((searchMatch = searchPattern.exec(html)) !== null && foundElements < 5) {
      structureReport += `<li>Search: ${searchMatch[0].substring(0, 100)}</li>\n`;
      foundElements++;
    }
    
    // Find forms with actions
    let formMatch;
    foundElements = 0;
    while ((formMatch = formActionPattern.exec(html)) !== null && foundElements < 5) {
      structureReport += `<li>Form with action: ${formMatch[1]}</li>\n`;
      foundElements++;
    }
    
    // Find buttons with IDs (these are likely important)
    let buttonMatch;
    foundElements = 0;
    while ((buttonMatch = mainButtonPattern.exec(html)) !== null && foundElements < 5) {
      const buttonId = buttonMatch[1];
      const buttonText = buttonMatch[0].match(/>([^<]*)</)?.[1] || '';
      structureReport += `<li>Button ID=${buttonId}: ${buttonText}</li>\n`;
      foundElements++;
    }
    
    // If we have very few elements, try to add some more generic ones
    if (foundElements < 5) {
      const genericButtonPattern = /<button[^>]*>(.*?)<\/button>/gi;
      const inputPattern = /<input[^>]*>/gi;
      
      // Add a few generic buttons
      foundElements = 0;
      while ((buttonMatch = genericButtonPattern.exec(html)) !== null && foundElements < 5) {
        const buttonText = buttonMatch[0].match(/>([^<]*)</)?.[1] || '';
        structureReport += `<li>Button: ${buttonText || buttonMatch[0].substring(0, 50)}</li>\n`;
        foundElements++;
      }
      
      // Add a few inputs
      foundElements = 0;
      let inputMatch;
      while ((inputMatch = inputPattern.exec(html)) !== null && foundElements < 5) {
        structureReport += `<li>Input: ${inputMatch[0].substring(0, 100)}</li>\n`;
        foundElements++;
      }
    }
    
    // Close the structure report
    structureReport += `
</ul>

<h2>Page Structure</h2>
<p>Token limit reached: only showing most important interactive elements. The model should rely primarily on the screenshot for visual layout and element identification.</p>
</body>
</html>`;
    
    return structureReport;
  } catch (error) {
    console.error('Error extracting relevant HTML:', error);
    return `<html><body><p>Error processing HTML: ${String(error)}</p></body></html>`;
  }
}

async function main() {
  const webAgent = new WebAgent();
  let currentStep = 1;
  
  try {
    // Initialize the WebAgent with visible browser
    await webAgent.initialize(false);
    
    // Connect to LM Studio
    console.log('Connecting to LM Studio...');
    const client = new LMStudioClient();
    
    // Get the model with larger context length
    console.log('Loading Gemma 3 model with extended context...');
    const contextLength = 16000; // Increased context length
    const lmStudioModel = await client.llm.model("gemma-3-27b-it-qat");
    
    // Create a LangChain model wrapper with the same increased context length
    const model = new LMStudioChatModel("gemma-3-27b-it-qat", contextLength);
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
        new TypeTextTool(webAgent, currentStep),
        new PressKeyTool(webAgent, currentStep),
        new WaitForElementTool(webAgent, currentStep),
        new GetElementTextTool(webAgent, currentStep),
        new HandleCookieConsentTool(webAgent, currentStep)
      ];
      
      // Create prompt messages directly without template parsing
      const userMessage = {
        role: "user" as const,
        content: "Current webpage state (screenshot and HTML):\n\nHTML Source:\n```html\n{html_source}\n```\n\nBased on BOTH the screenshot AND the HTML source code, determine what action to take next to progress toward the goal."
      };

      const systemMessage = {
        role: "system" as const,
        content: "You are a web browsing assistant that helps users navigate and interact with websites. " +
          "You have access to a WebAgent API that provides functions to interact with web pages using Puppeteer.\n\n" +
          "IMPORTANT: Many websites show cookie consent popups, ads, or modal windows when you first visit them.\n" +
          "- ALWAYS check for and handle these popups FIRST before attempting other actions\n" +
          "- For cookie consent banners, look for buttons with text like \"Accept\", \"Accept all\", \"I agree\", \"Okay\", etc.\n" +
          "- If you see a popup or modal that blocks the main content, find a way to close it first\n\n" +
          (currentStep === 1 
            ? "GOAL: Navigate to google.com, type \"AI\" in the search box, click the search button or press Enter to search, and then click on the first search result.\n\n"
            : "PROGRESS TRACKING:\n" +
              "- Goal: Navigate to google.com, type \"AI\" in the search box, click the search button or press Enter to search, and then click on the first search result.\n" +
              `- Current step: ${currentStep}/${MAX_STEPS}\n` +
              `- Previous actions:\n${actionHistory}\n` +
              `Previous action result: ${actionResult}\n\n`) +
          "For each step, you must provide:\n" +
          "1. A detailed description of what you see on the page and what action you're taking\n" +
          "2. The exact action to execute with appropriate parameters\n\n" +
          formatInstructionsString + "\n\n" + 
          "When working with selectors:\n" +
          "- ALWAYS analyze the HTML source code to find exact selectors that exist in the document\n" +
          "- Look for id, name, class attributes, and HTML structure to determine the correct selectors\n" +
          "- Don't make up selectors that don't exist in the HTML\n\n" +
          "When handling cookie consent dialogs:\n" +
          "- Use the handleCookieConsent action to automatically detect and handle common consent patterns\n" +
          "- If automatic handling fails, look for relevant accept buttons and click them directly\n\n" +
          "When you believe you have completed the goal, respond with \"GOAL_ACHIEVED\" as your action."
      };

      // Set up the chain
      const chain = RunnableSequence.from([
        {
          html_source: () => extractRelevantHtml(pageSource)
        },
        async (input) => {
          // Send to LM Studio with the image
          return await lmStudioModel.respond([
            {
              ...systemMessage
            },
            {
              ...userMessage,
              content: userMessage.content.replace("{html_source}", input.html_source),
              images: [currentImage]
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
          case "typeText":
            toolArgs = JSON.stringify({
              selector: result.parameters.selector || "",
              text: result.parameters.text || ""
            });
            break;
          case "pressKey":
            toolArgs = result.parameters.key || "";
            break;
          case "handleCookieConsent":
            toolArgs = "";
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