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
  ClickAtCoordinatesTool,
  delay
} from "./tools";
import { LMStudioChatModel } from "./models/lm-studio-chat-model";

// Define the schema for the agent's action responses using Zod
const ActionSchema = z.object({
  description: z.string().describe("A detailed description of what you observe on the page and what action you're taking"),
  action: z.enum([
    "navigateTo",
    "clickElement",
    "clickAtCoordinates",
    "typeText",
    "pressKey",
    "waitForElement", 
    "getElementText",
    "GOAL_ACHIEVED"
  ]).describe("The action to perform"),
  parameters: z.object({
    url: z.string().optional().describe("URL to navigate to (for navigateTo action)"),
    selector: z.string().optional().describe("CSS selector to target an element (for clickElement, typeText, waitForElement, getElementText)"),
    text: z.string().optional().describe("Text to type (for typeText action)"),
    key: z.string().optional().describe("Key to press (for pressKey action)"),
    x: z.number().optional().describe("X coordinate for coordinate-based clicking (for clickAtCoordinates action)"),
    y: z.number().optional().describe("Y coordinate for coordinate-based clicking (for clickAtCoordinates action)")
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
      "enum": ["navigateTo", "clickElement", "clickAtCoordinates", "typeText", "pressKey", "waitForElement", "getElementText", "GOAL_ACHIEVED"],
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
        },
        "x": {
          "type": "number",
          "description": "X coordinate for coordinate-based clicking (for clickAtCoordinates action)"
        },
        "y": {
          "type": "number",
          "description": "Y coordinate for coordinate-based clicking (for clickAtCoordinates action)"
        }
      },
      "description": "Parameters for the selected action"
    }
  },
  "required": ["description", "action", "parameters"]
}
\`\`\`
`;

/**
 * Extracts interactive elements from HTML and returns them in a structured format.
 * This helps the LLM identify elements to interact with using selectors.
 */
function getInteractiveElementList(html: string): string {
  try {
    // Extract the title for informational purposes
    const titleMatch = /<title>(.*?)<\/title>/i.exec(html);
    const title = titleMatch ? titleMatch[1] : 'Unknown Page';
    
    let result = `# Page: ${title}\n\n`;
    result += `## Interactive Elements List\n\n`;
    
    // Extract buttons
    result += `### Buttons\n`;
    
    // Regular button elements
    const buttonRegex = /<button[^>]*>(.*?)<\/button>/gi;
    let buttonMatch;
    let buttonFound = false;
    
    while ((buttonMatch = buttonRegex.exec(html)) !== null) {
      buttonFound = true;
      const buttonTag = buttonMatch[0];
      const buttonText = buttonMatch[1].replace(/<[^>]*>/g, '').trim();
      
      const idMatch = /id=["']([^"']*)["']/i.exec(buttonTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(buttonTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(buttonTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(buttonTag);
      
      result += `- Button: "${buttonText || '[No text]'}"\n`;
      result += `  - CSS: button`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: button[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: button[name="${nameMatch[1]}"]`;
      }
      
      if (buttonText) {
        result += `\n  - Text Selector: text="${buttonText}"`;
      }
      
      result += '\n\n';
    }
    
    // Input elements (buttons)
    const inputButtonRegex = /<input[^>]*type=["'](submit|button)["'][^>]*>/gi;
    while ((buttonMatch = inputButtonRegex.exec(html)) !== null) {
      buttonFound = true;
      const buttonTag = buttonMatch[0];
      
      const valueMatch = /value=["']([^"']*)["']/i.exec(buttonTag);
      const buttonText = valueMatch ? valueMatch[1] : '[No text]';
      
      const idMatch = /id=["']([^"']*)["']/i.exec(buttonTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(buttonTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(buttonTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(buttonTag);
      const typeMatch = /type=["']([^"']*)["']/i.exec(buttonTag);
      const type = typeMatch ? typeMatch[1] : 'submit';
      
      result += `- Input ${type} button: "${buttonText}"\n`;
      result += `  - CSS: input[type="${type}"]`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: input[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: input[name="${nameMatch[1]}"]`;
      }
      
      if (valueMatch) {
        result += `\n  - Value Selector: input[value="${buttonText}"]`;
        result += `\n  - Text Selector: text="${buttonText}"`;
      }
      
      result += '\n\n';
    }
    
    if (!buttonFound) {
      result += `No button elements found.\n\n`;
    }
    
    // Extract input fields
    result += `### Input Fields\n`;
    
    const inputRegex = /<input[^>]*>/gi;
    let inputMatch;
    let inputFound = false;
    
    while ((inputMatch = inputRegex.exec(html)) !== null) {
      const inputTag = inputMatch[0];
      
      // Skip buttons and hidden inputs
      const typeMatch = /type=["']([^"']*)["']/i.exec(inputTag);
      const inputType = typeMatch ? typeMatch[1].toLowerCase() : 'text';
      
      if (inputType === 'button' || inputType === 'submit' || inputType === 'hidden') {
        continue;
      }
      
      inputFound = true;
      
      const idMatch = /id=["']([^"']*)["']/i.exec(inputTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(inputTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(inputTag);
      const placeholderMatch = /placeholder=["']([^"']*)["']/i.exec(inputTag);
      const valueMatch = /value=["']([^"']*)["']/i.exec(inputTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(inputTag);
      
      // Look for a label that might be associated with this input
      let labelText = '';
      if (idMatch) {
        const labelForRegex = new RegExp(`<label[^>]*for=["']${idMatch[1]}["'][^>]*>(.*?)<\/label>`, 'i');
        const labelMatch = labelForRegex.exec(html);
        if (labelMatch) {
          labelText = labelMatch[1].replace(/<[^>]*>/g, '').trim();
        }
      }
      
      const description = labelText || placeholderMatch?.[1] || nameMatch?.[1] || idMatch?.[1] || inputType;
      
      result += `- Input (${inputType}): "${description}"\n`;
      result += `  - CSS: input[type="${inputType}"]`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: input[id="${idMatch[1]}"]`;
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: input[name="${nameMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (placeholderMatch) {
        result += `\n  - Placeholder Selector: input[placeholder="${placeholderMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      result += '\n\n';
    }
    
    if (!inputFound) {
      result += `No input elements found.\n\n`;
    }
    
    // Extract links
    result += `### Links\n`;
    
    const linkRegex = /<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/gi;
    let linkMatch;
    let linkFound = false;
    
    while ((linkMatch = linkRegex.exec(html)) !== null) {
      linkFound = true;
      const linkTag = linkMatch[0];
      const href = linkMatch[1];
      const linkText = linkMatch[2].replace(/<[^>]*>/g, '').trim();
      
      if (!linkText || href.startsWith('javascript:') || href === '#') {
        continue; // Skip empty links or JS links
      }
      
      const idMatch = /id=["']([^"']*)["']/i.exec(linkTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(linkTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(linkTag);
      
      result += `- Link: "${linkText}" (${href})\n`;
      result += `  - CSS: a`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: a[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (linkText) {
        result += `\n  - Text Selector: text="${linkText}"`;
      }
      
      if (href && !href.startsWith('#')) {
        result += `\n  - Href Selector: a[href="${href}"]`;
      }
      
      result += '\n\n';
    }
    
    if (!linkFound) {
      result += `No link elements found.\n\n`;
    }
    
    // Extract select/dropdown elements
    result += `### Dropdowns\n`;
    
    const selectRegex = /<select[^>]*>([\s\S]*?)<\/select>/gi;
    let selectMatch;
    let selectFound = false;
    
    while ((selectMatch = selectRegex.exec(html)) !== null) {
      selectFound = true;
      const selectTag = selectMatch[0];
      const selectContent = selectMatch[1];
      
      const idMatch = /id=["']([^"']*)["']/i.exec(selectTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(selectTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(selectTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(selectTag);
      
      // Look for label for this select
      let labelText = '';
      if (idMatch) {
        const labelForRegex = new RegExp(`<label[^>]*for=["']${idMatch[1]}["'][^>]*>(.*?)<\/label>`, 'i');
        const labelMatch = labelForRegex.exec(html);
        if (labelMatch) {
          labelText = labelMatch[1].replace(/<[^>]*>/g, '').trim();
        }
      }
      
      const description = labelText || nameMatch?.[1] || idMatch?.[1] || 'Dropdown';
      
      result += `- Select dropdown: "${description}"\n`;
      result += `  - CSS: select`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: select[id="${idMatch[1]}"]`;
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: select[name="${nameMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
        }
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      // Extract options
      const optionRegex = /<option[^>]*value=["']([^"']*)["'][^>]*>(.*?)<\/option>/gi;
      let optionMatch;
      let optionsText = '\n  - Options:';
      let optionsFound = false;
      
      while ((optionMatch = optionRegex.exec(selectContent)) !== null) {
        optionsFound = true;
        const value = optionMatch[1];
        const text = optionMatch[2].replace(/<[^>]*>/g, '').trim();
        optionsText += `\n    - "${text}" (value: ${value})`;
      }
      
      if (optionsFound) {
        result += optionsText;
      }
      
      result += '\n\n';
    }
    
    if (!selectFound) {
      result += `No select dropdown elements found.\n\n`;
    }
    
    // Extract clickable/interactive div elements
    result += `### Clickable Divs/Spans\n`;
    
    const clickableDivRegex = /<(div|span)[^>]*(onclick|role=["'](button|link|tab|menuitem)["']|class=["'][^"']*\b(btn|button|clickable)\b[^"']*["'])[^>]*>(.*?)<\/\1>/gi;
    let divMatch;
    let divFound = false;
    
    while ((divMatch = clickableDivRegex.exec(html)) !== null) {
      divFound = true;
      const divTag = divMatch[0];
      const tagName = divMatch[1]; // div or span
      const divText = divMatch[5].replace(/<[^>]*>/g, '').trim();
      
      const idMatch = /id=["']([^"']*)["']/i.exec(divTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(divTag);
      const roleMatch = /role=["']([^"']*)["']/i.exec(divTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(divTag);
      
      const role = roleMatch ? roleMatch[1] : 'clickable';
      
      result += `- ${tagName.charAt(0).toUpperCase() + tagName.slice(1)} (${role}): "${divText || '[No text]'}"\n`;
      result += `  - CSS: ${tagName}`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: ${tagName}[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (roleMatch) {
        result += `\n  - Role Selector: ${tagName}[role="${roleMatch[1]}"]`;
      }
      
      if (divText) {
        result += `\n  - Text Selector: text="${divText}"`;
      }
      
      result += '\n\n';
    }
    
    if (!divFound) {
      result += `No clickable div/span elements found.\n\n`;
    }
    
    return result;
  } catch (error) {
    console.error('Error extracting interactive elements:', error);
    return `Error extracting interactive elements: ${String(error)}`;
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
    
    //const modelName = "gemma-3-27b-it-qat";
    const modelName = "gemma-3-12b-it-qat";
    // Get the model with larger context length
    console.log(`Loading ${modelName} model...`);
    //const lmStudioModel = await client.llm.model(modelName);
    const lmStudioModel = await client.llm.model(modelName, {
      config: {
        contextLength: 20000,
        gpu: {
          ratio: 0.5,
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
      
      // Create prompt messages directly without template parsing
      const userMessage = {
        role: "user" as const,
        content: "Current webpage interactive elements:\n\n{html_source}\n\nBased on the list of interactive elements above, determine what action to take next to progress toward the goal."
      };

      const systemMessage = {
        role: "system" as const,
        content: "You are a web browsing assistant that helps users navigate and interact with websites. " +
          "You have access to a WebAgent API that provides functions to interact with web pages using Puppeteer.\n\n" +
          "IMPORTANT: Many websites show cookie consent popups, ads, or modal windows when you first visit them.\n" +
          "- When you see a cookie consent dialog, use the clickElement action with text=\"Accept\" or a similar selector\n" +
          "- To click a button with specific text, use selector format: text=\"Button Text\" (e.g., text=\"Accept all\")\n" +
          "- If a popup or modal blocks the main content, find a way to close it first\n\n" +
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
          "- CAREFULLY EXAMINE the provided list of interactive elements to find correct selectors\n" +
          "- For text-based selection, use: text=\"Text to find\" (e.g., text=\"Accept cookies\")\n" +
          "- For standard CSS selectors, look for these in the provided list:\n" +
          "  * ID selectors (e.g., #APjFqb)\n" +
          "  * Full selectors (e.g., input[id=\"APjFqb\"])\n" +
          "  * Name selectors (e.g., input[name=\"q\"])\n" +
          "  * Class selectors (e.g., .gLFyf)\n" +
          "- For search boxes specifically, look for inputs with type=\"search\" or name=\"q\"\n" +
          "- NEVER invent selectors - always use the exact selector from the provided list\n" +
          "- Use the most specific and reliable selector available\n\n" +
          "Google-specific guidance:\n" +
          "- Google's search input often has id=\"APjFqb\" or name=\"q\" or class containing \"gLFyf\"\n" +
          "- Search button often has type=\"submit\" or class containing \"gNO89b\"\n" +
          "- Search results are usually links with text describing the result\n\n" +
          "When you believe you have completed the goal, respond with \"GOAL_ACHIEVED\" as your action."
      };

      // Set up the chain
      const chain = RunnableSequence.from([
        {
          //html_source: () => extractRelevantHtml(pageSource)
          //html_source: () => pageSource
          html_source: () => getInteractiveElementList(pageSource)
        },
        async (input) => {
          // Send to LM Studio with the image
          return await lmStudioModel.respond([
            {
              ...systemMessage
            },
            {
              ...userMessage,
              content: userMessage.content.replace("{html_source}", input.html_source)
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