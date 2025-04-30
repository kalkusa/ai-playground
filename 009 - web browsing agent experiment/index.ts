import { LMStudioClient } from "@lmstudio/sdk";
import WebAgent from "./web-agent";
import path from "path";
import fs from "fs";

// Helper function to pause execution for a specified time
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Prompt describing how to use the WebAgent API
const webAgentPrompt = `
You are a web browsing assistant that helps users navigate and interact with websites.
You have access to a WebAgent API that provides functions to interact with web pages using Puppeteer.

For each step, you will receive:
1. A screenshot of the current webpage state
2. The HTML source code of the webpage

Your task is to navigate through websites by examining both the screenshot and HTML source code to determine what actions to take next.
The HTML source will help you identify the correct selectors to use for interacting with elements.

IMPORTANT: Many websites show cookie consent popups, ads, or modal windows when you first visit them.
- ALWAYS check for and handle these popups FIRST before attempting other actions
- For cookie consent banners, look for buttons with text like "Accept", "Accept all", "I agree", "Okay", etc.
- Common selectors for cookie buttons: "#accept-cookies", ".cookie-accept", "[aria-label='accept cookies']", "button:contains('Accept')"
- If you see a popup or modal that blocks the main content, find a way to close it first

CURRENT GOAL: Navigate to google.com, type "AI" in the search box, click the search button or press Enter to search, and then click on the first search result.

For each step, provide TWO parts in your response:
1. DESCRIPTION: Describe what you see on the current page and explain what you plan to do next to progress toward the goal.
2. ACTION: Provide the exact action command to execute.

Format your response like this:
DESCRIPTION: [Your description of what you see and what you plan to do]
ACTION: [action_command]

Here are the functions available in the WebAgent API:

1. navigateTo(url: string)
   - Navigates to the specified URL
   - Example: "navigateTo:https://www.google.com"

2. clickElement(selector: string)
   - Clicks on an element matching the provided CSS selector
   - Example: "clickElement:#search-button"
   - USE THE HTML SOURCE to find the correct selector

3. typeText(selector: string, text: string)
   - Types text into an input field matching the provided CSS selector
   - Example: "typeText:input[name='q'],AI"
   - USE THE HTML SOURCE to find the correct selector

4. pressKey(key: string)
   - Presses a specific key on the keyboard (Enter, Tab, ArrowDown, etc.)
   - Example: "pressKey:Enter"

5. waitForElement(selector: string)
   - Waits for an element matching the selector to appear on the page
   - Example: "waitForElement:.results-container"
   - USE THE HTML SOURCE to find the correct selector

6. getElementText(selector: string)
   - Gets the text content from an element matching the provided selector
   - Example: "getElementText:h1"
   - USE THE HTML SOURCE to find the correct selector

7. handleCookieConsent()
   - Automatically attempts to detect and accept cookie consent dialogs
   - Example: "handleCookieConsent:"
   - This will try various common selectors for cookie consent buttons
   - ALWAYS try this when first loading a new page, especially Google

DO NOT make up selectors. ANALYZE the HTML source code to find exact selectors that exist in the document.
Look for id, name, class attributes, and HTML structure to determine the correct selectors.

COOKIE POPUP EXAMPLES:
- If you see "Weiter ohne Zustimmen" (Continue without agreeing) in German, use clickElement with that button's selector
- If there are "Accept All" or "Accept Cookies" buttons, click those first
- Look for buttons with class names containing "consent", "cookie", "accept", "agree", etc.
- Or simply use the handleCookieConsent: action which will try common selectors automatically

This is CRITICAL: When you are on the Google homepage, before looking for the search box, FIRST handle any cookie consent dialogs by using handleCookieConsent: or clicking the specific consent button.

When you believe you have completed the goal, respond with "GOAL_ACHIEVED" as your action.
`;

// Function to parse the model's response to extract description and action
function parseModelResponse(response: string): { description: string, action: string } {
  const descriptionMatch = response.match(/DESCRIPTION:\s*([\s\S]*?)(?=ACTION:|$)/i);
  const actionMatch = response.match(/ACTION:\s*(.*?)(?=$|\n)/i);
  
  const description = descriptionMatch ? descriptionMatch[1].trim() : "";
  const action = actionMatch ? actionMatch[1].trim() : "";
  
  return { description, action };
}

async function executeAction(webAgent: WebAgent, action: string, stepNumber: number): Promise<{screenshotPath: string, pageSource: string}> {
  if (!action || typeof action !== 'string') {
    throw new Error("Invalid action provided");
  }

  console.log(`Executing action: ${action}`);

  if (action === "GOAL_ACHIEVED") {
    console.log("Goal has been achieved!");
    return { screenshotPath: "DONE", pageSource: "" };
  }

  // Parse the action string
  const [command, ...params] = action.split(":");
  const parameters = params.join(":").split(",");
  const actionName = command.trim().toLowerCase();

  try {
    let screenshotName = `step_${stepNumber}_${actionName}`;
    
    switch (actionName) {
      case "navigateto":
        await webAgent.navigateTo(parameters[0]);
        break;
      
      case "clickelement":
        await webAgent.clickElement(parameters[0]);
        break;
      
      case "typetext":
        await webAgent.typeText(parameters[0], parameters.slice(1).join(","));
        break;
      
      case "presskey":
        await webAgent.pressKey(parameters[0] as any);
        break;
      
      case "waitforelement":
        await webAgent.waitForElement(parameters[0]);
        break;
      
      case "getelementtext":
        const text = await webAgent.getElementText(parameters[0]);
        console.log(`Text content: ${text}`);
        break;
        
      case "handlecookieconsent":
        const handled = await webAgent.handleCookieConsent();
        screenshotName = `step_${stepNumber}_cookie_consent_${handled ? 'handled' : 'not_found'}`;
        break;
      
      default:
        console.warn(`Unknown command: ${command}`);
        break;
    }

    // Add a pause after each action to give time for the page to update
    console.log(`Pausing for 500ms...`);
    await delay(500);
    
    // Take a screenshot after the action
    const screenshotPath = await webAgent.takeScreenshot(screenshotName);
    
    // Get the page source after the action
    console.log('Getting page source...');
    const pageSource = await webAgent.getPageSource();
    
    // Save the HTML source to a file for reference
    const htmlFileName = `${screenshotName}_source.html`;
    const htmlFilePath = path.join('./screenshots', htmlFileName);
    fs.writeFileSync(htmlFilePath, pageSource);
    console.log(`Saved HTML source to: ${htmlFilePath}`);
    
    return { screenshotPath, pageSource };
  } catch (error) {
    console.error(`Error executing action: ${error}`);
    // Take a screenshot even if the action failed
    const errorScreenshotPath = await webAgent.takeScreenshot(`step_${stepNumber}_error_${actionName}`);
    
    // Try to get the page source even after an error
    let errorPageSource = "";
    try {
      errorPageSource = await webAgent.getPageSource();
      const htmlFileName = `step_${stepNumber}_error_${actionName}_source.html`;
      const htmlFilePath = path.join('./screenshots', htmlFileName);
      fs.writeFileSync(htmlFilePath, errorPageSource);
    } catch (sourceError) {
      console.error('Could not get page source after error:', sourceError);
    }
    
    return { screenshotPath: errorScreenshotPath, pageSource: errorPageSource };
  }
}

// Format HTML source for inclusion in prompts by truncating to a reasonable size
function formatHtmlSource(html: string, maxLength: number = 5000): string {
  if (html.length <= maxLength) return html;
  
  // Get first part of HTML (typically contains <head> and opening <body>)
  const firstPart = html.substring(0, maxLength / 2);
  
  // Get last part of HTML (typically contains main content and closing tags)
  const lastPart = html.substring(html.length - maxLength / 2);
  
  return `${firstPart}\n\n... [HTML truncated for brevity] ...\n\n${lastPart}`;
}

async function main() {
  const webAgent = new WebAgent();
  
  try {
    // Initialize the WebAgent with visible browser
    await webAgent.initialize(false);
    
    // Connect to LM Studio
    console.log('Connecting to LM Studio...');
    const client = new LMStudioClient();
    
    // Get the model
    console.log('Loading Gemma 3 model...');
    const model = await client.llm.model("gemma-3-27b-it-qat");
    
    // Take an initial screenshot of blank page
    const initialScreenshotPath = await webAgent.takeScreenshot('step_0_initial_state');
    
    // Get initial page source (should be about:blank)
    const initialPageSource = await webAgent.getPageSource();
    
    // Save initial HTML source
    fs.writeFileSync(path.join('./screenshots', 'step_0_initial_state_source.html'), initialPageSource);
    
    // Prepare the image file
    console.log('Preparing initial image...');
    const initialImage = await client.files.prepareImage(path.resolve(initialScreenshotPath));
    
    // Initial prompt to the model with initial screenshot and HTML
    console.log(`Step 1: Asking model for first action...`);
    
    let result = await model.respond([{
      role: "user",
      content: `${webAgentPrompt}\n\nCurrent webpage state (screenshot and HTML):\n\nHTML Source:\n\`\`\`html\n${formatHtmlSource(initialPageSource)}\n\`\`\`\n\nProvide a description of what you see and the action to take. Format your response as specified with DESCRIPTION and ACTION sections.`,
      images: [initialImage]
    }]);
    
    let modelResponse = result.content.trim();
    console.log("Model response:");
    console.log(modelResponse);
    
    // Parse model response to get description and action
    let { description, action } = parseModelResponse(modelResponse);
    
    // Display the model's description
    console.log("\nModel's description:");
    console.log(description);
    console.log("\nModel's action:");
    console.log(action);
    
    // Default to navigating to Google if the action is invalid
    if (!action || !action.startsWith("navigateTo:")) {
      action = "navigateTo:https://www.google.com";
      console.log("Using default action:", action);
    }
    
    let completedSteps = 1;
    const MAX_STEPS = 10; // Safety limit to prevent infinite loops
    
    // Main interaction loop
    while (completedSteps <= MAX_STEPS) {
      // Execute the action and get the path to the new screenshot and page source
      const { screenshotPath, pageSource } = await executeAction(webAgent, action, completedSteps);
      
      if (screenshotPath === "DONE") {
        console.log("Goal achieved! Task completed successfully.");
        await webAgent.takeScreenshot(`step_${completedSteps}_goal_achieved`);
        break;
      }
      
      // Prepare the new screenshot for the model
      const newImage = await client.files.prepareImage(path.resolve(screenshotPath));
      
      // Get next action from model
      console.log(`Step ${completedSteps + 1}: Asking model for next action based on screenshot and HTML...`);
      
      // Take a screenshot of the decision-making step
      await webAgent.takeScreenshot(`step_${completedSteps}_decision`);
      
      // Add a pause before asking the model for the next action
      console.log(`Pausing for 500ms before next step...`);
      await delay(500);

      // Create a prompt for the next action
      const formattedHtml = formatHtmlSource(pageSource);
      
      const nextPrompt = `Here's the screenshot and HTML after the action: "${action}".

HTML Source:
\`\`\`html
${formattedHtml}
\`\`\`

Based on BOTH the screenshot AND the HTML source code:
1. Describe what you see on the current page and what you plan to do next to progress toward the goal.
2. Provide the exact action command to execute.

Remember our goal: Navigate to google.com, type "AI" in the search box, click the search button or press Enter to search, and then click on the first search result.

Format your response like this:
DESCRIPTION: [Your description of what you see and what you plan to do]
ACTION: [action_command]

If you believe you have completed the goal, respond with "GOAL_ACHIEVED" as your action.`;
      
      result = await model.respond([{
        role: "user",
        content: nextPrompt,
        images: [newImage]
      }]);
      
      modelResponse = result.content.trim();
      console.log("Model response:");
      console.log(modelResponse);
      
      // Parse model response to get description and action
      ({ description, action } = parseModelResponse(modelResponse));
      
      // Display the model's description
      console.log("\nModel's description:");
      console.log(description);
      console.log("\nModel's action:");
      console.log(action);
      
      // Save the description to a file
      fs.writeFileSync(
        path.join('./screenshots', `step_${completedSteps}_description.txt`), 
        description
      );
      
      // If action is invalid, provide guidance
      if (!action) {
        console.log("No valid action found in model response. Asking for clarification...");
        
        result = await model.respond([{
          role: "user",
          content: `I couldn't determine the action from your previous response. Please provide a clear action in the format "ACTION: command" where command is one of: navigateTo, clickElement, typeText, pressKey, waitForElement, getElementText, or handleCookieConsent.`,
          images: [newImage]
        }]);
        
        modelResponse = result.content.trim();
        console.log("Model's clarified response:");
        console.log(modelResponse);
        
        // Parse model response again
        ({ description, action } = parseModelResponse(modelResponse));
        
        console.log("\nModel's clarified action:");
        console.log(action);
        
        // If still no valid action, use a default
        if (!action) {
          action = "GOAL_ACHIEVED"; // Assume the goal is achieved if no action is provided
          console.log("Still no valid action. Assuming goal is achieved.");
        }
      }
      
      completedSteps++;
    }
    
    if (completedSteps > MAX_STEPS) {
      console.log("Maximum number of steps reached. Task did not complete.");
      // Take a final screenshot
      await webAgent.takeScreenshot(`step_${completedSteps}_max_steps_reached`);
    }
    
  } catch (error) {
    console.error('Error in main function:', error);
  } finally {
    // Always clean up WebAgent resources
    await webAgent.cleanup();
  }
}

main().catch(console.error);

// Export for testing/importing
export { webAgentPrompt, main }; 