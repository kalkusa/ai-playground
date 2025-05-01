// Hardcoded format instructions to avoid template parsing issues
export const formatInstructionsString = `The output should be formatted as a JSON instance that conforms to the JSON schema below.

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

// Function to create the system message with progress tracking
export function createSystemMessage(
  currentStep: number, 
  maxSteps: number, 
  actionHistory: string, 
  actionResult: string
): { role: "system", content: string } {
  return {
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
          `- Current step: ${currentStep}/${maxSteps}\n` +
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
}

// Standard user message template
export const userMessageTemplate = {
  role: "user" as const,
  content: "Current webpage interactive elements:\n\n{html_source}\n\nBased on the list of interactive elements above, determine what action to take next to progress toward the goal."
}; 