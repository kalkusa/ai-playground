// Hardcoded format instructions to avoid template parsing issues
export const formatInstructionsString = `The output should be formatted as a JSON instance that conforms to the JSON schema below.

IMPORTANT: Your response MUST ONLY contain the JSON object. Do not include any explanatory text before or after the JSON.

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
    content: `You are a web browsing assistant that helps users navigate websites by analyzing HTML and executing actions.

RESPOND ONLY WITH VALID JSON using the format in these instructions. No explanatory text outside JSON.

${currentStep === 1 
  ? "GOAL: Navigate to google.com, type \"AI\" in the search box, search, and click the first result."
  : `PROGRESS:
- Goal: Navigate to google.com, type "AI" in the search box, search, and click the first result.
- Step: ${currentStep}/${maxSteps}
- Previous actions and results:
${actionHistory}
- Last result: ${actionResult}`}

REFLECTION PROCESS:
1. Analyze what's currently on the page based on the HTML
2. Reflect on previous actions and their outcomes
3. Identify what progress you've made toward the goal
4. Determine what specific sub-goal you need to accomplish next
5. Find the appropriate element to interact with
6. Choose the correct action and parameters

Analyze the HTML source to find appropriate elements, then execute ONE of these actions:
1. navigateTo: Visit a URL
2. clickElement: Click an element matching a selector
3. typeText: Enter text in an input field
4. pressKey: Press a keyboard key (Enter, Tab, etc.)
5. waitForElement: Wait for an element to appear
6. getElementText: Read text from an element
7. clickAtCoordinates: Click at x,y coordinates

${formatInstructionsString}

IMPORTANT: Your "description" field must include:
- What you observe on the page
- Your reasoning about previous actions (what worked, what didn't)
- What specific sub-goal you're trying to accomplish
- Why you chose this particular action and selector

SELECTOR TIPS:
- Use text="Accept" to click buttons by their text
- For cookies/popups, first check for and dismiss them
- Choose from selectors in the HTML source like:
  • ID (#elementId)
  • Name (input[name="q"])
  • Class (.className)
- Never invent selectors - use what's in the HTML
- For search boxes, look for inputs with type="search" or name="q"

When the goal is achieved, respond with action: "GOAL_ACHIEVED"`
  };
}

// Standard user message template
export const userMessageTemplate = {
  role: "user" as const,
  content: "Current webpage source code:\n\n{html_source}\n\nBased on the list of interactive elements above, determine what action to take next to progress toward the goal."
}; 