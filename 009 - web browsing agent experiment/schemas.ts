import { z } from "zod";

// Define the schema for the agent's action responses using Zod
export const ActionSchema = z.object({
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
export type ActionType = z.infer<typeof ActionSchema>; 