import path from "path";
import fs from "fs";
import WebAgentTool from "./base-tool";
import WebAgent from "../web-agent";
import { delay } from "./utils";

/**
 * Tool for clicking elements
 */
export class ClickElementTool extends WebAgentTool {
  constructor(webAgent: WebAgent, stepNumber: number) {
    super(
      "clickElement",
      "Clicks on an element matching the provided CSS selector. Input should be a valid CSS selector.",
      webAgent,
      stepNumber
    );
  }
  
  async _call(selector: string): Promise<string> {
    console.log(`Executing clickElement: ${selector}`);
    try {
      await this.webAgent.clickElement(selector);
      
      // Add a pause after the action
      console.log(`Pausing for 500ms...`);
      await delay(500);
      
      // Take a screenshot after the action
      const screenshotName = `step_${this.stepNumber}_click_${selector.replace(/[^a-z0-9]/gi, '_').substring(0, 30)}`;
      const screenshotPath = await this.webAgent.takeScreenshot(screenshotName);
      
      // Get the page source
      const pageSource = await this.webAgent.getPageSource();
      
      // Save the HTML source to a file
      const htmlFileName = `${screenshotName}_source.html`;
      const htmlFilePath = path.join('./screenshots', htmlFileName);
      fs.writeFileSync(htmlFilePath, pageSource);
      
      return `Successfully clicked element ${selector}. Screenshot saved to ${screenshotPath}`;
    } catch (error) {
      console.error(`Error clicking element ${selector}:`, error);
      return `Error clicking element ${selector}: ${error}`;
    }
  }
}

/**
 * Tool for typing text
 */
export class TypeTextTool extends WebAgentTool {
  constructor(webAgent: WebAgent, stepNumber: number) {
    super(
      "typeText",
      "Types text into an input field matching the provided CSS selector. Input should be a JSON object with selector and text properties.",
      webAgent,
      stepNumber
    );
  }
  
  async _call(args: string): Promise<string> {
    const { selector, text } = JSON.parse(args);
    console.log(`Executing typeText: ${selector}, ${text}`);
    try {
      await this.webAgent.typeText(selector, text);
      
      // Add a pause after the action
      console.log(`Pausing for 500ms...`);
      await delay(500);
      
      // Take a screenshot after the action
      const screenshotName = `step_${this.stepNumber}_type_${text.replace(/[^a-z0-9]/gi, '_').substring(0, 30)}`;
      const screenshotPath = await this.webAgent.takeScreenshot(screenshotName);
      
      // Get the page source
      const pageSource = await this.webAgent.getPageSource();
      
      // Save the HTML source to a file
      const htmlFileName = `${screenshotName}_source.html`;
      const htmlFilePath = path.join('./screenshots', htmlFileName);
      fs.writeFileSync(htmlFilePath, pageSource);
      
      return `Successfully typed "${text}" into ${selector}. Screenshot saved to ${screenshotPath}`;
    } catch (error) {
      console.error(`Error typing text into ${selector}:`, error);
      return `Error typing text into ${selector}: ${error}`;
    }
  }
}

/**
 * Tool for pressing keys
 */
export class PressKeyTool extends WebAgentTool {
  constructor(webAgent: WebAgent, stepNumber: number) {
    super(
      "pressKey",
      "Presses a specific key on the keyboard. Input should be a key string (e.g., 'Enter', 'Tab').",
      webAgent,
      stepNumber
    );
  }
  
  async _call(key: string): Promise<string> {
    console.log(`Executing pressKey: ${key}`);
    try {
      await this.webAgent.pressKey(key as any);
      
      // Add a pause after the action
      console.log(`Pausing for 500ms...`);
      await delay(500);
      
      // Take a screenshot after the action
      const screenshotName = `step_${this.stepNumber}_press_${key}`;
      const screenshotPath = await this.webAgent.takeScreenshot(screenshotName);
      
      // Get the page source
      const pageSource = await this.webAgent.getPageSource();
      
      // Save the HTML source to a file
      const htmlFileName = `${screenshotName}_source.html`;
      const htmlFilePath = path.join('./screenshots', htmlFileName);
      fs.writeFileSync(htmlFilePath, pageSource);
      
      return `Successfully pressed key ${key}. Screenshot saved to ${screenshotPath}`;
    } catch (error) {
      console.error(`Error pressing key ${key}:`, error);
      return `Error pressing key ${key}: ${error}`;
    }
  }
}

/**
 * Tool for getting element text
 */
export class GetElementTextTool extends WebAgentTool {
  constructor(webAgent: WebAgent, stepNumber: number) {
    super(
      "getElementText",
      "Gets the text content from an element matching the provided selector. Input should be a valid CSS selector.",
      webAgent,
      stepNumber
    );
  }
  
  async _call(selector: string): Promise<string> {
    console.log(`Executing getElementText: ${selector}`);
    try {
      const text = await this.webAgent.getElementText(selector);
      
      // Add a pause after the action
      console.log(`Pausing for 500ms...`);
      await delay(500);
      
      // Take a screenshot after the action
      const screenshotName = `step_${this.stepNumber}_get_text_${selector.replace(/[^a-z0-9]/gi, '_').substring(0, 30)}`;
      const screenshotPath = await this.webAgent.takeScreenshot(screenshotName);
      
      // Get the page source
      const pageSource = await this.webAgent.getPageSource();
      
      // Save the HTML source to a file
      const htmlFileName = `${screenshotName}_source.html`;
      const htmlFilePath = path.join('./screenshots', htmlFileName);
      fs.writeFileSync(htmlFilePath, pageSource);
      
      return `Text content from ${selector}: "${text}". Screenshot saved to ${screenshotPath}`;
    } catch (error) {
      console.error(`Error getting text from element ${selector}:`, error);
      return `Error getting text from element ${selector}: ${error}`;
    }
  }
} 