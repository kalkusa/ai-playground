import path from "path";
import fs from "fs";
import WebAgentTool from "./base-tool";
import WebAgent from "../web-agent";
import { delay } from "./utils";

/**
 * Tool for navigating to a URL
 */
export class NavigateToTool extends WebAgentTool {
  constructor(webAgent: WebAgent, stepNumber: number) {
    super(
      "navigateTo",
      "Navigates to the specified URL. Input should be a URL string.",
      webAgent,
      stepNumber
    );
  }
  
  async _call(url: string): Promise<string> {
    console.log(`Executing navigateTo: ${url}`);
    try {
      await this.webAgent.navigateTo(url);
      
      // Add a pause after the action
      console.log(`Pausing for 500ms...`);
      await delay(500);
      
      // Take a screenshot after the action
      const screenshotName = `step_${this.stepNumber}_navigate_to_${url.replace(/[^a-z0-9]/gi, '_').substring(0, 30)}`;
      const screenshotPath = await this.webAgent.takeScreenshot(screenshotName);
      
      // Get the page source
      const pageSource = await this.webAgent.getPageSource();
      
      // Save the HTML source to a file
      const htmlFileName = `${screenshotName}_source.html`;
      const htmlFilePath = path.join('./screenshots', htmlFileName);
      fs.writeFileSync(htmlFilePath, pageSource);
      
      return `Successfully navigated to ${url}. Screenshot saved to ${screenshotPath}`;
    } catch (error) {
      console.error(`Error navigating to ${url}:`, error);
      return `Error navigating to ${url}: ${error}`;
    }
  }
}

/**
 * Tool for waiting for element to appear
 */
export class WaitForElementTool extends WebAgentTool {
  constructor(webAgent: WebAgent, stepNumber: number) {
    super(
      "waitForElement",
      "Waits for an element matching the selector to appear on the page. Input should be a valid CSS selector.",
      webAgent,
      stepNumber
    );
  }
  
  async _call(selector: string): Promise<string> {
    console.log(`Executing waitForElement: ${selector}`);
    try {
      await this.webAgent.waitForElement(selector);
      
      // Add a pause after the action
      console.log(`Pausing for 500ms...`);
      await delay(500);
      
      // Take a screenshot after the action
      const screenshotName = `step_${this.stepNumber}_wait_for_${selector.replace(/[^a-z0-9]/gi, '_').substring(0, 30)}`;
      const screenshotPath = await this.webAgent.takeScreenshot(screenshotName);
      
      // Get the page source
      const pageSource = await this.webAgent.getPageSource();
      
      // Save the HTML source to a file
      const htmlFileName = `${screenshotName}_source.html`;
      const htmlFilePath = path.join('./screenshots', htmlFileName);
      fs.writeFileSync(htmlFilePath, pageSource);
      
      return `Successfully waited for element ${selector}. Screenshot saved to ${screenshotPath}`;
    } catch (error) {
      console.error(`Error waiting for element ${selector}:`, error);
      return `Error waiting for element ${selector}: ${error}`;
    }
  }
} 