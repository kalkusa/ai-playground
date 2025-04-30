import puppeteer, { Browser, Page, KeyInput } from 'puppeteer';
import path from 'path';
import fs from 'fs';

class WebAgent {
  private browser: Browser | null = null;
  private page: Page | null = null;
  private screenshotsDir: string = './screenshots';

  /**
   * Initialize the web agent by launching a browser and creating a page
   */
  async initialize(headless: boolean = true): Promise<void> {
    try {
      // Create screenshots directory if it doesn't exist
      if (!fs.existsSync(this.screenshotsDir)) {
        fs.mkdirSync(this.screenshotsDir, { recursive: true });
      }

      // Launch the browser
      this.browser = await puppeteer.launch({
        headless,
        defaultViewport: { width: 1280, height: 800 }
      });

      // Create a new page
      this.page = await this.browser.newPage();
      
      // Set viewport size
      await this.page.setViewport({
        width: 1280,
        height: 800,
      });

      console.log('WebAgent initialized successfully');
    } catch (error) {
      console.error('Error initializing WebAgent:', error);
      throw error;
    }
  }

  /**
   * Attempts to detect and handle common cookie consent dialogs
   */
  async handleCookieConsent(): Promise<boolean> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log('Attempting to detect and handle cookie consent dialogs...');
      
      // List of common cookie accept button selectors
      const commonSelectors = [
        // English
        'button[id*="accept"]', 'button[class*="accept"]', 'button[id*="cookie"]', 'button[class*="cookie"]',
        'a[id*="accept"]', 'a[class*="accept"]', '#accept-cookies', '.accept-cookies',
        'button:has-text("Accept")', 'button:has-text("Accept all")', 'button:has-text("I agree")', 'button:has-text("OK")',
        // German
        'button:has-text("Akzeptieren")', 'button:has-text("Alle akzeptieren")', 'button:has-text("Zustimmen")',
        // French
        'button:has-text("Accepter")', 'button:has-text("J\'accepte")', 
        // Spanish
        'button:has-text("Aceptar")', 'button:has-text("Acepto")',
        // Generic
        '[aria-label*="cookie" i]', '[aria-label*="consent" i]', '[title*="cookie" i]', '[title*="consent" i]',
        // Google specific
        'button[id="L2AGLb"]', 'form[action="https://consent.google.com"] button', '.cookieBar button', '#consent-bump button'
      ];
      
      // Try each selector with a short timeout
      for (const selector of commonSelectors) {
        try {
          const exists = await this.page.waitForSelector(selector, { visible: true, timeout: 1000 });
          if (exists) {
            console.log(`Found cookie consent element with selector: ${selector}`);
            await this.page.click(selector);
            console.log('Clicked cookie consent button');
            await this.takeScreenshot('cookie_consent_handled');
            return true;
          }
        } catch (e) {
          // Selector not found, continue to next one
        }
      }
      
      console.log('No cookie consent dialog detected or could not handle it');
      return false;
    } catch (error) {
      console.error('Error handling cookie consent:', error);
      return false;
    }
  }

  /**
   * Navigate to a specific URL
   */
  async navigateTo(url: string): Promise<void> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log(`Navigating to ${url}...`);
      await this.page.goto(url, { waitUntil: 'networkidle2' });
      
      // Take a screenshot after navigation
      await this.takeScreenshot(`navigate_to_${this.sanitizeFilename(url)}`);
      
      // Attempt to handle cookie consent dialogs after navigation
      await this.handleCookieConsent();
    } catch (error) {
      console.error(`Error navigating to ${url}:`, error);
      throw error;
    }
  }

  /**
   * Click on an element matching the provided selector
   */
  async clickElement(selector: string): Promise<void> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log(`Clicking element with selector: ${selector}`);
      
      // Wait for the element to be visible with reduced timeout
      await this.page.waitForSelector(selector, { visible: true, timeout: 3000 });
      
      // Click the element
      await this.page.click(selector);
      
      // Take a screenshot after clicking
      await this.takeScreenshot(`click_${this.sanitizeFilename(selector)}`);
    } catch (error) {
      console.error(`Error clicking element with selector ${selector}:`, error);
      throw error;
    }
  }

  /**
   * Type text into an input field matching the provided selector
   */
  async typeText(selector: string, text: string): Promise<void> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log(`Typing "${text}" into element with selector: ${selector}`);
      
      // Wait for the element to be visible with reduced timeout
      await this.page.waitForSelector(selector, { visible: true, timeout: 3000 });
      
      // Clear the input field first
      await this.page.evaluate((sel) => {
        const element = document.querySelector(sel);
        if (element && 'value' in element) {
          (element as HTMLInputElement).value = '';
        }
      }, selector);
      
      // Type the text
      await this.page.type(selector, text);
      
      // Take a screenshot after typing
      await this.takeScreenshot(`type_${this.sanitizeFilename(text)}`);
    } catch (error) {
      console.error(`Error typing text into element with selector ${selector}:`, error);
      throw error;
    }
  }

  /**
   * Press a specific key on the keyboard
   */
  async pressKey(key: KeyInput): Promise<void> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log(`Pressing key: ${key}`);
      await this.page.keyboard.press(key);
      await this.takeScreenshot(`press_key_${key}`);
    } catch (error) {
      console.error(`Error pressing key ${key}:`, error);
      throw error;
    }
  }

  /**
   * Wait for navigation to complete
   */
  async waitForNavigation(): Promise<void> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log('Waiting for navigation to complete...');
      await this.page.waitForNavigation({ waitUntil: 'networkidle2' });
      await this.takeScreenshot('after_navigation');
    } catch (error) {
      console.error('Error waiting for navigation:', error);
      throw error;
    }
  }

  /**
   * Wait for a specific element to appear on the page
   */
  async waitForElement(selector: string, timeoutMs: number = 3000): Promise<void> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log(`Waiting for element with selector: ${selector}`);
      await this.page.waitForSelector(selector, { 
        visible: true,
        timeout: timeoutMs
      });
      await this.takeScreenshot(`wait_for_${this.sanitizeFilename(selector)}`);
    } catch (error) {
      console.error(`Error waiting for element with selector ${selector}:`, error);
      throw error;
    }
  }

  /**
   * Get text content from an element matching the provided selector
   */
  async getElementText(selector: string): Promise<string> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      console.log(`Getting text from element with selector: ${selector}`);
      
      // Wait for the element to be visible with reduced timeout
      await this.page.waitForSelector(selector, { visible: true, timeout: 3000 });
      
      // Get the text content
      const text = await this.page.$eval(selector, (element) => element.textContent?.trim() || '');
      
      return text;
    } catch (error) {
      console.error(`Error getting text from element with selector ${selector}:`, error);
      throw error;
    }
  }

  /**
   * Take a screenshot of the current page state
   */
  async takeScreenshot(name: string = 'screenshot'): Promise<string> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      // Create a timestamp for unique filenames but preserve the step naming format
      const timestamp = new Date().toISOString().replace(/:/g, '-');
      const filename = `${name}_${timestamp}.png`;
      const screenshotPath = path.join(this.screenshotsDir, filename);
      
      console.log(`Taking screenshot: ${screenshotPath}`);
      
      // Take the screenshot
      await this.page.screenshot({
        path: screenshotPath,
        fullPage: false,
      });
      
      return screenshotPath;
    } catch (error) {
      console.error('Error taking screenshot:', error);
      throw error;
    }
  }

  /**
   * Get the current page URL
   */
  async getCurrentUrl(): Promise<string> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    return this.page.url();
  }

  /**
   * Sanitize a string for use in filenames
   */
  private sanitizeFilename(input: string): string {
    // Replace non-alphanumeric characters with underscores
    return input
      .replace(/[^a-z0-9]/gi, '_')
      .toLowerCase()
      .substring(0, 50); // Limit length to avoid excessively long filenames
  }

  /**
   * Close the browser and cleanup
   */
  async cleanup(): Promise<void> {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      this.page = null;
      console.log('WebAgent cleaned up successfully');
    }
  }

  /**
   * Get the HTML source code of the current page
   */
  async getPageSource(): Promise<string> {
    if (!this.page) {
      throw new Error('WebAgent not initialized. Call initialize() first.');
    }

    try {
      // Get the page content
      const content = await this.page.content();
      return content;
    } catch (error) {
      console.error('Error getting page source:', error);
      throw error;
    }
  }
}

export default WebAgent; 