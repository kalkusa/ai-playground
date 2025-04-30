import puppeteer from 'puppeteer';
import path from 'path';

export async function getFacebookScreenshot(outputPath: string): Promise<string> {
  console.log('Launching browser...');
  
  // Launch the browser
  const browser = await puppeteer.launch({
    headless: true,
  });
  
  try {
    // Create a new page
    const page = await browser.newPage();
    
    // Set viewport size for a decent screenshot
    await page.setViewport({
      width: 1280,
      height: 800,
    });
    
    console.log('Navigating to Facebook...');
    // Navigate to Facebook
    await page.goto('https://facebook.com', {
      waitUntil: 'networkidle2', // Wait until the network is idle
    });
    
    // Resolve the output path
    const screenshotPath = path.resolve(outputPath);
    
    console.log('Taking screenshot...');
    // Take the screenshot
    await page.screenshot({
      path: screenshotPath,
      fullPage: false,
    });
    
    console.log(`Screenshot saved to: ${screenshotPath}`);
    return screenshotPath;
  } catch (error) {
    console.error('Error taking screenshot:', error);
    throw error;
  } finally {
    // Always close the browser
    await browser.close();
  }
} 