import { LMStudioClient } from "@lmstudio/sdk";

/**
 * Custom LMStudio model adapter for LangChain
 */
export class LMStudioChatModel {
  private client: LMStudioClient;
  private modelName: string;
  private lmStudioModel: any;
  private modelOptions: any;

  constructor(modelName: string) {
    this.client = new LMStudioClient();
    this.modelName = modelName;
    this.lmStudioModel = null;
    this.modelOptions = {
      maxTokens: 5000,
      temperature: 0.3,
      //topP: 0.9
    };
  }

  async init() {
    
    console.log(`Loading model ${this.modelName}`);
    this.lmStudioModel = await this.client.llm.model(this.modelName);
    
    return this;
  }

  async invoke(messages: any[], options?: { imageFiles?: string[] }) {
    if (!this.lmStudioModel) {
      await this.init();
    }

    const formattedMessages = messages.map(msg => {
      const formattedMsg: any = {
        role: msg.role,
        content: msg.content
      };

      if (options?.imageFiles && msg.role === "user" && options.imageFiles.length > 0) {
        formattedMsg.images = options.imageFiles;
      }

      return formattedMsg;
    });

    try {
      // We're explicitly passing context length in the response parameters
      const response = await this.lmStudioModel.respond(formattedMessages, {
        ...this.modelOptions,
      });
      
      return response.content;
    } catch (error) {
      console.error("Error from LM Studio:", error);
      // If we hit token limit issues, try with HTML extraction
      if (String(error).includes("context") && String(error).includes("overflows")) {
        console.warn("Hit context limit, trying to extract key HTML elements instead...");
        
        // Find the user message with HTML content
        const userMessageIndex = formattedMessages.findIndex(msg => 
          msg.role === "user" && msg.content.includes("<html")
        );
        
        if (userMessageIndex >= 0) {
          // Extract just interactive elements from the HTML
          const htmlContent = formattedMessages[userMessageIndex].content;
          const htmlMatch = /```html\n([\s\S]*?)\n```/g.exec(htmlContent);
          
          if (htmlMatch && htmlMatch[1]) {
            const extractedHtml = this.extractInteractiveElements(htmlMatch[1]);
            formattedMessages[userMessageIndex].content = 
              htmlContent.replace(htmlMatch[1], extractedHtml);
              
            // Try again with the reduced HTML
            console.log("Retrying with extracted HTML elements");
            const response = await this.lmStudioModel.respond(formattedMessages, this.modelOptions);
            return response.content;
          }
        }
        
        throw new Error("Context window overflow - HTML too large to process even after extraction");
      }
      throw error;
    }
  }
  
  /**
   * Extracts just the interactive elements from HTML to reduce token usage
   */
  private extractInteractiveElements(html: string): string {
    try {
      // Extract the title
      const titleMatch = /<title>(.*?)<\/title>/i.exec(html);
      const title = titleMatch ? titleMatch[1] : 'Unknown Page';
      
      let result = `<html><head><title>${title}</title></head><body>\n`;
      result += `<p>HTML was ${html.length} characters - extracted only interactive elements:</p>\n`;
      
      // Extract common interactive elements
      const extractPatterns = [
        { type: 'Form', pattern: /<form[^>]*>([\s\S]*?)<\/form>/gi },
        { type: 'Button', pattern: /<button[^>]*>(.*?)<\/button>/gi },
        { type: 'Input', pattern: /<input[^>]*>/gi },
        { type: 'Select', pattern: /<select[^>]*>([\s\S]*?)<\/select>/gi },
        { type: 'Link', pattern: /<a[^>]*href="([^"]*)"[^>]*>(.*?)<\/a>/gi }
      ];
      
      // Extract each element type
      for (const { type, pattern } of extractPatterns) {
        let match;
        while ((match = pattern.exec(html)) !== null) {
          result += `<${type.toLowerCase()}>${match[0]}</${type.toLowerCase()}>\n`;
        }
      }
      
      result += '</body></html>';
      return result;
    } catch (error) {
      console.error('Error extracting HTML elements:', error);
      return html.substring(0, 4000) + '... [truncated]';
    }
  }
} 