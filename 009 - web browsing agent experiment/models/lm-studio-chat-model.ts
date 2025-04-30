import { LMStudioClient } from "@lmstudio/sdk";

/**
 * Custom LMStudio model adapter for LangChain
 */
export class LMStudioChatModel {
  private client: LMStudioClient;
  private modelName: string;
  private lmStudioModel: any;
  private contextLength: number;

  constructor(modelName: string, contextLength: number = 16000) {
    this.client = new LMStudioClient();
    this.modelName = modelName;
    this.lmStudioModel = null;
    this.contextLength = contextLength;
  }

  async init() {
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

    const response = await this.lmStudioModel.respond(formattedMessages, {
      maxTokens: 4000,
      contextLength: this.contextLength,
      temperature: 0.7,
      topP: 0.9
    });
    
    return response.content;
  }
} 