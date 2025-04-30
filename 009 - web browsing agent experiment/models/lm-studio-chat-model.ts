import { LMStudioClient } from "@lmstudio/sdk";

/**
 * Custom LMStudio model adapter for LangChain
 */
export class LMStudioChatModel {
  private client: LMStudioClient;
  private modelName: string;
  private lmStudioModel: any;

  constructor(modelName: string) {
    this.client = new LMStudioClient();
    this.modelName = modelName;
    this.lmStudioModel = null;
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

    const response = await this.lmStudioModel.respond(formattedMessages);
    return response.content;
  }
} 