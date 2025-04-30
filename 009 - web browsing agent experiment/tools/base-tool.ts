import { Tool } from "@langchain/core/tools";
import WebAgent from "../web-agent";

/**
 * Base class for all WebAgent tools
 */
abstract class WebAgentTool extends Tool {
  name: string;
  description: string;
  webAgent: WebAgent;
  stepNumber: number;
  
  constructor(name: string, description: string, webAgent: WebAgent, stepNumber: number) {
    super();
    this.name = name;
    this.description = description;
    this.webAgent = webAgent;
    this.stepNumber = stepNumber;
  }
  
  abstract _call(args: any): Promise<string>;
}

export default WebAgentTool; 