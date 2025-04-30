/**
 * Helper function to pause execution for a specified time
 * @param ms Time to pause in milliseconds
 */
export const delay = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms)); 