/**
 * Extracts interactive elements from HTML and returns them in a structured format.
 * This helps the LLM identify elements to interact with using selectors.
 */
export function getInteractiveElementList(html: string): string {
  try {
    // Extract the title for informational purposes
    const titleMatch = /<title>(.*?)<\/title>/i.exec(html);
    const title = titleMatch ? titleMatch[1] : 'Unknown Page';
    
    let result = `# Page: ${title}\n\n`;
    result += `## Interactive Elements List\n\n`;
    
    // Extract buttons
    result += `### Buttons\n`;
    
    // Regular button elements
    const buttonRegex = /<button[^>]*>(.*?)<\/button>/gi;
    let buttonMatch;
    let buttonFound = false;
    
    while ((buttonMatch = buttonRegex.exec(html)) !== null) {
      buttonFound = true;
      const buttonTag = buttonMatch[0];
      const buttonText = buttonMatch[1].replace(/<[^>]*>/g, '').trim();
      
      const idMatch = /id=["']([^"']*)["']/i.exec(buttonTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(buttonTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(buttonTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(buttonTag);
      
      result += `- Button: "${buttonText || '[No text]'}"\n`;
      result += `  - CSS: button`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: button[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: button[name="${nameMatch[1]}"]`;
      }
      
      if (buttonText) {
        result += `\n  - Text Selector: text="${buttonText}"`;
      }
      
      result += '\n\n';
    }
    
    // Input elements (buttons)
    const inputButtonRegex = /<input[^>]*type=["'](submit|button)["'][^>]*>/gi;
    while ((buttonMatch = inputButtonRegex.exec(html)) !== null) {
      buttonFound = true;
      const buttonTag = buttonMatch[0];
      
      const valueMatch = /value=["']([^"']*)["']/i.exec(buttonTag);
      const buttonText = valueMatch ? valueMatch[1] : '[No text]';
      
      const idMatch = /id=["']([^"']*)["']/i.exec(buttonTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(buttonTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(buttonTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(buttonTag);
      const typeMatch = /type=["']([^"']*)["']/i.exec(buttonTag);
      const type = typeMatch ? typeMatch[1] : 'submit';
      
      result += `- Input ${type} button: "${buttonText}"\n`;
      result += `  - CSS: input[type="${type}"]`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: input[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: input[name="${nameMatch[1]}"]`;
      }
      
      if (valueMatch) {
        result += `\n  - Value Selector: input[value="${buttonText}"]`;
        result += `\n  - Text Selector: text="${buttonText}"`;
      }
      
      result += '\n\n';
    }
    
    if (!buttonFound) {
      result += `No button elements found.\n\n`;
    }
    
    // Extract input fields
    result += `### Input Fields\n`;
    
    // Input elements - Enhanced with special attention to textareas
    const inputRegex = /<(input|textarea)[^>]*>/gi;
    let inputMatch;
    let inputFound = false;
    
    while ((inputMatch = inputRegex.exec(html)) !== null) {
      const inputTag = inputMatch[0];
      const tagName = inputMatch[1].toLowerCase(); // input or textarea
      
      // Skip buttons and hidden inputs (only for input tags)
      if (tagName === 'input') {
        const typeMatch = /type=["']([^"']*)["']/i.exec(inputTag);
        const inputType = typeMatch ? typeMatch[1].toLowerCase() : 'text';
        
        if (inputType === 'button' || inputType === 'submit' || inputType === 'hidden') {
          continue;
        }
      }
      
      inputFound = true;
      
      // Get input type for either input or textarea
      let inputType = 'text';
      if (tagName === 'input') {
        const typeMatch = /type=["']([^"']*)["']/i.exec(inputTag);
        inputType = typeMatch ? typeMatch[1].toLowerCase() : 'text';
      } else if (tagName === 'textarea') {
        inputType = 'textarea';
      }
      
      const idMatch = /id=["']([^"']*)["']/i.exec(inputTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(inputTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(inputTag);
      const placeholderMatch = /placeholder=["']([^"']*)["']/i.exec(inputTag);
      const valueMatch = /value=["']([^"']*)["']/i.exec(inputTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(inputTag);
      const ariaLabelMatch = /aria-label=["']([^"']*)["']/i.exec(inputTag);
      
      // Look for a label that might be associated with this input
      let labelText = '';
      if (idMatch) {
        const labelForRegex = new RegExp(`<label[^>]*for=["']${idMatch[1]}["'][^>]*>(.*?)<\/label>`, 'i');
        const labelMatch = labelForRegex.exec(html);
        if (labelMatch) {
          labelText = labelMatch[1].replace(/<[^>]*>/g, '').trim();
        }
      }
      
      // Add aria-label to the possible description sources
      const description = ariaLabelMatch?.[1] || labelText || placeholderMatch?.[1] || nameMatch?.[1] || idMatch?.[1] || (tagName === 'textarea' ? 'Textarea' : 'Input');
      
      const inputTypeText = tagName === 'textarea' ? 'textarea' : `input (${inputType || 'text'})`;
      result += `- ${inputTypeText}: "${description}"\n`;
      result += `  - CSS: ${tagName}`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: ${tagName}[id="${idMatch[1]}"]`;
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: ${tagName}[name="${nameMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (placeholderMatch) {
        result += `\n  - Placeholder Selector: ${tagName}[placeholder="${placeholderMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (ariaLabelMatch) {
        result += `\n  - Aria-Label Selector: ${tagName}[aria-label="${ariaLabelMatch[1]}"]`;
      }
      
      result += '\n\n';
    }
    
    if (!inputFound) {
      result += `No input elements found.\n\n`;
    }
    
    // Extract links
    result += `### Links\n`;
    
    const linkRegex = /<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/gi;
    let linkMatch;
    let linkFound = false;
    
    while ((linkMatch = linkRegex.exec(html)) !== null) {
      linkFound = true;
      const linkTag = linkMatch[0];
      const href = linkMatch[1];
      const linkText = linkMatch[2].replace(/<[^>]*>/g, '').trim();
      
      if (!linkText || href.startsWith('javascript:') || href === '#') {
        continue; // Skip empty links or JS links
      }
      
      const idMatch = /id=["']([^"']*)["']/i.exec(linkTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(linkTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(linkTag);
      
      result += `- Link: "${linkText}" (${href})\n`;
      result += `  - CSS: a`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: a[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (linkText) {
        result += `\n  - Text Selector: text="${linkText}"`;
      }
      
      if (href && !href.startsWith('#')) {
        result += `\n  - Href Selector: a[href="${href}"]`;
      }
      
      result += '\n\n';
    }
    
    if (!linkFound) {
      result += `No link elements found.\n\n`;
    }
    
    // Extract select/dropdown elements
    result += `### Dropdowns\n`;
    
    const selectRegex = /<select[^>]*>([\s\S]*?)<\/select>/gi;
    let selectMatch;
    let selectFound = false;
    
    while ((selectMatch = selectRegex.exec(html)) !== null) {
      selectFound = true;
      const selectTag = selectMatch[0];
      const selectContent = selectMatch[1];
      
      const idMatch = /id=["']([^"']*)["']/i.exec(selectTag);
      const nameMatch = /name=["']([^"']*)["']/i.exec(selectTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(selectTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(selectTag);
      
      // Look for label for this select
      let labelText = '';
      if (idMatch) {
        const labelForRegex = new RegExp(`<label[^>]*for=["']${idMatch[1]}["'][^>]*>(.*?)<\/label>`, 'i');
        const labelMatch = labelForRegex.exec(html);
        if (labelMatch) {
          labelText = labelMatch[1].replace(/<[^>]*>/g, '').trim();
        }
      }
      
      const description = labelText || nameMatch?.[1] || idMatch?.[1] || 'Dropdown';
      
      result += `- Select dropdown: "${description}"\n`;
      result += `  - CSS: select`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: select[id="${idMatch[1]}"]`;
      }
      
      if (nameMatch) {
        result += `\n  - Name Selector: select[name="${nameMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
        }
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      // Extract options
      const optionRegex = /<option[^>]*value=["']([^"']*)["'][^>]*>(.*?)<\/option>/gi;
      let optionMatch;
      let optionsText = '\n  - Options:';
      let optionsFound = false;
      
      while ((optionMatch = optionRegex.exec(selectContent)) !== null) {
        optionsFound = true;
        const value = optionMatch[1];
        const text = optionMatch[2].replace(/<[^>]*>/g, '').trim();
        optionsText += `\n    - "${text}" (value: ${value})`;
      }
      
      if (optionsFound) {
        result += optionsText;
      }
      
      result += '\n\n';
    }
    
    if (!selectFound) {
      result += `No select dropdown elements found.\n\n`;
    }
    
    // Extract clickable/interactive div elements
    result += `### Clickable Divs/Spans\n`;
    
    const clickableDivRegex = /<(div|span)[^>]*(onclick|role=["'](button|link|tab|menuitem)["']|class=["'][^"']*\b(btn|button|clickable)\b[^"']*["'])[^>]*>(.*?)<\/\1>/gi;
    let divMatch;
    let divFound = false;
    
    while ((divMatch = clickableDivRegex.exec(html)) !== null) {
      divFound = true;
      const divTag = divMatch[0];
      const tagName = divMatch[1]; // div or span
      const divText = divMatch[5].replace(/<[^>]*>/g, '').trim();
      
      const idMatch = /id=["']([^"']*)["']/i.exec(divTag);
      const classMatch = /class=["']([^"']*)["']/i.exec(divTag);
      const roleMatch = /role=["']([^"']*)["']/i.exec(divTag);
      const dataIdMatch = /data-id=["']([^"']*)["']/i.exec(divTag);
      
      const role = roleMatch ? roleMatch[1] : 'clickable';
      
      result += `- ${tagName.charAt(0).toUpperCase() + tagName.slice(1)} (${role}): "${divText || '[No text]'}"\n`;
      result += `  - CSS: ${tagName}`;
      
      if (idMatch) {
        result += `\n  - ID Selector: #${idMatch[1]}`;
        result += `\n  - Full Selector: ${tagName}[id="${idMatch[1]}"]`;
      }
      
      if (dataIdMatch) {
        result += `\n  - Data-ID Selector: [data-id="${dataIdMatch[1]}"]`;
      }
      
      if (classMatch) {
        const classes = classMatch[1].trim().split(/\s+/);
        if (classes.length > 0) {
          result += `\n  - Class Selector: .${classes[0]}`;
          if (classes.length > 1) {
            result += ` (additional classes: ${classes.slice(1).join(', ')})`;
          }
        }
      }
      
      if (roleMatch) {
        result += `\n  - Role Selector: ${tagName}[role="${roleMatch[1]}"]`;
      }
      
      if (divText) {
        result += `\n  - Text Selector: text="${divText}"`;
      }
      
      result += '\n\n';
    }
    
    if (!divFound) {
      result += `No clickable div/span elements found.\n\n`;
    }
    
    return result;
  } catch (error) {
    console.error('Error extracting interactive elements:', error);
    return `Error extracting interactive elements: ${String(error)}`;
  }
} 