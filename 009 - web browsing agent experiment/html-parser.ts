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
    
    // Extract URL from any canonical link
    const canonicalRegex = /<link[^>]*rel=["']canonical["'][^>]*href=["']([^"']*)["'][^>]*>/i;
    const canonicalMatch = canonicalRegex.exec(html);
    if (canonicalMatch) {
      result += `URL: ${canonicalMatch[1]}\n\n`;
    }
    
    // Extract JSON-LD structured data if present
    const jsonLdRegex = /<script[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
    let jsonLdMatch;
    let structuredDataFound = false;
    
    result += `## Structured Data\n`;
    while ((jsonLdMatch = jsonLdRegex.exec(html)) !== null) {
      try {
        const jsonData = JSON.parse(jsonLdMatch[1].trim());
        structuredDataFound = true;
        
        // Extract key information from the structured data
        if (jsonData["@type"]) {
          result += `- Type: ${jsonData["@type"]}\n`;
        }
        
        if (jsonData.name) {
          result += `- Name: "${jsonData.name}"\n`;
        }
        
        if (jsonData.description) {
          result += `- Description: "${jsonData.description.substring(0, 150)}${jsonData.description.length > 150 ? '...' : ''}"\n`;
        }
        
        // Include other useful properties depending on the type
        if (jsonData.offers) {
          if (typeof jsonData.offers === 'object' && jsonData.offers.price) {
            result += `- Price: ${jsonData.offers.price} ${jsonData.offers.priceCurrency || ''}\n`;
          }
        }
        
        if (jsonData.author) {
          const authorName = typeof jsonData.author === 'object' ? jsonData.author.name : jsonData.author;
          if (authorName) {
            result += `- Author: ${authorName}\n`;
          }
        }
        
        result += `\n`;
      } catch (e) {
        // Ignore JSON parsing errors
      }
    }
    
    if (!structuredDataFound) {
      result += `No structured data (JSON-LD) found on the page.\n`;
    }
    
    // Extract page layout information
    result += `\n## Page Layout Analysis\n`;
    
    // Check for common layout elements
    const hasHeader = /<header[^>]*>/.test(html);
    const hasFooter = /<footer[^>]*>/.test(html);
    const hasAside = /<aside[^>]*>/.test(html);
    const hasSidebar = /class=["'][^"']*\b(sidebar|side-bar)\b[^"']*["']/.test(html);
    const hasArticle = /<article[^>]*>/.test(html);
    const hasMain = /<main[^>]*>/.test(html);
    
    // Count sections and important containers
    const sectionCount = (html.match(/<section[^>]*>/g) || []).length;
    const divCount = (html.match(/<div[^>]*>/g) || []).length;
    
    // Determine if it's likely a single-column or multi-column layout
    const columnClassRegex = /class=["'][^"']*\b(column|col-|grid-col)\b[^"']*["']/g;
    const columnMatches = html.match(columnClassRegex) || [];
    const likelyMultiColumn = columnMatches.length > 2;
    
    result += `Structural elements detected:\n`;
    result += `- ${hasHeader ? '✓' : '✗'} Header section\n`;
    result += `- ${hasMain ? '✓' : '✗'} Main content area\n`;
    result += `- ${hasAside || hasSidebar ? '✓' : '✗'} Sidebar/Aside\n`;
    result += `- ${hasFooter ? '✓' : '✗'} Footer section\n`;
    result += `- ${hasArticle ? '✓' : '✗'} Article content\n`;
    result += `- ${sectionCount} section elements\n`;
    result += `- Layout appears to be ${likelyMultiColumn ? 'multi-column' : 'single-column'}\n`;
    
    // Check for forms and iframes which often indicate specific functionality
    const formCount = (html.match(/<form[^>]*>/g) || []).length;
    const iframeCount = (html.match(/<iframe[^>]*>/g) || []).length;
    
    if (formCount > 0) {
      result += `- Contains ${formCount} form(s)\n`;
    }
    
    if (iframeCount > 0) {
      result += `- Contains ${iframeCount} iframe(s) (embedded content)\n`;
    }
    
    // Extract page headings for context
    result += `\n## Page Structure\n`;
    
    // Extract h1-h3 headings
    const headingRegex = /<h([1-3])[^>]*>(.*?)<\/h\1>/gi;
    let headingMatch;
    let headingsFound = false;
    
    while ((headingMatch = headingRegex.exec(html)) !== null) {
      headingsFound = true;
      const headingLevel = headingMatch[1];
      const headingText = headingMatch[2].replace(/<[^>]*>/g, '').trim();
      
      if (headingText) {
        result += `${'#'.repeat(parseInt(headingLevel))} ${headingText}\n`;
      }
    }
    
    if (!headingsFound) {
      result += `No main headings found.\n`;
    }
    
    // Extract main paragraphs (limited to first 5 for brevity)
    const paragraphRegex = /<p[^>]*>(.*?)<\/p>/gi;
    let paragraphMatch;
    let paragraphCount = 0;
    const MAX_PARAGRAPHS = 5;
    
    result += `\n### Main Content Text:\n`;
    while ((paragraphMatch = paragraphRegex.exec(html)) !== null && paragraphCount < MAX_PARAGRAPHS) {
      const paragraphText = paragraphMatch[1].replace(/<[^>]*>/g, '').trim();
      
      if (paragraphText && paragraphText.length > 20) { // Only include substantial paragraphs
        result += `- "${paragraphText.substring(0, 150)}${paragraphText.length > 150 ? '...' : ''}"\n`;
        paragraphCount++;
      }
    }
    
    if (paragraphCount === 0) {
      result += `No main paragraphs found.\n`;
    }
    
    // Extract meta description if available
    const metaDescriptionRegex = /<meta[^>]*name=["']description["'][^>]*content=["']([^"']*)["'][^>]*>/i;
    const metaDescriptionMatch = metaDescriptionRegex.exec(html);
    if (metaDescriptionMatch) {
      result += `\n### Meta Description:\n"${metaDescriptionMatch[1]}"\n`;
    }
    
    // Extract key labels and sections
    result += `\n### Key Labels and Sections:\n`;
    
    // Find form elements with labels
    const formElementsRegex = /<form[^>]*>([\s\S]*?)<\/form>/gi;
    let formMatch;
    let formsFound = false;
    
    while ((formMatch = formElementsRegex.exec(html)) !== null) {
      formsFound = true;
      const formContent = formMatch[1];
      
      // Look for a form title or heading inside the form
      const formTitleRegex = /<h[1-4][^>]*>(.*?)<\/h[1-4]>/i;
      const formTitleMatch = formTitleRegex.exec(formContent);
      const formTitle = formTitleMatch ? formTitleMatch[1].replace(/<[^>]*>/g, '').trim() : 'Form';
      
      result += `- Form: "${formTitle}"\n`;
      
      // Extract labels inside the form
      const labelRegex = /<label[^>]*>(.*?)<\/label>/gi;
      let labelMatch;
      let labelsFound = false;
      
      while ((labelMatch = labelRegex.exec(formContent)) !== null) {
        const labelText = labelMatch[1].replace(/<[^>]*>/g, '').trim();
        
        if (labelText) {
          labelsFound = true;
          result += `  - Label: "${labelText}"\n`;
        }
      }
      
      if (!labelsFound) {
        result += `  - No explicit labels found in form\n`;
      }
    }
    
    if (!formsFound) {
      result += `- No forms detected on page\n`;
    }
    
    // Extract main navigation/menu items
    result += `\n### Navigation Items:\n`;
    const navRegex = /<nav[^>]*>([\s\S]*?)<\/nav>/gi;
    let navMatch;
    let navFound = false;
    
    while ((navMatch = navRegex.exec(html)) !== null) {
      navFound = true;
      const navContent = navMatch[1];
      
      // Extract links from navigation
      const navLinkRegex = /<a[^>]*>(.*?)<\/a>/gi;
      let navLinkMatch;
      let linksFound = false;
      
      while ((navLinkMatch = navLinkRegex.exec(navContent)) !== null) {
        const linkText = navLinkMatch[1].replace(/<[^>]*>/g, '').trim();
        
        if (linkText) {
          linksFound = true;
          result += `- Nav link: "${linkText}"\n`;
        }
      }
      
      if (!linksFound) {
        result += `- No clear navigation links found\n`;
      }
    }
    
    if (!navFound) {
      // Try to find alternative navigation elements
      const potentialMenuRegex = /<(ul|div)[^>]*(class|id)=["']([^"']*\b(menu|nav|navigation)\b[^"']*)["'][^>]*>([\s\S]*?)<\/\1>/gi;
      let potentialMenuMatch;
      let altNavFound = false;
      
      while ((potentialMenuMatch = potentialMenuRegex.exec(html)) !== null) {
        altNavFound = true;
        const menuContent = potentialMenuMatch[5];
        
        // Extract links from this potential menu
        const menuLinkRegex = /<a[^>]*>(.*?)<\/a>/gi;
        let menuLinkMatch;
        let menuLinksFound = false;
        
        while ((menuLinkMatch = menuLinkRegex.exec(menuContent)) !== null) {
          const linkText = menuLinkMatch[1].replace(/<[^>]*>/g, '').trim();
          
          if (linkText) {
            menuLinksFound = true;
            result += `- Menu link: "${linkText}"\n`;
          }
        }
        
        if (!menuLinksFound) {
          result += `- Potential menu found but no clear links extracted\n`;
        }
      }
      
      if (!altNavFound) {
        result += `- No clear navigation structure identified\n`;
      }
    }
    
    result += `\n## Interactive Elements List\n\n`;
    
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

/**
 * Simplifies HTML by removing header tag completely and cleaning the body
 * of script, style, noscript and other non-content tags.
 * @param html - The HTML string to simplify
 * @param removeDataAttr - Whether to remove data-* attributes (default: true)
 * @param removeAria - Whether to remove aria-* attributes (default: true)
 * @returns A simplified HTML string
 */
export function getSimplifiedHtml(
  html: string, 
  removeDataAttr: boolean = true, 
  removeAria: boolean = true
): string {
  try {
    // Remove the entire header section if it exists
    let simplified = html.replace(/<head[\s\S]*?<\/head>/i, '');
    
    // Also remove any header tags that might be in the body
    simplified = simplified.replace(/<header[\s\S]*?<\/header>/gi, '');
    
    // Remove various script and style tags
    const tagsToRemove = [
      'script',
      'style',
      'noscript',
      'iframe',
      'svg',
      'canvas',
      'object',
      'embed',
      'link',
      'meta'
    ];
    
    // Create a regex pattern to match all these tags
    const tagPattern = new RegExp(
      `<(${tagsToRemove.join('|')})([\\s\\S]*?)<\\/\\1>|<(${tagsToRemove.join('|')})([^>]*?)\\/>`, 
      'gi'
    );
    
    // Keep removing tags until no more matches are found (handles nested tags)
    let prevHtml = '';
    while (prevHtml !== simplified) {
      prevHtml = simplified;
      simplified = simplified.replace(tagPattern, '');
    }
    
    // Also remove inline event handlers and javascript: URLs that might contain logic
    simplified = simplified.replace(/\s(on\w+)="[^"]*"/gi, ''); // Remove on* event handlers
    simplified = simplified.replace(/\shref="javascript:[^"]*"/gi, ' href="#"'); // Replace javascript: URLs
    
    // Remove ping attributes from a tags (tracking/analytics feature)
    simplified = simplified.replace(/\sping="[^"]*"/gi, '');
    
    // Remove src attributes from img tags (prevents loading images)
    simplified = simplified.replace(/<img([^>]*)src="[^"]*"([^>]*)>/gi, '<img$1$2>');
    
    // Remove data-* attributes if requested
    if (removeDataAttr) {
      simplified = simplified.replace(/\sdata-[a-zA-Z0-9_-]+="[^"]*"/gi, '');
    }
    
    // Remove aria-* attributes if requested
    if (removeAria) {
      simplified = simplified.replace(/\saria-[a-zA-Z0-9_-]+="[^"]*"/gi, '');
    }
    
    // Remove HTML comments
    simplified = simplified.replace(/<!--[\s\S]*?-->/g, '');
    
    // Remove excess whitespace
    simplified = simplified.replace(/\s{2,}/g, ' ').trim();
    
    return simplified;
  } catch (error) {
    console.error('Error simplifying HTML:', error);
    return html; // Return original if there's an error
  }
}

/**
 * Trims HTML content to fit within a specified maximum length
 * while preserving the structure by removing less important elements first.
 * 
 * @param html - The HTML string to trim
 * @param maxLength - Maximum length in characters (default: 7000)
 * @returns A trimmed HTML string that attempts to preserve important elements
 */
export function trimHtmlContent(html: string, maxLength: number = 7000): string {
  // If HTML is already under the limit, return it as is
  if (html.length <= maxLength) {
    return html;
  }
  
  try {
    // Create a simplified version for starters
    let trimmed = getSimplifiedHtml(html, true, true);
    
    // If it's still too long, start removing elements by priority
    if (trimmed.length > maxLength) {
      // First approach: Remove less important sections
      
      // 1. Try to preserve the main content area
      const mainContentRegex = /<main[^>]*>([\s\S]*?)<\/main>/i;
      const mainMatch = mainContentRegex.exec(trimmed);
      
      if (mainMatch && mainMatch[1]) {
        // If we found main content, focus on that
        let mainContent = mainMatch[1];
        
        // Wrap it in a basic HTML structure
        mainContent = `<html><body><main>${mainContent}</main></body></html>`;
        
        // Check if this is short enough
        if (mainContent.length <= maxLength) {
          return mainContent;
        }
      }
      
      // 2. If that doesn't work, remove deeper nested elements
      // Remove footer completely
      trimmed = trimmed.replace(/<footer[\s\S]*?<\/footer>/gi, '');
      
      // Remove aside/sidebar elements
      trimmed = trimmed.replace(/<aside[\s\S]*?<\/aside>/gi, '');
      trimmed = trimmed.replace(/<div[^>]*class=["'][^"']*\b(sidebar|side-bar)\b[^"']*["'][^>]*>[\s\S]*?<\/div>/gi, '');
      
      // If still too long, limit lists to first few items
      const listReplacer = (match: string) => {
        const items = match.match(/<li[^>]*>[\s\S]*?<\/li>/gi) || [];
        // Keep at most 3 list items, plus list container tags
        const shortened = match.replace(/(<li[^>]*>[\s\S]*?<\/li>)/gi, (item, i) => {
          return i < 3 ? item : '';
        });
        return shortened;
      };
      
      trimmed = trimmed.replace(/<(ul|ol)[^>]*>[\s\S]*?<\/(ul|ol)>/gi, listReplacer);
      
      // If still too long, start truncating content but preserve structure
      if (trimmed.length > maxLength) {
        // Extract body content
        const bodyMatch = /<body[^>]*>([\s\S]*?)<\/body>/i.exec(trimmed);
        let bodyContent = bodyMatch ? bodyMatch[1] : trimmed;
        
        // Truncate body content while preserving important elements
        const preserveTags = ['button', 'a', 'input', 'select', 'textarea', 'form'];
        
        // Find all important elements to prioritize
        let interactiveElements = '';
        preserveTags.forEach(tag => {
          const tagRegex = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>|<${tag}[^>]*\\/>`, 'gi');
          let match;
          while ((match = tagRegex.exec(bodyContent)) !== null) {
            interactiveElements += match[0] + '\n';
          }
        });
        
        // Create a minimal document with just the interactive elements
        const minimalHtml = `<html><body>${interactiveElements}</body></html>`;
        
        // Use that if it's under the limit
        if (minimalHtml.length <= maxLength) {
          return minimalHtml;
        }
        
        // Last resort: hard truncation with a note
        const truncated = trimmed.substring(0, maxLength - 50);
        // Find the last complete tag
        const lastTagEnd = truncated.lastIndexOf('>');
        if (lastTagEnd > 0) {
          return truncated.substring(0, lastTagEnd + 1) + 
                 '<div>[Content truncated due to size limits]</div></body></html>';
        }
        
        // If all else fails, return a simplified stub with a message
        return `<html><body><div>Page content was too large (${html.length} characters) and had to be truncated</div></body></html>`;
      }
    }
    
    return trimmed;
  } catch (error) {
    console.error('Error trimming HTML content:', error);
    // Return a simplified version if trimming fails
    return `<html><body><div>Error processing page content: ${String(error)}</div></body></html>`;
  }
} 