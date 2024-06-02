
/**
 * Abstract interface for XML interaction to reduce coupling
 * 
 * Current Libraries (6/2/2024)
 * * @xmldom/xmldom - for parsing xml to document object model (dom)
 * * xpath - for utilizing xpath queries
 */

import { readFileSync } from 'fs';

import { XmlElement } from './Element.js';
import { assert } from 'console';
import { DOMParser } from '@xmldom/xmldom';

export * from './Element.js'
export * from './ForLoop.js'
export * from './Expression.js'

// namespace
export const ns: any = {'xmlns': 'http://www.srcML.org/srcML/src'}

export function parseXmlFile(filePath: string) : XmlElement {
    return parseXmlString(readFileSync(filePath).toString());
}

export function parseXmlString(xmlString: string) : XmlElement {
    const document = new DOMParser().parseFromString(xmlString);
    const root = document.documentElement;
    if (root) {
        return new XmlElement(root);
    }
    throw new Error("Could not parse Xml String");
}

/**
 * Returns true if the `<name>` node does not contain "." or "[]", false 
 * otherwise
 * @param nameNode 
 */
export function isComplexName(nameNode: XmlElement) : boolean {
    assert(nameNode.name === "name");
    return nameNode.contains("./xmlns:operator") 
        || nameNode.contains("./xmlns:index");
}
