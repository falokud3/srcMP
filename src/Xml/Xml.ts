
/**
 * Abstract interface for XML interaction to reduce coupling
 * 
 * ! Current Library (5/21/2023): libxmljs2 (has a high vulnerability)
 */

import { readFileSync } from 'fs';
import * as libxmljs2 from 'libxmljs2';
import { Element as xmlElement } from './Element.js';
import assert from "assert";

export * from './Element.js'
export * from './Loop.js'
export * from './Expression.js'

// namespace
export const ns: libxmljs2.StringMap = {'xmlns': 'http://www.srcML.org/srcML/src'}

export function parseXmlFile(filePath: string) {
    return parseXmlString(readFileSync(filePath).toString());
}

export function parseXmlString(xmlString: string) {
    const doc = libxmljs2.parseXmlString(xmlString);
    return new xmlElement(doc.root());
}

/**
 * Returns true if the `<name>` node does not contain "." or "[]", false 
 * otherwise
 * @param nameNode 
 */
export function isComplexName(nameNode: xmlElement) : boolean {
    assert(nameNode.name === "name");
    return nameNode.contains("./xmlns:operator") 
        || nameNode.contains("./xmlns:index");
}
