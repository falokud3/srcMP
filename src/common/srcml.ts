/**
 * Interface for generating srcML
 */

import { execSync } from 'child_process';
import * as Xml from './Xml/Xml.js';

/**
 * Creates the srcML of a string utilizing a specific language
 * @returns The root XML Element
 */
export function createXml(str: string, language: string) : Xml.Element {
    if (str.startsWith("-")) {
        str = `(${str})`;
    }

    const buffer = execSync(`srcml --text "${str}" --language ${language}`, {timeout: 10000});
    const bufferRoot = Xml.parseXmlString(buffer.toString());
    const xml = bufferRoot.child(0);
    if (!bufferRoot || !xml) throw new Error("Failed to create srcML.");
    return xml;
}

/**
 * Converts file contents to an Xml Object
 * * Program assumes that .xml files passed to program are srcml applied to one file
 * @param srcPath the path to the file as a string
 * @returns an xml object representing the file contents
 */
export function getFileXml(srcPath: string) : Xml.Element {
    const fileExtension = srcPath.substring(srcPath.lastIndexOf("."));
    if (fileExtension === ".xml") {
        return Xml.parseXmlFile(srcPath);
    } else if (fileExtension === ".py") {
        // TODO: Store py2srcml location in configuration file instead of hardcoded
        const buffer = execSync(`python3 py2srcML/py2srcml.py ${srcPath}`, {timeout: 10000, maxBuffer: 1024 * 1024 * 10});
        return Xml.parseXmlString(buffer.toString());
    } else {
        const buffer = execSync(`srcml --position ${srcPath}`, {timeout: 10000, maxBuffer: 1024 * 1024 * 10});
        return Xml.parseXmlString(buffer.toString());
    }
}