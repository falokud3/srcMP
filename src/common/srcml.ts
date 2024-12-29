import { execSync } from 'child_process';
import * as Xml from './Xml/Xml.js';

export function createXml(str: string, language: string) : Xml.Element | null {
    if (str.startsWith("-")) {
        str = `(${str})`;
    }

    const buffer = execSync(`srcml --text "${str}" --language ${language}`, {timeout: 10000});
    const bufferRoot = Xml.parseXmlString(buffer.toString());
    if (!bufferRoot) throw new Error("TODO"); // TODO: Write better error
    return bufferRoot.child(0);
}

/**
 * Converts file contents to an Xml Object
 * * Program assumes that .xml files passed to program are srcml applied to one file
 * @param srcPath the path to the file as a string
 * @returns an xml object representing the file contents
 */
export function getFileXml(srcPath: string) : Xml.Element {
    
    const fileExtension = srcPath.substring(srcPath.lastIndexOf("."));
    
    if (fileExtension !== ".xml") {
        const buffer = execSync(`srcml --position ${srcPath}`, {timeout: 10000, maxBuffer: 1024 * 1024 * 10});
        return Xml.parseXmlString(buffer.toString());
    } else {
        return Xml.parseXmlFile(srcPath);
    }
}