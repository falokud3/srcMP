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