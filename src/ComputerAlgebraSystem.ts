/**
 * Abstract interface for interacting with a computer algebra system (CAS) 
 * to reduce coupling
 * 
 * * Current Library: Algebrite (typing outlinded in modules/)
 */
// import * as Algebrite from 'algebrite'
import * as Xml from './Xml/Xml.js'
import { execSync } from 'child_process';
import * as Algebrite from 'algebrite'



/**
 * Takes a mathematical expression represented as a string and returns an 
 * expression object representing it
 * @param expression
 */
export function simplify(expression: string) {
    return Algebrite.simplify(expression).toString();
}

export function simplifyXml(expression: Xml.XmlElement) : Xml.XmlElement | null {
    const newExpression = simplify(expression.text);
    if (newExpression === "nil") return expression;
    const language = expression.get("/xmlns:unit")?.getAttribute("language") ?? "";
    const buffer = execSync(`srcml --text ${newExpression} --language ${language}`, {timeout: 10000});
    const bufferRoot = Xml.parseXmlString(buffer.toString());
    if (!bufferRoot) throw new Error("TODO") // TODO: Write better error
    return bufferRoot.get("./xmlns:expr");
}

export function invertExpression(toXml: Xml.XmlElement, from: Xml.XmlElement) : Xml.XmlElement | null {
    if (toXml === null || !from) return null;
    const newExpression = simplify(`${toXml.text} - (${from.text})`)
    if (newExpression === "nil") return null;
    const language = toXml.get("/xmlns:unit")?.getAttribute("language") ?? "";
    const buffer = execSync(`srcml --text ${newExpression} --language ${language}`, {timeout: 10000});
    const rhsRoot = Xml.parseXmlString(buffer.toString());
    return rhsRoot!.get("./expr");
}