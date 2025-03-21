import XmlElement from "./Element.js";
import { execSync } from "child_process";

import * as Xml from './Xml.js';

export function hasAugAssignmentOperator(xml: XmlElement) : boolean {
    const op = xml.get("./xmlns:operator");

    if (!op) return false;

    return isAugAssignmentOperator(op);
}

export function hasAssignmentOperator(xml: XmlElement) : boolean {
    const op = xml.get("./xmlns:operator");

    if (!op) return false;

    return isAssignmentOperator(op);
}

export function isAugAssignmentOperator(op: XmlElement) : boolean {
    return [...op.text].filter((char) => char === "=").length === 1 && 
        op.text.length > 1 && !['<=', '>=', '!=', ].includes(op.text);
}

export function isAssignmentOperator(op: XmlElement) : boolean {
    return op.text.length === 1 && op.text === "=";
}

export function getRHSFromExpr(exprXml: XmlElement) : XmlElement {
    const expr = exprXml.copy();

    if (!expr) throw new Error();

    const children = Array.from(expr.domNode.childNodes);

    const op = expr.childElements.find((node) => {return isAssignmentOperator(node) 
        || isAugAssignmentOperator(node);});
    const stopIndex = children.findIndex((child) => {
        return child.textContent === op?.text;
    });

    for (let i = 0; i <= stopIndex; i++) {
        expr.domNode.removeChild(children[i]);
    }

    return expr;
}

export function getRHSFromOp(op: XmlElement) : XmlElement {
    const expr = op.parentElement?.copy();

    if (!expr) throw new Error();

    const children = Array.from(expr.domNode.childNodes);
    const stopIndex = children.findIndex((child) => {
        return child.textContent === op.text;
    });

    for (let i = 0; i <= stopIndex; i++) {
        expr.domNode.removeChild(children[i]);
    }

    return expr;
}

export function regularizeAugAssignment(augAssign: XmlElement) : XmlElement {
    const expr = augAssign.copy();
    
    const op = expr.childElements.find((node) => isAugAssignmentOperator(node));
    const from = op?.prevElement;
    if (!op || !from) throw new Error('Improper Augmented Assignment Form');

    const newExpression = `(${getRHSFromOp(op).text.trim()}) ${op.text.substring(0,op.text.length - 1)} ${from.text}`;
    const language = augAssign.get("/xmlns:unit")?.getAttribute("language") ?? "";
    const buffer = execSync(`srcml --text '${newExpression}' --language ${language}`, {timeout: 10000});
    const xml = Xml.parseXmlString(buffer.toString()).get("./xmlns:expr")!;

    expr.domNode.parentNode?.replaceChild(xml.domNode, expr.domNode);
    return expr;

}