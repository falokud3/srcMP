
import { execPath } from "process";
import XmlElement from "./Element.js";
import { execSync } from "child_process";

import * as Xml from './Xml.js'

// TODO: Refactor

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

function isAugAssignmentOperator(op: XmlElement) : boolean {
    return [...op.text].filter((char) => char === "=").length === 1 && 
        op.text.length > 1 && !['<=', '>=', '!=', ].includes(op.text);
}

function isAssignmentOperator(op: XmlElement) : boolean {
    return op.text.length === 1 && op.text === "=";
}

export function getRHSFromExpr(exprXml: XmlElement) : XmlElement {
    const expr = exprXml.copy();

    if (!expr) throw new Error();

    const children = Array.from(expr.domElement.childNodes);

    const op = expr.childElements.find((node) => {return isAssignmentOperator(node) 
        || isAugAssignmentOperator(node);});
    const stopIndex = children.findIndex((child) => {
        return child.textContent === op?.text;
    });

    for (let i = 0; i <= stopIndex; i++) {
        expr.domElement.removeChild(children[i]);
    }

    return expr;
}

export function getRHSFromOp(op: XmlElement) : XmlElement {
    const expr = op.parentElement?.copy();

    if (!expr) throw new Error();

    const children = Array.from(expr.domElement.childNodes);
    const stopIndex = children.findIndex((child) => {
        return child.textContent === op.text;
    });

    for (let i = 0; i <= stopIndex; i++) {
        expr.domElement.removeChild(children[i]);
    }

    return expr;
}

export function regularizeAugAssignment(augAssign: XmlElement) : XmlElement {
    const expr = augAssign.copy();
    
    const op = expr.childElements.find((node) => isAugAssignmentOperator(node));
    const from = op?.prevElement
    if (!op || !from) throw new Error('Improper Augmented Assignment Form')

    const newExpression = `(${getRHSFromOp(op).text.trim()}) ${op.text.substring(0,op.text.length - 1)} ${from.text}`;
    const language = augAssign.get("/xmlns:unit")?.getAttribute("language") ?? "";
    const buffer = execSync(`srcml --text '${newExpression}' --language ${language}`, {timeout: 10000});
    const xml = Xml.parseXmlString(buffer.toString()).get("./xmlns:expr")!;

    expr.domElement.parentNode?.replaceChild(xml.domElement, expr.domElement);
    return expr;

}