
import XmlElement from "./Element.js";

export function hasAugAssignmentOperator(xml: XmlElement) : boolean {
    const op = xml.get("./xmlns:operator");

    if (!op) return false;

    return [...op.text].filter((char) => char === "=").length === 1 && 
        op.text.length > 1;
}

export function hasAssignmentOperator(xml: XmlElement) : boolean {
    const op = xml.get("./xmlns:operator");

    if (!op) return false;

    return op.text.length === 1 && op.text === "=";
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