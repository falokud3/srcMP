
import { XmlElement } from "./Element.js";

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