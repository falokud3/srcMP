/**
 * Abstract interface for interacting with a computer algebra system (CAS) 
 * to reduce coupling
 * 
 * * Current Library: Algebrite (typing outlinded in modules/)
 */
// import * as Algebrite from 'algebrite'
import * as Xml from './Xml/Xml.js';
import { execSync } from 'child_process';
import algebrite from 'algebrite';

/**
 * Algebrite outputs this value for true comparisons (==, <, etc.)
 */
export const TRUE = 1;

/**
 * Algebrite outputs this value for false comparisons (==, <, etc.)
 */
export const FALSE = 0;

/**
 * Takes a mathematical expression represented as a string and returns an 
 * expression object representing it
 * @param expression
 */
// TODO: CHANGE TO return string | number
export function simplify(expression: string)  {
    algebrite.clearall();
    try {
        return algebrite.simplify(expression).toString();
    } catch (Error) {
        console.log(Error);
        return expression;
    }
}

export function safeSimplify(expression: string) {
    expression = expression.replace(/([\w\*\&\[\]\.\(\)]+)\s*\+\+/gm, '$1');
    expression = expression.replace(/\+\+\s*([\w\*\&\[\]\.\(\)]+)/gm, '$1');
    expression = expression.replace(/([\w\*\&\[\]\.\(\)]+)\s*--/gm, '$1');
    expression = expression.replace(/--\s*([\w\*\&\[\]\.\(\)]+)/gm, '$1');

    // TODO: convert . expressions to variables
    // TODO: deconvert variables to expressions

    return simplify(expression);

}

// TODO: REMOVE split into simply and  create XML
export function simplifyXml(expression: Xml.Element) : Xml.Element | null {
    let newExpression = safeSimplify(expression.text);
    if (newExpression === "nil") return expression;

    // srcml currently has a bug that causes errors when parsing text leading with "-"
    if (newExpression.startsWith("-")) {
        newExpression = `(${newExpression})`;
    }

    const language = expression.get("/xmlns:unit")?.getAttribute("language") ?? "";
    const buffer = execSync(`srcml --text ${newExpression} --language ${language}`, {timeout: 10000});
    const bufferRoot = Xml.parseXmlString(buffer.toString());
    if (!bufferRoot) throw new Error("TODO"); // TODO: Write better error
    return bufferRoot.get("./xmlns:expr");
}

export function invertExpression(toXml: Xml.Element, from: Xml.Element) : Xml.Element | null {
    if (toXml === null || !from) return null;
    const diff = simplify(`${toXml.text} - (${from.text})`);
    if (diff === "nil" || diff.includes(toXml.text)) return null;

    let inverted = simplify(`${toXml.text} + (${diff})`);

    // srcml currently has a bug that causes errors when parsing text leading with "-"
    if (inverted.startsWith("-")) {
        inverted = `(${inverted})`;
    }
    
    // const newExpr = Number(diff)
    // // srcML won't parse solo negative numbers like -1
    // if (newExpr < 0) {
    //     const exprXML = `<expr><operator>-</operator><literal type="number">${Math.abs(newExpr)}</literal></expr>`;
    //     return Xml.parseXmlString(exprXML);
    // }

    const language = toXml.get("/xmlns:unit")?.getAttribute("language") ?? "";
    const buffer = execSync(`srcml --text "${inverted}" --language ${language}`, {timeout: 10000});
    const rhsRoot = Xml.parseXmlString(buffer.toString());
    return rhsRoot.get("./xmlns:expr");
}

/**
 * Extract variables from an expressino
 * @param expression 
 */
export function getVariables(expression: string) : Set<string> {
    const regex = /[a-zA-Z_][a-zA-Z0-9_]*/g;
    const matches = regex.exec(expression);
    return matches ? new Set<string>(matches) : new Set<string>();
}

export type Inequaliy = '<' | '>' | '<=' | '>=' | '=';

// undefined represetns != AND scenarios where the inequality cannot be determined do to symbolic
    // values
export function compare(lhs: string, rhs: string) : Inequaliy | undefined {
    let ret: Inequaliy | undefined = undefined;
    if (Number(simplify(`${lhs} < ${rhs}`)) === TRUE) ret = '<';
    else if (Number(simplify(`${lhs} > ${rhs}`)) === TRUE) ret = '>';

    if (Number(simplify(`${lhs} == ${rhs}`)) === TRUE) {
        if (ret === undefined) ret = '=';
        else if (ret === '<') ret = '<=';
        else if (ret === '>') ret = '>=';
    }

    return ret;
}