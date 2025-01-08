
import { assert } from "console";

import { ArrayAccess } from '../../srcMP/DataDependenceTesting/ArrayAccess.js';
import XmlElement from "./Element.js";
import * as Xml from './Xml.js';
import * as CAS from '../ComputerAlgebraSystem.js';


export class ForLoop extends XmlElement {

    public constructor(domElement: Element) {
        assert(domElement.tagName === "for");
        super(domElement);
    }

    public static createLoop(element: XmlElement) {
        return new ForLoop(element.domNode);
    }

    // gets the loopnest that surrounds this loop (includes loop)
    public getEnclosingLoopNest() : ForLoop[] {
        return <ForLoop[]> this.find("ancestor-or-self::xmlns:for");
    }

    // gets the loops nested within this loop (includes input loop)
    public getInnerLoopNest() : ForLoop[] {
        return <ForLoop[]> this.find("descendant-or-self::xmlns:for");
    }

    // returns the intersection of two enclosing loop nests
    public getCommonEnclosingLoopNest(otherLoop: ForLoop) : ForLoop[] {
        const loopNest = this.getEnclosingLoopNest();
        const other_loopNest = otherLoop.getEnclosingLoopNest();
        return ForLoop.getCommonLoops(loopNest, other_loopNest);
    }

    public get header() : XmlElement {
        const header = this.get("./xmlns:control");
        if (!header) throw new Error("SrcML is not properly formatted");
        return header;
    }

    public get initialization() : XmlElement {
        const init = this.get("./xmlns:control/xmlns:init");
        if (!init) throw new Error("SrcML is not properly formatted");
        return init;
    }

    public get condition() : XmlElement {
        const cond = this.get("./xmlns:control/xmlns:condition");
        if (!cond) throw new Error("SrcML is not properly formatted");
        return cond;
    }

    public get increment() : XmlElement {
        const incr = this.get("./xmlns:control/xmlns:incr");
        if (!incr) throw new Error("SrcML is not properly formatted");
        return incr;
    }

    public get body() : XmlElement {
        const body = this.get("./xmlns:block/xmlns:block_content");
        if (!body) throw new Error("SrcML is not properly formatted");
        return body;
    }

    public getLoopIndexVariableName() : XmlElement | null {
        assert(this.name === "for");
        return this.get("xmlns:control/xmlns:incr/xmlns:expr/xmlns:name");
    }

    /**
     * Returns a map containing all the array access of a loop where the
     * key is the array name and the value is a list of each array acess
     * ! Assumes <index> is associated with an array not an object
     */
    public getArrayAccesses() : Map<string, ArrayAccess[]> {
        const accessMap = new Map<string, ArrayAccess[]>();
        const accesses = this.find('.//xmlns:name[xmlns:index]');
        accesses.forEach((access: Xml.Element) => {
            const parentStmt = access.parentElement;

            if (!parentStmt) return;
            if (parentStmt.name === "decl") return;

            const accessOnLHS = parentStmt.child(0)?.equals(access);
            const newAccesses: ArrayAccess[] = [];

            if (accessOnLHS && Xml.hasAssignmentOperator(parentStmt)) {
                newAccesses.push(new ArrayAccess(access, ArrayAccess.WRITE_ACCESS));
            } else if (accessOnLHS && Xml.hasAugAssignmentOperator(parentStmt)) {
                newAccesses.push(new ArrayAccess(access, ArrayAccess.WRITE_ACCESS));
                newAccesses.push(new ArrayAccess(access, ArrayAccess.READ_ACCESS));
            } else {
                newAccesses.push(new ArrayAccess(access, ArrayAccess.READ_ACCESS));
            }

            const hasUnaryAssignment = ['++', '--'].includes(access.prevElement?.text ?? "") 
                || ['++', '--'].includes(access.nextElement?.text ?? "") ;
            if (hasUnaryAssignment && newAccesses.length < 2) {
                newAccesses.push(new ArrayAccess(access, 
                    newAccesses[0].access_type === ArrayAccess.READ_ACCESS ? 
                    ArrayAccess.WRITE_ACCESS : ArrayAccess.READ_ACCESS ));
            }

            const arrayAccesses = accessMap.get(newAccesses[0].arrayName) ?? [];
            if (arrayAccesses.length === 0) accessMap.set(newAccesses[0].arrayName, arrayAccesses);
            arrayAccesses.push(...newAccesses);
        });
        return accessMap;
    }

    // returns the intersection of two loop nests
    public static getCommonLoops(loopNest: ForLoop[], other_loopNest: ForLoop[]) : ForLoop[] {
        return loopNest.filter((node: ForLoop) => {
            return other_loopNest.some((inner) => {
                return node.text === inner.text;
            });
        });
    }

    public get upperboundExpression() : string | number {
        const xmlCondOp = this.get('xmlns:control/xmlns:condition/xmlns:expr/xmlns:operator');
        if (!xmlCondOp) throw new Error("UpperBoundExpression Missing Condition"); //TODO: change this
        const condOp = xmlCondOp.text;
        const step = getCanonicalIncrementValue(this);
        const rhs = Xml.getRHSFromOp(xmlCondOp).text;
        let ub : string | undefined;
        if (condOp === '<=' || condOp === '>=') {
            ub = CAS.simplify(rhs);
        } else if (condOp === '<') {
            ub = CAS.simplify(`${rhs} - (${step})`);
        } else if (condOp === '>') {
            ub = CAS.simplify(`${rhs} + (${step})`);
        }

        if (ub === undefined) {
            throw new Error("Unexpected Op");
        }
        
        const num = Number(ub);

        if (!Number.isNaN(num)) {
            return num;
        }

        // TODO: RANGE ANALYSIS
        // RangeAnalysis.query(xmlCondOp)?.substituteForward(ub);
        return ub;

    }

    public get lowerboundExpression() : string | number {
        const init = this.initialization;
        let expr: Xml.Element | null = null;
        if (init.contains('xmlns:decl')) {
            expr = init.get('xmlns:decl/xmlns:init/xmlns:expr');
        } else if (init.contains('xmlns:expr')) {
            expr = init.get('xmlns:expr');
        }

        if (!expr) throw new Error("Missing Lowerbound Expresion");

        const exprString = CAS.simplify(expr.text);
        const exprNum = Number(exprString);
        // TODO: Range Analysis
        return !Number.isNaN(exprNum) ? exprNum : exprString;
    }

}

/**
 * Returns true if the loop increment can be resolved to an integer value
 * ! Assumes that the loop is in canonical form
 * @param loop Loop in canonical form
 */
export function getCanonicalIncrementValue(loop: Xml.ForLoop): number | string {
   const incrExpr = loop.increment.child(0);
   let incrStep: Xml.Element = loop;
   let isNegativeStep: boolean = false;
   if (!incrExpr) return 'N/A'; // TODO : should probably be undefined

   // TODO: Add Assert
   if (incrExpr.contains("./xmlns:operator[text()='++']")) {
      return 1;
   } else if (incrExpr.contains("./xmlns:operator[text()='--']")) {
      return -1;
   } else if (incrExpr.contains("./xmlns:operator[text()='+=']")) {
      incrStep = incrExpr.child(2)!;
   } else if (incrExpr.contains("./xmlns:operator[text()='-=']")) {
      isNegativeStep = true;
      incrStep = incrExpr.child(2)!;
   } else if (incrExpr.contains("./xmlns:operator[text()='=']")) {
      const indexVariable = getCanonicalIndexVariable(loop);
      // i = i - step    i = i + step    i = step + i
      if (incrExpr.child(3)?.text === "-") {
         isNegativeStep = true;
         incrStep = incrExpr.child(4)!;
      } else {
         if (incrExpr.child(2)?.equals(indexVariable!)) {
            incrStep = incrExpr.child(4)!;
         } else {
            incrStep = incrExpr.child(2)!;
         }
      }
   }

   if (incrStep.name === "literal") {

      const stepValue = Number(incrStep.text);

      if (Number.isInteger(stepValue)) {
        return isNegativeStep ? -1 * stepValue : stepValue;
      } else {
        return incrStep.text;
      }

   }
   return incrStep.text;
}/**
 * Returns the loop index variable if the initilization expression has one of
 * the following forms (null otherwise):
 * * indexVar = lb
 * * integer-type indexVar = lb
 */
export function getCanonicalIndexVariable(loop: Xml.ForLoop): Xml.Element | null {
// TODO: check if variable is type int and allow for declarations outside init
   const init = loop.initialization;
   // handles having no init and multiple init scenarios
   if (init.childElements.length !== 1) return null;

   const initStatement = init.child(0)!;
   const variableLocation = initStatement.name === "decl" ? 1 : 0;
   const variable = initStatement.child(variableLocation)!;

   if (Xml.isComplexName(variable)) return null;

   // Disasllows Augmented Assignment & cases like i = j = 0
   if (initStatement.name === "expr" && (initStatement.child(1)?.text !== "="
      || initStatement.find("./xmlns:operator[contains(text(),'=')]").length !== 1)) return null;

   return variable;
}

export function isLoopInvariant(loop: Xml.ForLoop, expr: string) {
    const incrSymbols = (loop.get('xmlns:control/xmlns:incr')?.defSymbols ?? new Set<Xml.Element>());
    const bodySymbols = loop.get('xmlns:body')?.defSymbols ?? new Set<Xml.Element>();

    const loopSymbols = new Set<string>();
    const addSymbols = (node: Xml.Element) => loopSymbols.add(node.text);

    incrSymbols.forEach(addSymbols);
    bodySymbols.forEach(addSymbols);

    const exprSymbols = CAS.getVariables(expr);

    for (const symbol of exprSymbols) {
        if (loopSymbols.has(symbol)) {
            return false;
        }
    }
    return true;
}

