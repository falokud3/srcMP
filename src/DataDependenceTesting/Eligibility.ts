import { getCanonicalIndexVariable } from '../Xml/ForLoop.js';
import { getCanonicalIncrementValue } from '../Xml/ForLoop.js';
import * as Xml from '../Xml/Xml.js';

export function getTestableLoops(root: Xml.Element): Xml.ForLoop[] {
   const loopElements = <Xml.ForLoop[]>root.find(".//xmlns:for[count(ancestor::xmlns:for)=0]");

   loopElements.forEach((loopNode) => {
      isLoopTestEligible(loopNode);
   });

   // filter for eligibility
   return loopElements;
}
function isLoopTestEligible(loop: Xml.ForLoop): boolean {
   // TODO : Allow parallelizable calls from standard library
   return isCanonicalLoop(loop) && !loop.contains(".//xmlns:call")
      && getCanonicalIncrementValue(loop) !== undefined;
}
/**
 * Returns true if the loop is in canonical form
 * @see https://www.openmp.org/spec-html/5.1/openmpsu45.html
 */
function isCanonicalLoop(loop: Xml.ForLoop): boolean {
   const indexVariable = getCanonicalIndexVariable(loop);
   if (!indexVariable) return false;

   return hasCanonicalCondition(loop, indexVariable)
      && hasCanonicalIncrement(loop, indexVariable)
      && hasCanonicalBody(loop, indexVariable);
}
/**
 * Returns true if the conditions expression has one of the following forms
 * (false otherwise):
 * * indexVar relational-op ub
 * * ub relational-op index indexVar
 *
 * ! Note that != is not currently supported and returns false
 */
function hasCanonicalCondition(loop: Xml.ForLoop, indexVariable: Xml.Element): boolean {
   if (loop.condition.childElements.length != 1) return false;

   const conditionExpression = loop.condition.child(0)!;

   // TODO: Allow for != casw wehre incr-expr == 1
   const operators = conditionExpression.find("./xmlns:operator");
   operators.filter((op: Xml.Element) => {
      return ["&lt;", "&gt;", "&lt;=", "&gt;="].includes(op.text);
   });
   if (operators.length != 1) return false;

   return (operators[0].prevElement?.equals(indexVariable)
      || operators[0].nextElement?.equals(indexVariable)) ?? false;
}
/**
 * Returns true if there is one and only one expression with the indexVariable
 * in the increment statement, and the increment must be of a standard form
 * like indexVar++ or indexVar = indexVar + step
 * @see https://www.openmp.org/spec-html/5.1/openmpsu45.html
 */
function hasCanonicalIncrement(loop: Xml.ForLoop, indexVariable: Xml.Element): boolean {
   if (loop.increment.childElements.length !== 1) return false;

   const expr = loop.increment.child(0)!;
   if (expr.contains("./xmlns:operator[text()='++' or text()='--']")
      && indexVariable.equals(expr.get("./xmlns:name")!)
      && expr.childElements.length === 2) {
         return true;
   } else if (expr.contains("./xmlns:operator[text()='=']")) {
      if (expr.childElements.length !== 5 || !indexVariable.equals(expr.child(0)!)) return false;

      if (expr.child(3)?.text === "+") {
         return indexVariable.equals(expr.child(2)!) || indexVariable.equals(expr.child(4)!);
      } else if (expr.child(3)?.text === "-") {
         return indexVariable.equals(expr.child(2)!);
      }
   } else if (expr.contains("./xmlns:operator[text()='+=' or text()='-=']")) {
      return expr.childElements.length === 3
         && indexVariable.equals(expr.child(0)!);
   }

   return false;
}
/**
 * Returns true if the indexVariable is not redfined within the loop body and
 * if there are no jump statements within the loop boyd
 * @param loop
 * @param indexVariable
 * @returns
 */
function hasCanonicalBody(loop: Xml.ForLoop, indexVariable: Xml.Element): boolean {
   const body = loop.body;

   const indexVariableInstances = body.find(`.//xmlns:name[text()='${indexVariable.text}']`);
   const isIVRedefined = indexVariableInstances.some((instance: Xml.Element) => {
      let prevCond: boolean = false;
      let nextCond: boolean = false;
      if (instance.prevElement) {
         prevCond = ["++", "--"].includes(instance.prevElement.text);
      }

      if (instance.nextElement) {
         nextCond = ["++", "--"].includes(instance.nextElement?.text)
            || [...instance.nextElement?.text].filter((char) => char === '=').length === 1;
      }
      return prevCond || nextCond;
   });

   return !isIVRedefined && !body.contains(".//xmlns:break")
      && !body.contains(".//xmlns:continue") && !body.contains(".//xmlns:return")
      && !body.contains(".//xmlns:label") && !body.contains(".//xmlns:goto");
}
