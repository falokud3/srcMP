import { CLIMessage, Verbosity, numFormat, errorStart, examples } from '../../common/CommandLineOutput.js';
import { getCanonicalIndexVariable } from '../../common/Xml/ForLoop.js';
import { getCanonicalIncrementValue } from '../../common/Xml/ForLoop.js';
import * as Xml from '../../common/Xml/Xml.js';

import * as CLO from '../../common/CommandLineOutput.js'
import chalk from 'chalk';

// TODO: Test
export function extractOutermostDependenceTestEligibleLoops(root: Xml.Element): [Xml.ForLoop[], EligiblityMessage[]] {
   const outerLoops = <Xml.ForLoop[]>root.find("descendant-or-self::xmlns:for[count(ancestor::xmlns:for)=0]");

   const ret: Xml.ForLoop[] = [];

   let curr: Xml.ForLoop | undefined;
   const messages: EligiblityMessage[] = [];
   while (curr = outerLoops.at(0)) {
      const nestedLoops = curr.find('descendant::xmlns:for') as Xml.ForLoop[];
      let loopIsEligibile: boolean, loopMessage: EligiblityMessage;
      [loopIsEligibile, loopMessage] = isLoopTestEligible(curr);
      for (const nestedLoop of nestedLoops) {
         const nestedLoopIsEligible = isLoopTestEligible(nestedLoop)[0];
         if (!nestedLoopIsEligible) {
            loopMessage.nestedInelligibleLoop = nestedLoop;
            loopIsEligibile = false;
            break;
         }
      }

      messages.push(loopMessage);
      outerLoops.shift();
      if (loopIsEligibile) {
         ret.push(curr);
      } else {
         const nestedLoop = nestedLoops.at(0);
         if (nestedLoop) outerLoops.unshift(nestedLoop);
      }
   }
   
   return [ret, messages];
}
function isLoopTestEligible(loop: Xml.ForLoop): [boolean, EligiblityMessage] {
   const message = new EligiblityMessage(loop);
   
   const isCanonical = isCanonicalLoop(loop, message);
   // TODO : Allow parallelizable calls from standard library
   const containsMethodCall = loop.contains(".//xmlns:call");
   const incrementValue = getCanonicalIncrementValue(loop);

   message.containsMethodCall = containsMethodCall;
   message.incrementValue = incrementValue;

   const eligible = isCanonical && !containsMethodCall
      && typeof incrementValue === 'number';

   return [eligible, message];
}
/**
 * Returns true if the loop is in canonical form
 * Optional eligibilitymessage to store info on eligiblity
 * @see https://www.openmp.org/spec-html/5.1/openmpsu45.html
 */
function isCanonicalLoop(loop: Xml.ForLoop, message?: EligiblityMessage): boolean {
   const indexVariable = getCanonicalIndexVariable(loop);
   if (!indexVariable) return false;

   const validCondition = hasCanonicalCondition(loop, indexVariable);
   const validIncrement = hasCanonicalIncrement(loop, indexVariable);
   const validBody = hasCanonicalBody(loop, indexVariable);

   if (message) {
      message.indexVariable = indexVariable;
      message.hasCanonicalCondition = validCondition;
      message.hasCanonicalBody = validBody;
   }

   return validCondition && validIncrement && validBody;
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

   return Boolean(operators[0].prevElement?.text === indexVariable.text
      || operators[0].nextElement?.text === indexVariable.text);
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
      && indexVariable.text === expr.get("./xmlns:name")?.text
      && expr.childElements.length === 2) {
         return true;
   } else if (expr.contains("./xmlns:operator[text()='=']")) {
      if (expr.childElements.length !== 5 || indexVariable.text !== expr.child(0)?.text) return false;

      if (expr.child(3)?.text === "+") {
         return indexVariable.text === expr.child(2)?.text 
            || indexVariable.text === expr.child(4)?.text;
      } else if (expr.child(3)?.text === "-") {
         return indexVariable.text === expr.child(2)?.text;
      }
   } else if (expr.contains("./xmlns:operator[text()='+=' or text()='-=']")) {
      return expr.childElements.length === 3
         && indexVariable.text === expr.child(0)?.text;
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

export class EligiblityMessage implements CLIMessage {

   loop: Xml.ForLoop;

   indexVariable: Xml.Element | null = null;
   incrementValue: string | number = "NULL";

   hasCanonicalCondition: boolean = false;

   hasCanonicalBody: boolean = false;

   containsMethodCall: boolean = true;

   nestedInelligibleLoop: Xml.ForLoop | null = null;

   constructor(loop: Xml.ForLoop) {
      this.loop = loop;
   }

   get eligible() : boolean {
      return this.hasCanonicalBody 
         && this.hasCanonicalCondition 
         && typeof this.incrementValue === 'number'
         && !this.containsMethodCall
         && !this.nestedInelligibleLoop;
   }

   public get indexVariableRedefinitions() : Xml.Element[] {
      return this.loop.body
         .find(`.//xmlns:name[text()='${this.indexVariable?.text ?? ""}']`)
         .filter((instance: Xml.Element) => {
         let prevCond: boolean = false;
         let nextCond: boolean = false;
         if (instance.prevElement) {
            prevCond = ["++", "--"].includes(instance.prevElement.text);
         }

         if (instance.nextElement) {
            nextCond = ["++", "--"].includes(instance.nextElement?.text)
               || Xml.isAssignmentOperator(instance.nextElement) 
               || Xml.isAugAssignmentOperator(instance.nextElement);
         }
         return prevCond || nextCond;
      });
   }
   
   public get jumpStatements() : Xml.Element[] {
      return this.loop.find('.//xmlns:break')
         .concat(this.loop.find('.//xmlns:continue'))
         .concat(this.loop.find('.//xmlns:return'))
         .concat(this.loop.find('.//xmlns:goto'));
   }

   public get simpleFormat() : string {
      let output = `${this.loop.line}:${this.loop.col}| for${this.loop.header.text} ${chalk.red('Test Inelligible')}: `;
      if (!this.indexVariable) {
         output += 'Could not determine the loop\'s index variable.';
      } else if (typeof this.incrementValue !== 'number') {
         output += 'Could not determine the value of the loop\'s increment.';
      } else if (!this.hasCanonicalCondition) {
         output += 'The loop condition is not in canonical form.';
      } else if (this.indexVariableRedefinitions.length !== 0) {
         output += `The loop body redefines the index variable ${numFormat(this.indexVariableRedefinitions.length, 'time')}.`;
      } else if (this.jumpStatements.length !== 0) {
         output += `The loop body has ${numFormat(this.jumpStatements.length, 'jump statement')}.`;
      } else if (this.loop.find('.//xmlns:label').length !== 0) {
         output += `The loop body has ${numFormat(this.loop.find('.//xmlns:label').length, 'label statement')}.`
      } else if (this.containsMethodCall) {
         output += `The loop body has ${numFormat(this.loop.find('.//xmlns:call').length,'unparallelizable method call')}.`;
      } else {
         if (this.nestedInelligibleLoop) {
            output += `A loop nested within this loop at line ${this.nestedInelligibleLoop.line}, col ${this.nestedInelligibleLoop.col} is ineligible.`;
         } else {
            output = '';
         }
      }
      return output;
   }

   public buildComplexBody() : string {

      let body = `The for loop ${this.loop.header.text} at line ${this.loop.line}, column ${this.loop.col} is not eligible for data dependence testing\n\n`;
      let issues = 0;
      if (!this.indexVariable) {
         body += `${errorStart(issues++)} the loop\'s index variable could not be determined.\n`;
      }

      if (this.indexVariable && typeof this.incrementValue !== 'number') {
         body += `${errorStart(issues++)} value of the loop's increment could not be determined from the expression \u201C${this.loop.increment.text}\u201D.\n`
      }

      if (this.indexVariable && !this.hasCanonicalCondition) {
         body += `${errorStart(issues++)} loop's condition \u201C${this.loop.condition.text}\u201D is not in canonical from.\n`;
      }
      
      const ivDefs = this.indexVariableRedefinitions.map((redef: Xml.Element) => redef.parentElement ?? redef);
      if (this.indexVariable && ivDefs.length !== 0) {
         body += `${errorStart(issues++)} loop redefines the index variable ${numFormat(ivDefs.length, 'time')}. For example:\n`
         body += examples(ivDefs) + '\n';
      }

      const jumps = this.jumpStatements;
      if (jumps.length !== 0) {
         body += `${errorStart(issues++)} loop body has ${numFormat(jumps.length, 'jump statement')}. For example:\n`;
         body += examples(jumps) + '\n';
      }

      const labels = this.loop.find('.//xmlns:label');
      if (labels.length !== 0) {
         body += `${errorStart(issues++)} loop body has ${numFormat(labels.length, 'label statements')}. For example:\n`
         body += examples(labels) + '\n';
      }

      const calls = this.loop.find('.//xmlns:call');
      if (calls.length > 0) {
         body += `${errorStart(issues++)} loop body has ${numFormat(calls.length,'unparallelizable method call')}.\n`;
         body += examples(calls) + '\n';
      }

      if (this.nestedInelligibleLoop) {
         body += `There's a loop nested within this loop at line ${this.nestedInelligibleLoop.line}, col ${this.nestedInelligibleLoop.col} that is ineligible.\n`;
         body += `\t ${this.nestedInelligibleLoop.line}:${this.nestedInelligibleLoop.col}| ${this.nestedInelligibleLoop.header.text}\n`;
      }
      return body.substring(0, body.length - 1); // eliminates trailing \n
   }

   public get complexFormat() : string {
      if (this.eligible) return '';

      const filename = this.loop.get('/xmlns:unit')?.getAttribute('filename') ?? '';
      
      const paddingLength = 80 - (25 + 1 + filename.length);
      const padding = '-'.repeat(paddingLength > 0 ? paddingLength : 0);

      const header = chalk.red(`-- UNTESTABLE FOR LOOP --${padding} ${filename}`);
      const body = this.buildComplexBody();
      const footer = chalk.red(`-`.repeat(80));

      return `${header}\n${body}\n${footer}`;
   }

   public get internalFormat() : string {
      if (!this.eligible) return this.complexFormat;
      const filename = this.loop.get('/xmlns:unit')?.getAttribute('filename') ?? '';
      
      const paddingLength = 80 - (23 + 1 + filename.length);
      const padding = '+'.repeat(paddingLength > 0 ? paddingLength : 0);

      const header = chalk.green(`++ TESTABLE FOR LOOP ++${padding} ${filename}`);
      let body = `The for loop ${this.loop.header.text} is elligible for data dependence testing.\n`;
      body += `Initialization: ${this.indexVariable!.parentElement!.text}\n`
      body += `Condition: ${this.loop.condition.text}\n`
      body += `Increment Value: ${this.incrementValue}`
      const footer = chalk.green(`${'+'.repeat(80)}`);

      return `${header}
${body}
${footer}`;
   }

   format(verbosity: Verbosity) : string {
      if (verbosity === Verbosity.Simple) {
         return this.simpleFormat;
      } else if (verbosity === Verbosity.Complex) {
         return this.complexFormat;
      } else if (verbosity === Verbosity.Internal) {
         return this.internalFormat;
      } else {
         return '';
      }
   }

   
}
