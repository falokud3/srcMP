// Data Dependence Test Framework
import * as Xml from '../Xml/Xml.js'

import { ArrayAccess } from './ArrayAccess.js';
import { DependenceVector, DependenceDir } from './DependenceVector.js';
import { SubscriptPair } from './SubscriptPair.js';

import { RangeTest } from './RangeTest.js';
import { BanerjeeTest } from './BanerjeeTest.js';
import * as ComplexMath from '../ComplexMath.js'
import { DataDependenceGraph } from './DataDependenceGraph.js';
import * as AliasAnalysis from './AliasAnalysis.js'


export function run(root: Xml.Element) : void {

   const dataDepGraph = new DataDependenceGraph();

   AliasAnalysis.run();

   getTestableLoops(root);

   return;

   // ExtractOutermostDependenceTestEligibleLoops


   // for each loop
      // analyze for dependences
      // add arcs to ddg

   // return DDG


   const forLoops = <Xml.Loop[]> root.find('//xmlns:for');

   // TODO: Handeling of nested loops
      // TODO: extracting only the outermost loops
   forLoops.forEach((forNode: Xml.Loop) => {
       analyzeLoopForDependence(forNode);
   });
}

function getTestableLoops(root: Xml.Element) : Xml.Loop[] {
   const loopElements = <Xml.Loop[]> root.find(".//xmlns:for[count(ancestor::xmlns:for)=0]")



   loopElements.forEach((loopNode) => {
      isLoopTestEligible(loopNode);
      // console.log();
      // console.log(loopNode.libraryXmlObject.toString());
   });

   // filter for eligibility

   return loopElements;
}

function isLoopTestEligible(loop: Xml.Loop) : boolean {

   if(!isCanonicalLoop(loop))


   // TODO : Allow parallelizable calls from standard library
   return isCanonicalLoop(loop) && !loop.contains(".//xmlns:call")
      && incrementValueIsDeterminable(loop);
}

/**
 * Returns true if the loop is in canonical form 
 * @see https://www.openmp.org/spec-html/5.1/openmpsu45.html
 */
function isCanonicalLoop(loop: Xml.Loop) : boolean {
   const indexVariable = getCanonicalInit(loop);
   if (!indexVariable) return false;

   return hasCanonicalCondition(loop, indexVariable) 
      && hasCanonicalIncrement(loop, indexVariable) 
      && hasCanonicalBody(loop, indexVariable);
}

/**
 * Returns the loop index variable if the initilization expression has one of 
 * the following forms (null otherwise):
 * * indexVar = lb
 * * integer-type indexVar = lb
 */
function getCanonicalInit(loop: Xml.Loop) : Xml.Element {
   const init = loop.initialization;
   // handles having no init and multiple init scenarios
   if (init.children.length != 1) return null;

   const initStatement = init.child(0);
   const variableLocation = initStatement.name === "decl" ? 1 : 0;
   const variable = initStatement.child(variableLocation);

   if (Xml.isComplexName(variable)) return null;

   // Disasllows Augmented Assignment & cases like i = j = 0
   if (initStatement.name === "expr" && (initStatement.child(1)?.text !== "=" 
       || initStatement.find("./xmlns:operator[contains(text(),'=')]").length !== 1)) return null;
   
   return variable;
}

/**
 * Returns true if the conditions expression has one of the following forms 
 * (false otherwise):
 * * indexVar relational-op ub
 * * ub relational-op index indexVar
 * 
 * ! Note that != is not currently supported and returns false
 */
function hasCanonicalCondition(loop: Xml.Loop, indexVariable: Xml.Element) : boolean {
   const conditionExpression = loop.condition.get("./xmlns:expr");
   if (!conditionExpression) return false;

   // TODO: Allow for != casw wehre incr-expr == 1
   const operators = conditionExpression.find("./xmlns:operator");
   operators.filter((op: Xml.Element) => {
      return ["&lt;","&gt;", "&lt;=", "&gt;="].includes(op.text);
   });
   if (operators.length != 1) return false;

   return operators[0].prevElement.equals(indexVariable)
      || operators[0].nextElement.equals(indexVariable);
}

/**
 * Returns true if there is one and only one expression with the indexVariable
 * in the increment statement, and the increment must be of a standard form
 * like indexVar++ or indexVar = indexVar + step
 * @see https://www.openmp.org/spec-html/5.1/openmpsu45.html 
 */
function hasCanonicalIncrement(loop: Xml.Loop, indexVariable: Xml.Element) : boolean {
   const incrementExpressions = loop.increment.find("./xmlns:expr");
   if (incrementExpressions.length !== 1) return false;

   const expr = incrementExpressions[0];
   if (expr.contains("./xmlns:operator[text()='++' or text()='--']")
       && indexVariable.equals(expr.get("./xmlns:name"))) { return true;
   } else if (expr.contains("./xmlns:operator[text()='=']")) {
      return expr.children.length === 5 
         && indexVariable.equals(expr.child(0))
         && (indexVariable.equals(expr.child(2)) || indexVariable.equals(expr.child(4)))
         && ["+", "-"].includes(expr.child(3).text);
   } else if (expr.contains("./xmlns:operator[text()='+=' or text()='-=']")) {
      return expr.children.length === 3
         && indexVariable.equals(expr.child(0));
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
function hasCanonicalBody(loop: Xml.Loop, indexVariable: Xml.Element) : boolean {
   const body = loop.body;

   const indexVariableInstances = body.find(`.//xmlns:name[text()='${indexVariable.text}']`);
   const isIVRefined = indexVariableInstances.some((instance: Xml.Element) => {
      return ["++", "--"].includes(instance.prevElement?.text) 
         || ["++", "--", "="].includes(instance.nextElement?.text);
   });

   return !isIVRefined && !body.contains(".//xmlns:break") 
   && !body.contains(".//xmlns:continue") && !body.contains(".//xmlns:return") 
   && !body.contains(".//xmlns:label") && !body.contains(".//xmlns:goto");
}

/**
 * Returns true if the loop increment can be resolved to an 
 * @param loop 
 */
function incrementValueIsDeterminable(loop: Xml.Loop) : boolean {
   return false;
}

// top level xml-parser call
// TODO
function analyzeLoopForDependence(loopNode: Xml.Loop) : void {
   // ? build/return DDG
   dataDependenceFramework(loopNode);

}

// runDDTest
// ? build/return DDG
function dataDependenceFramework(loopNode: Xml.Loop) : void {
   const array_access_map: Map<String, ArrayAccess[]> = 
      loopNode.getArrayAccesses();
   const innerLoopNest = loopNode.getInnerLoopNest();
   // TODO: INITIALIZE DDG
   for (const arrayName of array_access_map.keys()) {
      const array_accesses = array_access_map.get(arrayName);

      // TODO: ALIAS CHECKING

      for (let i = 0; i < array_accesses.length; i++) {
         const access_i = array_accesses.at(i);
         for (let j = 0; j < array_accesses.length; j++) {
            const access_j = array_accesses.at(j);

            if (access_i.getAccessType() === ArrayAccess.read_access &&
                access_j.getAccessType() === ArrayAccess.read_access) continue;

            // get common nest
            // NOTE: may not even need to get the 
            let relevantLoopNest = access_i.getEnclosingLoop().getCommonEnclosingLoopNest(access_j.getEnclosingLoop());
            // handle edge where loop being MT is not the outermost loop
            relevantLoopNest = Xml.Loop.getCommonLoops(relevantLoopNest, innerLoopNest);

            // TODO: Substitute Range Info

            const dvs: DependenceVector[] = [];
            let dependenceExists: boolean = testArrayAccessPair(access_i, access_j, 
               relevantLoopNest, dvs);
            
            if (dependenceExists) {
               const source_stmt = access_i.parentStatement.parent;
               const source_stmt_line = source_stmt.line
               const sink_stmt = access_j.parentStatement.parent;
               const prev_dependencies = sink_stmt.getAttribute("dependencies")
               if (prev_dependencies) {
                  if (!prev_dependencies.includes(String(source_stmt_line))) {
                     sink_stmt.setAttribute("dependencies", `${prev_dependencies}, ${String(source_stmt_line)}`);
                  }
               } else {
                  sink_stmt.setAttribute("dependencies", String(source_stmt_line))
               }
               loopNode.setAttribute("parallelizable", "false")
            }
         }
      }
   }

   console.log(loopNode.libraryXmlObject.toString());

}


//test AccessPair
// dvs is an OUT variable
function testArrayAccessPair(access: ArrayAccess, other_access: ArrayAccess,
   loopNest: Xml.Element[], dvs: DependenceVector[]) : boolean {
   let ret: boolean = false;
   // NOTE : THIS IS WHERE TO PICK DEPENDENCE TEST
   ret = testSubscriptBySubscript(access, other_access, loopNest, dvs);
   return ret;
}

// dvs is an OUT variable
function testSubscriptBySubscript(access: ArrayAccess, other_access: ArrayAccess,
   loopNest: Xml.Element[], dvs: DependenceVector[]) : boolean {
   if (access.getArrayDimensionality() == other_access.getArrayDimensionality()) {
      const pairs: SubscriptPair[] = [];
      const dimensions = access.getArrayDimensionality();
      // TODO: Change to foreach
      for (let dim = 1; dim <= dimensions; dim++) {
         const pair = new SubscriptPair(
            access.getDimension(dim),
            other_access.getDimension(dim),
            access,
            other_access,
            loopNest);
         pairs.push(pair);
      }
      
      const partitions: SubscriptPair[][] = partitionPairs(pairs);
      // TODO: sort partitions 

      // for a dependency to exist, all subscripts must have a depndency
      for (let i = 0; i < partitions.length; i++) {
         // if testPartition returns false then indepnence is proven
         if (!testPartition(partitions.at(i), dvs)) return false;
      }

   } else {
      // TODO: ALIAS NONSENSE
   }
   // may be dependence
   return true;
}

// getPartition
// based on partition psuedocode
// // could put ZIV first if wanted
function partitionPairs(pairs: SubscriptPair[]) : SubscriptPair[][] {
   const partitions: SubscriptPair[][] = [];
   pairs.forEach((pair: SubscriptPair) => {
      partitions.push([pair]);
   }) 

   const loopNest = pairs[0].getEnclosingLoops();
   loopNest.forEach((loopNode: Xml.Loop) => {
      const loop_indexVar = loopNode.getLoopIndexVariableName().text;
      let k: number;
      for (let i = 0; i < partitions.length; i++) {
         // check if parition has loop index variable
         const hasLoopIndex: boolean = partitions.at(i).some((pair: SubscriptPair) => {
            const sub1 = pair.getSubscript1();
            const sub2 = pair.getSubscript2();
            return sub1.containsName(loop_indexVar) ||
               sub2.containsName(loop_indexVar);
         });

         if (hasLoopIndex) {
            if (k === undefined) {
               k = i;
            } else {
               partitions[k] = partitions.at(k).concat(partitions.at(i));
               partitions.splice(i, 1); // deletes ith partition
               i--;
            }
         }
      }
   });
   return partitions;
}

// testPartition
function testPartition(parition: SubscriptPair[], dvs: DependenceVector[]) : boolean {
   let ret: boolean = false;
   // TODO: DV STUFF
   parition.forEach((pair: SubscriptPair) => {
      const complexity: number = pair.getComplexity();

      if (complexity === 0) {
         ret ||= testZIV(pair); // return false if independent
      } else if (complexity === 1) {
         ret ||= testSIV(pair);
      } else {
         ret ||= testMIV(pair);
      }

   });
   return ret;
}

function testZIV(pair: SubscriptPair) : boolean {
   const expr1 = pair.getSubscript1().child(1).text;
   const expr2 = pair.getSubscript2().child(1).text;

   const exprString = `${expr1} - (${expr2})`
   const expression = ComplexMath.simplify(exprString);

   if (!expression.isZero) {
      console.log("[testZIV] Could not determine independnece due to symoblic constants");
      return true;
   } 

   return expression.isZero;
}

function testSIV(pair: SubscriptPair) : boolean {
   console.log(pair);
   return true;
}

// test MIV
function testMIV(pair: SubscriptPair) : boolean {

   // const ddtest = new RangeTest(pair);
   const ddtest = new BanerjeeTest(pair);

   // TODO: Add *,*,* dependene vector

   if (!ddtest.pairIsElligible()) {
      return true;
   }

   const new_dvs: DependenceVector[] = testDependenceTree(ddtest)
   // TODO: Add new dependence vectors to total
   
   // no dependence vectors = no depedence
   return new_dvs.length != 0;
}
   // test tree

function testDependenceTree(ddtest: RangeTest | BanerjeeTest): DependenceVector[] {
   const dv_list: DependenceVector[] = []
   const dv: DependenceVector = new DependenceVector(ddtest.subscriptPair.getEnclosingLoops())

   if (ddtest.testDependence(dv)) testTree(ddtest, dv, 0, dv_list);

   return dv_list;
   
}

function testTree(ddtest: RangeTest | BanerjeeTest, dv: DependenceVector, pos: number, dv_list: DependenceVector[]) {
   let loopNest = ddtest.subscriptPair.getEnclosingLoops();
   let loop = loopNest[pos];
   for (let dir = DependenceDir.less; dir <= DependenceDir.greater; dir++) {

      dv.setDirection(loop, dir);

      if (!ddtest.testDependence(dv)) continue;

      if (!dv.containsDirection(DependenceDir.any) &&
         (!dv.isAllEqual() || ddtest.subscriptPair.isReachable())) {
         dv_list.push(dv.clone());
         console.log(`[Banerjee Test] Dependence from line ${ddtest.subscriptPair.getAccessLine(1)} to line ${ddtest.subscriptPair.getAccessLine(2)}`);
      }

      // recursive base case
      if (pos + 1 < loopNest.length) testTree(ddtest, dv, pos + 1, dv_list);

   }
   
   dv.setDirection(loop, DependenceDir.any); // ! may be unneeded

}

export {analyzeLoopForDependence}