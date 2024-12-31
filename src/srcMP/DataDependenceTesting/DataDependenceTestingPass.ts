// Data Dependence Test Framework
import * as Xml from '../../common/Xml/Xml.js';
import * as ComputerAlgebraSystem from '../../common/ComputerAlgebraSystem.js';

import { ArrayAccess } from './ArrayAccess.js';
import { DependenceVector, DependenceDir, mergeVectorSets } from './DependenceVector.js';
import { SubscriptPair } from './SubscriptPair.js';
import { BanerjeeTest } from './BanerjeeTest.js';
import { Arc, DataDependenceGraph as DDGraph } from './DataDependenceGraph.js';
import { extractOutermostDependenceTestEligibleLoops } from './Eligibility.js';
import { Verbosity, log, output } from '../../common/CommandLineOutput.js';

export function run(program: Xml.Element) : DDGraph {
   const startTime = performance.now();
   log('[Data Dependence Pass] Start', Verbosity.Internal);
   const ddg = new DDGraph();

   const [loops, messages] = extractOutermostDependenceTestEligibleLoops(program);

   output(...messages);
   
   loops.forEach((loop) => {
      log(`\nTesting ${loop.line}:${loop.col}|${loop.header.text}`, Verbosity.Internal);
      ddg.addAllArcs(analyzeLoopForDependence(loop));
   });

   log(`\nData Dependence Graph:\n${ddg.toString()}`, Verbosity.Internal);

   const endTime = performance.now();
   log(`[Data Dependence Pass] End -- Duration: ${(endTime - startTime).toFixed(3)}ms`, Verbosity.Internal);
   return ddg;
}

function analyzeLoopForDependence(loopNode: Xml.ForLoop) : DDGraph {
   const loopDDG = new DDGraph();

   const array_access_map: Map<string, ArrayAccess[]> = 
      loopNode.getArrayAccesses();
   const innerLoopNest = loopNode.getInnerLoopNest();
   let pairDepVectors: DependenceVector[];

   for (const arrayName of array_access_map.keys()) {
      log(`\nTesting array ${arrayName}`, Verbosity.Internal);
      const array_accesses = array_access_map.get(arrayName)!;

      for (let i = 0; i < array_accesses.length; i++) {
         const accessI = array_accesses.at(i)!;
         for (let j = 0; j < array_accesses.length; j++) {
            const accessJ = array_accesses.at(j)!;

            if (accessI.getAccessType() === ArrayAccess.READ_ACCESS &&
                accessJ.getAccessType() === ArrayAccess.READ_ACCESS) continue;

            let relevantLoopNest = accessI.enclosingLoop!
               .getCommonEnclosingLoopNest(accessJ.enclosingLoop!);
            // // handles edge case where loop being analyzed is not the outermost loop
            relevantLoopNest = Xml.ForLoop.getCommonLoops(relevantLoopNest, innerLoopNest);

            // TODO: Substitute Range Info

            pairDepVectors = [];
            const dependenceExists: boolean = testAccessPair(accessI, accessJ, 
               relevantLoopNest, pairDepVectors);
            
            if (dependenceExists) {
               for (const dv of pairDepVectors) {
                  loopDDG.addArc(new Arc(accessI, accessJ, dv));
               }
            }
         }
      }
   }

   return loopDDG;
}

/**
 * Used in Cetus to pick the dependence test. Because srcMP currently only supports
 * the Range Test, this method does not provide much value
 * @returns true for potential dependence, false for confirmed independence
 */
function testAccessPair(access: ArrayAccess, other_access: ArrayAccess,
   loopNest: Xml.Element[], dvs: DependenceVector[] /*out*/) : boolean {
   log(`Testing Access Pair: ${access.access.line}:${access.access.col}|${access.toString()} ${other_access.access.line}:${other_access.access.col}|${other_access.toString()}`, Verbosity.Internal);
   return testSubscriptBySubscript(access, other_access, loopNest, dvs);
}

/**
 * 
 * @param access 
 * @param other_access 
 * @param loopNest 
 * @param dvs out variable, will store the dependence vectors of the pair if there are any
 * @returns true for potential dependence, false for confirmed independence
 */
function testSubscriptBySubscript(access: ArrayAccess, other_access: ArrayAccess,
   loopNest: Xml.Element[], dvs: DependenceVector[] /*out*/) : boolean {
   const pairs: SubscriptPair[] = [];
   const dimensions = access.getArrayDimensionality();

   for (let dim = 1; dim <= dimensions; dim++) {
      // TODO: RANGE SUBSTITUTION
      const pair = new SubscriptPair(
         access.getDimension(dim)!,
         other_access.getDimension(dim)!,
         access,
         other_access,
         (loopNest as Xml.ForLoop[]));
      pairs.push(pair);
   }
   
   const partitions: SubscriptPair[][] = partitionPairs(pairs);
   const str: string[] = [];
   partitions.forEach((part) => {str.push(`[${part.join(', ')}]`);});
   log(`SubscriptPair Partitions: [${str.join(', ')}]`, Verbosity.Internal);
   
   // for a dependency to exist, all subscripts must have a depndency
   for (let i = 0; i < partitions.length; i++) {
      // if testPartition returns false then indepnence is proven
      if (!testPartition(partitions.at(i)!, dvs)) return false;
   }

   return true;
}

function partitionPairs(pairs: SubscriptPair[]) : SubscriptPair[][] {
   const partitions: SubscriptPair[][] = [];
   pairs.forEach((pair: SubscriptPair) => {
      // places ZIV partitions first, because their dependence tests are simpler
      // should later be replaced, by sorting the partions by complexity at the end of the function
      // if SIV testing is ever implemented
      if (pair.getComplexity() === 0 ) {
         partitions.unshift([pair]);
      } else {
         partitions.push([pair]);
      }
   }); 

   const loopNest = pairs[0].getEnclosingLoops();
   loopNest.forEach((loopNode: Xml.ForLoop) => {
      const loop_indexVar = loopNode.getLoopIndexVariableName()!.text;
      let k: number | undefined;
      for (let i = 0; i < partitions.length; i++) {
         // check if parition has loop index variable
         const hasLoopIndex: boolean = partitions.at(i)!.some((pair: SubscriptPair) => {
            const sub1 = pair.getSubscript1();
            const sub2 = pair.getSubscript2();
            return sub1.containsName(loop_indexVar) ||
               sub2.containsName(loop_indexVar);
         });

         if (hasLoopIndex) {
            if (k === undefined) {
               k = i;
            } else {
               partitions[k] = partitions.at(k)!.concat(partitions.at(i)!);
               partitions.splice(i, 1); // deletes ith partition
               i--;
            }
         }
      }
   });
   return partitions;
}

function testPartition(parition: SubscriptPair[], partitionDepVectors: DependenceVector[] /*out*/) : boolean {
   let ret: boolean = false;
   const testDepVectors: DependenceVector[] = [];
   parition.forEach((pair: SubscriptPair) => {
      const complexity: number = pair.getComplexity();
      const pairDepVectors: DependenceVector[] = [];

      if (complexity === 0) {
         ret ||= testZIV(pair, pairDepVectors);
      // } else if (complexity === 1) {
         // ret ||= testSIV(pair, pairDepVectors);
      } else {
         ret ||= testMIV(pair, pairDepVectors);
      }
      mergeVectorSets(testDepVectors, pairDepVectors);
   });

   if (ret) mergeVectorSets(partitionDepVectors, testDepVectors);

   return ret;
}

function testZIV(pair: SubscriptPair, pairDependenceVectors: DependenceVector[] /*out*/) : boolean {
   const expr1 = pair.getSubscript1().get('xmlns:expr')?.text;
   const expr2 = pair.getSubscript2().get('xmlns:expr')?.text;

   if (!expr1 || !expr2) {
      pairDependenceVectors.push(new DependenceVector(pair.getEnclosingLoops()));
      log(`ZIV Test: Dependent for ${pair.toString()}`, Verbosity.Internal);
      return true;
   }

   const exprString = `(${expr1} - (${expr2})) == 0`;
   const expression = ComputerAlgebraSystem.simplify(exprString);

   const result = Number(expression);

   if (Number.isNaN(result) || result === ComputerAlgebraSystem.TRUE) {
      pairDependenceVectors.push(new DependenceVector(pair.getEnclosingLoops()));
      log(`ZIV Test: Dependent for ${pair.toString()}`, Verbosity.Internal);
      return true;
   } 
   log(`ZIV Test: Independent for ${pair.toString()}`, Verbosity.Internal);
   return false; 
}

function testMIV(pair: SubscriptPair, pairDependenceVectors: DependenceVector[] /*out*/) : boolean {
   const ddtest = new BanerjeeTest(pair);

   if (!ddtest.pairIsElligible()) {
      pairDependenceVectors.push(new DependenceVector(pair.getEnclosingLoops()));
      log(`MIV Test: Dependent for ${pair.toString()}`, Verbosity.Internal);
      return true;
   }

   const new_dvs: DependenceVector[] = testDependenceTree(ddtest);
   pairDependenceVectors.push(...new_dvs);   
   log(`MIV Test: ${new_dvs.length !== 0 ? 'D' : 'Ind'}ependent for ${pair.toString()}`, Verbosity.Internal);
   return new_dvs.length !== 0;
}

function testDependenceTree(ddtest: BanerjeeTest): DependenceVector[] {
   const dv_list: DependenceVector[] = [];
   const dv: DependenceVector = new DependenceVector(ddtest.subscriptPair.getEnclosingLoops());

   if (ddtest.testDependence(dv)) testTree(ddtest, dv, 0, dv_list);

   return dv_list;
   
}

function testTree(ddtest: BanerjeeTest, dv: DependenceVector, pos: number, dv_list: DependenceVector[]) {
   const loopNest = ddtest.subscriptPair.getEnclosingLoops();
   const loop = loopNest[pos];
   for (let dir = DependenceDir.less; dir <= DependenceDir.greater; dir++) {

      dv.setDirection(loop, dir);

      if (!ddtest.testDependence(dv)) continue;

      if (!dv.containsDirection(DependenceDir.any) &&
         (!dv.isAllEqual() || ddtest.subscriptPair.isReachable())) {
         dv_list.push(dv.clone());
      }

      // recursive base case
      if (pos + 1 < loopNest.length) testTree(ddtest, dv, pos + 1, dv_list);

   }
   
   dv.setDirection(loop, DependenceDir.any); // ! may be unneeded

}
