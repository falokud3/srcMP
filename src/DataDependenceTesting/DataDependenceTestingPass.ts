// Data Dependence Test Framework
import * as Xml from '../Xml/Xml.js'

import { ArrayAccess } from './ArrayAccess.js';
import { DependenceVector, DependenceDir, mergeVectorSets } from './DependenceVector.js';
import { SubscriptPair } from './SubscriptPair.js';

import { RangeTest } from './RangeTest.js';
import { BanerjeeTest } from './BanerjeeTest.js';
import { Arc, DataDependenceGraph as DDGraph } from './DataDependenceGraph.js';
import * as AliasAnalysis from './AliasAnalysis.js'
import * as RangeAnalysis from './RangeAnalysis.js'
import { extractOutermostDependenceTestEligibleLoops } from './Eligibility.js';

import * as ComputerAlgebraSystem from '../ComputerAlgebraSystem.js'
import { Verbosity } from '../CommandLineOutput.js';
import * as CLO from '../CommandLineOutput.js'


export function run(program: Xml.Element) : DDGraph {
   const startTime = performance.now();
   CLO.output({format: (verbosity: Verbosity) => {
      if (verbosity !== Verbosity.Internal) return '';
      return '[Data Dependence Pass] Start'
   }});

   const ddg = new DDGraph();

   // AliasAnalysis.run();

   const [loops, messages] = extractOutermostDependenceTestEligibleLoops(program);

   CLO.output(...messages);
   
   loops.forEach((loop) => {
      ddg.addAllArcs(analyzeLoopForDependence(loop));
   });

   const endTime = performance.now();
   CLO.output({format: (verbosity: Verbosity) => {
      if (verbosity !== Verbosity.Internal) return '';
      return `[Data Dependence Pass] End -- Duration: ${(endTime - startTime).toFixed(3)}ms`
   }});

   return ddg;
}

// runDDTest
// ? build/return DDG
function analyzeLoopForDependence(loopNode: Xml.ForLoop) : DDGraph {
   const loopDDG = new DDGraph();

   const array_access_map: Map<String, ArrayAccess[]> = 
      loopNode.getArrayAccesses();
   const innerLoopNest = loopNode.getInnerLoopNest();
   let pairDepVectors: DependenceVector[];

   for (const arrayName of array_access_map.keys()) {
      const array_accesses = array_access_map.get(arrayName)!;

      // TODO: ALIAS CHECKING

      for (let i = 0; i < array_accesses.length; i++) {
         const access_i = array_accesses.at(i)!;
         for (let j = 0; j < array_accesses.length; j++) {
            const access_j = array_accesses.at(j)!;

            if (access_i.getAccessType() === ArrayAccess.READ_ACCESS &&
                access_j.getAccessType() === ArrayAccess.READ_ACCESS) continue;

            let relevantLoopNest = access_i.enclosingLoop!
               .getCommonEnclosingLoopNest(access_j.enclosingLoop!);
            // // handles edge case where loop being analyzed is not the outermost loop
            relevantLoopNest = Xml.ForLoop.getCommonLoops(relevantLoopNest, innerLoopNest);

            // TODO: Substitute Range Info

            pairDepVectors = [];
            let dependenceExists: boolean = testAccessPair(access_i, access_j, 
               relevantLoopNest, pairDepVectors);
            
            if (dependenceExists) {
               for (const dv of pairDepVectors) {
                  loopDDG.addArc(new Arc(access_i, access_j, dv));
               }
            }
         }
      }
   }

   return loopDDG;
}


//test AccessPair
// dvs is an OUT variable
function testAccessPair(access: ArrayAccess, other_access: ArrayAccess,
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

      const language = loopNest[0].get("/xmlns:unit")?.getAttribute("language") ?? "C++"

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
      // TODO: sort partitions 

      // for a dependency to exist, all subscripts must have a depndency
      for (let i = 0; i < partitions.length; i++) {
         // if testPartition returns false then indepnence is proven
         if (!testPartition(partitions.at(i)!, dvs)) return false;
      }

   } else {
      // TODO: ALIAS NONSENSE
   }
   // may be dependence
   return true;
}

// getPartition
// based on partition psuedocode
// TODO: put ZIV first if wanted
function partitionPairs(pairs: SubscriptPair[]) : SubscriptPair[][] {
   const partitions: SubscriptPair[][] = [];
   pairs.forEach((pair: SubscriptPair) => {
      partitions.push([pair]);
   }) 

   const loopNest = pairs[0].getEnclosingLoops() as Xml.ForLoop[];
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
            if (!k) {
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

// testPartition
function testPartition(parition: SubscriptPair[], partitoinDepVectors: DependenceVector[]) : boolean {
   let ret: boolean = false;
   const testDepVectors: DependenceVector[] = [];
   parition.forEach((pair: SubscriptPair) => {
      const complexity: number = pair.getComplexity();
      const pairDepVectors: DependenceVector[] = [];

      if (complexity === 0) {
         ret ||= testZIV(pair, pairDepVectors); // return false if independent
      // } else if (complexity === 1) {
         // ret ||= testSIV(pair, pairDependenceVectors);
      } else {
         ret ||= testMIV(pair, pairDepVectors);
      }
      mergeVectorSets(testDepVectors, pairDepVectors);
   });

   if (ret) mergeVectorSets(partitoinDepVectors, testDepVectors);

   return ret;
}

function testZIV(pair: SubscriptPair, pairDependenceVectors: DependenceVector[]) : boolean {
   const expr1 = pair.getSubscript1().get('xmlns:expr')?.text // pair.getSubscript1()?.child(1)?.text;
   const expr2 = pair.getSubscript2().get('xmlns:expr')?.text

   if (!expr1 || !expr2) {
      pairDependenceVectors.push(new DependenceVector(pair.getEnclosingLoops()));
      return true;
   }

   const exprString = `(${expr1} - (${expr2})) == 0`
   const expression = ComputerAlgebraSystem.simplify(exprString);

   const result = Number(expression);

   if (Number.isNaN(result) || result === ComputerAlgebraSystem.TRUE) {
      pairDependenceVectors.push(new DependenceVector(pair.getEnclosingLoops()));
      return true;
   } 
   return false; 
}

function testSIV(pair: SubscriptPair, pairDependenceVectors: DependenceVector[]) : boolean {
   throw new Error("Not Yet Implemented")
}

// test MIV
function testMIV(pair: SubscriptPair, pairDependenceVectors: DependenceVector[]) : boolean {

   // const ddtest = new RangeTest(pair);
   const ddtest = new BanerjeeTest(pair);

   if (!ddtest.pairIsElligible()) {
      pairDependenceVectors.push(new DependenceVector(pair.getEnclosingLoops()));
      return true;
   }

   const new_dvs: DependenceVector[] = testDependenceTree(ddtest)
   pairDependenceVectors.push(...new_dvs);
   
   // no dependence vectors = no depedence
   return new_dvs.length !== 0;
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
         // console.log(`[Banerjee Test] Dependence from line ${ddtest.subscriptPair.getAccessLine(1)} to line ${ddtest.subscriptPair.getAccessLine(2)}`);
      }

      // recursive base case
      if (pos + 1 < loopNest.length) testTree(ddtest, dv, pos + 1, dv_list);

   }
   
   dv.setDirection(loop, DependenceDir.any); // ! may be unneeded

}