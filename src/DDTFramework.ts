// Data Dependence Test Framework
import * as xml from 'libxmljs2'
import * as LoopTools from './util/LoopTools.js'
import * as XmlTools from './util/XmlTools.js'
import { ArrayAccess } from './ArrayAccess.js';
import { DependenceVector, DependenceDir } from './DependenceVector.js';
import { SubscriptPair } from './SubscriptPair.js';
import * as cortex from '@cortex-js/compute-engine';
import { RangeTest } from './RangeTest.js';


// top level xml-parser call
// TODO
function analyzeLoopForDependence(loopNode: xml.Element) : void {
   // ? build/return DDG
   dataDependenceFramework(loopNode);

}

// runDDTest
// ? build/return DDG
function dataDependenceFramework(loopNode: xml.Element) : void {
   const array_access_map: Map<String, ArrayAccess[]> = 
      LoopTools.getArrayAccesses(loopNode);
   const innerLoopNest = LoopTools.getInnerLoopNest(loopNode);
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
            let relevantLoopNest: xml.Element[] = LoopTools.getCommonEnclosingLoopNest(
               access_i.getEnclosingLoop(), access_j.getEnclosingLoop());
            // handle edge where loop being MT is not the outermost loop
            relevantLoopNest = LoopTools.getCommonLoops(relevantLoopNest, innerLoopNest);

            // TODO: Substitute Range Info

            const dvs: DependenceVector[] = [];
            let dependenceExists: boolean = testArrayAccessPair(access_i, access_j, 
               relevantLoopNest, dvs);
         }
      }
   }
}


//test AccessPair
// dvs is an OUT variable
function testArrayAccessPair(access: ArrayAccess, other_access: ArrayAccess,
   loopNest: xml.Element[], dvs: DependenceVector[]) : boolean {
   let ret: boolean = false;
   // NOTE : THIS IS WHERE TO PICK DEPENDENCE TEST
   ret = testSubscriptBySubscript(access, other_access, loopNest, dvs);
   return ret;
}

// dvs is an OUT variable
function testSubscriptBySubscript(access: ArrayAccess, other_access: ArrayAccess,
   loopNest: xml.Element[], dvs: DependenceVector[]) : boolean {
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
         if (!testPartition(partitions.at(i), dvs)) return false;
      }

   } else {
      // TODO: ALIAS NONSENSE
   }
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
   loopNest.forEach((loopNode: xml.Element) => {
      const loop_indexVar = LoopTools.getLoopIndexVariable(loopNode).text();
      let k: number;
      for (let i = 0; i < partitions.length; i++) {
         // check if parition has loop index variable
         const hasLoopIndex: boolean = partitions.at(i).some((pair: SubscriptPair) => {
            const sub1 = pair.getSubscript1();
            const sub2 = pair.getSubscript2();
            return XmlTools.containsName(sub1, loop_indexVar) ||
               XmlTools.containsName(sub2, loop_indexVar);
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
         ret ||= testZIV(pair);
      } else if (complexity === 1) {
         ret ||= testSIV(pair);
      } else {
         ret ||= testMIV(pair);
      }

   });
   return ret;
}
// test ZIV
function testZIV(pair: SubscriptPair) : boolean {
   const expr1 = (pair.getSubscript1().child(1) as xml.Element).text();
   const expr2 = (pair.getSubscript2().child(1) as xml.Element).text();

   const subtraction = `${expr1} - (${expr2})`
   
   const ce = new cortex.ComputeEngine();

   const simp_sub = ce.parse(subtraction).simplify();


   // NOTE: if .isZERO is undefined then the cortexEngine couldn't determine
   // * if the value was zero due to variables

   if (simp_sub.isZero === undefined) {
      // can be more specific with range analysis
      return true; //conserative
   } 

   return simp_sub.isZero;
}
// test SIV
function testSIV(pair: SubscriptPair) : boolean {

   return true;
}

// test MIV
function testMIV(pair: SubscriptPair) : boolean {

   const ddtest = new RangeTest(pair);

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

function testDependenceTree(ddtest: RangeTest): DependenceVector[] {
   const dv_list: DependenceVector[] = []
   const dv: DependenceVector = new DependenceVector(ddtest.subscriptPair.getEnclosingLoops())

   if (ddtest.testDependence(dv)) testTree(ddtest, dv, 0, dv_list);

   return dv_list;
   
}

function testTree(ddtest: RangeTest, dv: DependenceVector, pos: number, dv_list: DependenceVector[]) {
   let loopNest = ddtest.subscriptPair.getEnclosingLoops();
   let loop = loopNest[pos];
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

export {analyzeLoopForDependence}