// Data Dependence Test Framework
import * as xml from 'libxmljs2'
import * as LoopTools from './LoopTools.js'
import * as XmlTools from './XmlTools.js'
import { ArrayAccess } from './ArrayAccess.js';
import { DependenceVector } from './DependenceVector.js';
import { SubscriptPair } from './SubscriptPair.js';

// top level xml-parser call
// TODO
function analyzeLoopForDependence(loopNode: xml.Element) : void {
   console.log((loopNode.get('xmlns:control', XmlTools.ns) as xml.Element).text() + "\n");
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
            if (dependenceExists) {
               // add DV to DDG
            }
         }
      }
   }
}


//test AccessPair
// dvs is an OUT variable
function testArrayAccessPair(access: ArrayAccess, other_access: ArrayAccess,
   loopNest: xml.Element[], dvs: DependenceVector[]) : boolean {
   // TODO: remove
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
      console.log(access.toString());
      console.log(other_access.toString());
      partitions.forEach((partition) => {
         console.log('-Partition-');
         partition.forEach((pair) => {
            console.log(pair.toString());
         })
         console.log('-----------');
      });
      console.log();

   } else {
      // TODO: ALIAS NONSENSE
   }
   return true;
}

// getPartition
// based on partition psuedocode
function partitionPairs(pairs: SubscriptPair[]) : SubscriptPair[][] {
   const partitions: SubscriptPair[][] = [];
   pairs.forEach((pair: SubscriptPair) => {
      partitions.push([pair]);
   }) 

   const loopNest = pairs[0].getEnclosingLoops();
   loopNest.forEach((loopNode: xml.Element) => {
      const loop_indexVar = LoopTools.getLoopIndexVariable(loopNode);
      let k: number;
      for (let i = 0; i < partitions.length; i++) {
         // check if parition has loop index variable
         const hasLoopIndex: boolean = partitions.at(i).some((pair: SubscriptPair) => {
            const sub1 = pair.getSubscript1();
            const sub2 = pair.getSubscript2();
            return XmlTools.contains(sub1,
               `.//xmlns:name[text()='${loop_indexVar.text()}']`, XmlTools.ns) 
               || XmlTools.contains(sub2,
               `.//xmlns:name[text()='${loop_indexVar.text()}']`, XmlTools.ns);
         });

         if (hasLoopIndex) {
            if (k === undefined) {
               k = i;
            } else {
               partitions[k] = partitions.at(k).concat(partitions.at(i));
               partitions.splice(i, 1); // remove ith partition
               i--;
            }
         }
      }
   });
   console.log("partitions", partitions.length);
   return partitions;
}

// testPartition

// test ZIV
// test SIV
// test MIV
   // test tree

export {analyzeLoopForDependence}