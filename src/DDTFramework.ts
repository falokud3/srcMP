// Data Dependence Test Framework
import * as xml from 'libxmljs2'
import * as LoopTools from './LoopTools.js'
import { ArrayAccess } from './ArrayAccess.js';
import { DependenceVector } from './DependenceVector.js';

// top level xml-parser call
// TODO
function analyzeLoopForDependence(loopNode: xml.Element) : void {

   // ? build/return DDG

   // calculate loopnest



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
         const expr_i = array_accesses.at(i);
         for (let j = 0; j < array_accesses.length; j++) {
            const expr_j = array_accesses.at(j);

            if (expr_i.getAccessType() === ArrayAccess.read_access &&
                expr_j.getAccessType() === ArrayAccess.read_access) continue;

            // get common nest
            // NOTE: may not even need to get the 
            let relevantLoopNest: xml.Element[] = LoopTools.getCommonEnclosingLoopNest(
               expr_i.getEnclosingLoop(), expr_j.getEnclosingLoop());
            // handle edge where loop being MT is not the outermost loop
            relevantLoopNest = LoopTools.getCommonLoops(relevantLoopNest, innerLoopNest);

            // TODO: Substitute Range Info

            const depVectors: DependenceVector[] = [];
            let dependenceExists: boolean = testArrayAccessPair();
            if (dependenceExists) {
               // add DV to DDG
               console.log("THIS FEATURE HASN'T BEEN FINISHED");
            }
         }
      }
   }
}


//test AccessPair
function testArrayAccessPair() : boolean {
   return true;
}

//test subscriptBySubscript

// getPartition

// testPartition

// test ZIV
// test SIV
// test MIV
   // test tree

export {analyzeLoopForDependence}