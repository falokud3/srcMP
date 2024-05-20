#!/usr/bin/env node

import * as fs from 'fs';
import * as libxmljs from 'libxmljs2';
import * as LoopTools from './util/LoopTools.js'
import * as XmlTools from './util/XmlTools.js'
import * as DDFramework from './DDTFramework.js'
import * as CFG from './ControlFlowGraph.js'
import { Command } from 'commander'

import { exec } from 'child_process';

// TODO: PARSE unit tag for namespaces
const namespaces = {'xmlns': 'http://www.srcML.org/srcML/src'}
  
function autoparPass(root: libxmljs.Element) : void {
    // const forLoops = root.find('//xmlns:for', namespaces) as libxmljs.Element[];
    // TODO: Handeling of nested loops
       // TODO: extracting only the outermost loops
    // forLoops.forEach((forNode: libxmljs.Element) => {
        // if (!LoopTools.isLoopEligible(forNode)) return;
        // DDFramework.analyzeLoopForDependence(forNode);
    // });
    // console.log(root.toString())

    const test = root.find("//xmlns:function", namespaces) as libxmljs.Element[];
    for (const func of test) {
        const graph = CFG.CFGraph.buildControlFlowGraph(func);
        console.log(graph.toString());
    }
    
}

function begin_parse(srcPath: string) {

    const fileExtension = srcPath.substring(srcPath.lastIndexOf("."))
    
    // * Assumes that .xml passed to program srcml if not the program breaks
    if (fileExtension !== ".xml") {
        exec(`srcml ${srcPath}`, (error, stdout, stderr) => {
            if (error) {
                console.error(`exec error: ${error}`);
                return;
            }

            const xmlDoc = libxmljs.parseXmlString(stdout);
            autoparPass(xmlDoc.root());
          });
    } else {
        const xmlDoc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString());
        autoparPass(xmlDoc.root());
    }
    
}

function main() : number {
    const program = new Command();

    program.name('src-cetus')
        .description('A srcML based optimizing compiler')
        .version('0.0.1')
        .argument('<input-files...>', 'The files to be compiled');

    program.parse();    

    for (const inputFile of program.args) {
        try {
            begin_parse(inputFile);
        } catch (error) {
            console.error(inputFile + ": " + error.message + "(" + error.name + ")");
            return 1;
        }
    }

    return 0;
}


main();
