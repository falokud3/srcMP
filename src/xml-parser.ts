import * as fs from 'fs';
import * as libxmljs from 'libxmljs2';
// import libxmljs = require('libxmljs2')
import { DDG_Node, DirectedDependencyGraph } from './ddg.js';
import assert from 'assert';

import * as LoopTools from './LoopTools.js'

// TODO: PARSE unit tag for namespaces
const namespaces = {'xmlns': 'http://www.srcML.org/srcML/src'}

function nameToNode(name: libxmljs.Element) : DDG_Node {
    return new DDG_Node('');
}
  
/**
 * Takes a for loop xml element and builds a ddg from it. Throws an error
 * if the passed element is not a for loop
 * @param root the for xml element
 * @returns DDG representingb the for loop
 */
function buildLoopDDG(root: libxmljs.Element) : DirectedDependencyGraph {
    assert(root.name() === 'for');

    const ddg = new DirectedDependencyGraph();
    // go through all the declarations
    const decl_statements = root.find(".//xmlns:decl", namespaces) as libxmljs.Element[]; 
    decl_statements.forEach((decl) => {
        // TODO: add clause for INIT
        const type = (decl.get('./xmlns:type/xmlns:name', namespaces) as libxmljs.Element).text();
        const name = (decl.get('./xmlns:name', namespaces) as libxmljs.Element).text();
        ddg.addVertex(new DDG_Node(name));
    });

    // get all expressions
        // filter for init and assignment operator
    const expr_statements = root.find(".//xmlns:expr", namespaces) as libxmljs.Element[];
    expr_statements.forEach((expr) => {

        // check if expr has init parent
        const parent = expr.parent() as libxmljs.Element;
        if (parent.name() === 'init') {
            // parent of <init> is <decl> 
                // the first <name> child of <decl> is the 
                // variable being declared
            const target = parent.get('../xmlns:name', namespaces) as libxmljs.Element;

            // all <name> children of the expr are variables to be added to ddg
                // NOTE: unsure how constants appear as srcml
                // NOTE: object dependencies won't work yet;
            const vars = expr.find('./xmlns:name', namespaces) as libxmljs.Element[];
            vars.forEach((variable) => {
                ddg.addEdge( new DDG_Node(target.text()), new DDG_Node(variable.text()));
            });
        }

        // check if expr has <operator>=</operator>
        const operators = expr.find('./xmlns:operator', namespaces) as libxmljs.Element[];

        // using for over .reduce to avoid looping through entire array
            // unnecessarily
        let isAssignment = false;
        for (let i = 0; i < operators.length && !isAssignment; i++) {
            isAssignment = (operators[i].text() === '=');
        }
        if (isAssignment) {
            const variables = expr.find('./xmlns:name', namespaces) as libxmljs.Element[];
            const target = variables[0];
            variables.slice(1).forEach((source) => {
                ddg.addEdge(new DDG_Node(target.text()), new DDG_Node(source.text()))
            });
        }

    });

    console.log(ddg.toString());

    // TODO: treat array indices as variables
    // TODO: object members

    // variable usage through the name
        // exception: method names have call parent
        // exception: data types have type parent
        // TODO: figure out how to isolate the name of an object without
            // the method/attribute (vector.push_back())
    return ddg;
}

/**
 * Takes the root element of a srcML document and produces a 
 * DirectedDependencyGraph for each for loop within that document
 * @param root xml root element
 */
function forsToDDG(root: libxmljs.Element) : void {
    const forLoops = root.find('//xmlns:for', namespaces) as libxmljs.Element[];
    forLoops.forEach((forNode) => {
        const ddg = buildLoopDDG(forNode);
        // TODO: FIX CYCLIC
        if (!ddg.isCyclic()) console.log("Parallelizable For @ Line: " + forNode.line());
    });
    
}

function testGetArrayAccesses(root: libxmljs.Element) : void {
    const forLoops = root.find('//xmlns:for', namespaces) as libxmljs.Element[];
    forLoops.forEach((forNode) => {
        LoopTools.getArrayAccesses(forNode);
    });
    
}

function begin_parse(srcPath: string) {
    const doc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString())
    testGetArrayAccesses(doc.root())
}

function main() : void {
    if (process.argv.length < 3) {
        console.error("Specify python source file as command-line argument.")
    } else {
        for (let i = 2; i < process.argv.length; i++) {
            begin_parse(process.argv[i])
        }
    }
}


main();
