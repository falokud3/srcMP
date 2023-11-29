import * as fs from 'fs';
import * as libxmljs from 'libxmljs2';
// import libxmljs = require('libxmljs2')
import { DDG_Node, DDG } from './ddg.js';
import assert from 'assert';


// TO-DO: PARSE unit tag for namespaces
const namespaces = {'xmlns': 'http://www.srcML.org/srcML/src'}

function isElement(root: libxmljs.Element | libxmljs.Node): root is libxmljs.Element {
    return (root as libxmljs.Element).childNodes !== undefined;
  }


function forsToDDG(root: libxmljs.Element) : void {
    const forLoops = root.find('//xmlns:for', namespaces) as libxmljs.Element[];
    forLoops.forEach((forNode) => {
        const ddg = buildLoopDDG(forNode);
        // TO-DO: FIX CYCLIC
        if (!ddg.isCyclic()) console.log("Parallelizable For @ Line: " + forNode.line());
    });
    
}

  // OLD APPROACH
function simpleDOALL(root: libxmljs.Element) : void {
    // retrieves every for loop
    const forLoops = root.find('//xmlns:for', namespaces) as libxmljs.Element[];

    for (let i = 0; i < forLoops.length; i++) {
        const loopNode = forLoops[i] as libxmljs.Element
        // console.log(loopNode.name());
        const loopVariable = loopNode.find("xmlns:control/xmlns:init/xmlns:decl/xmlns:name", namespaces)[0] as libxmljs.Element
        const loopBody = loopNode.find("xmlns:block/xmlns:block_content", "http://www.srcML.org/srcML/src")[0] as libxmljs.Element
        // find all instances where the loopVariable is used, if it is being modified than assume loop is unparallelizable
        const names = loopBody.find('.//xmlns:name', 'http://www.srcML.org/srcML/src').filter((name) => { 
            if (isElement(name)) {
                return name.text() === loopVariable.text()
            }
            return false;
        })
        let canMT = true
        for (let i = 0; i < names.length; i ++) {
            if ((names[i].parent() as libxmljs.Element).childNodes().length > 1) {
                canMT = false
                break
            }
        }
        if (canMT) console.log("Parallelizable For @ Line: " + loopNode.line())
    }
}

/**
 * Takes a for loop xml element and builds a ddg from it. Throws an error
 * if the passed element is not a for loop
 * @param root the for xml element
 * @returns DDG representingb the for loop
 */
function buildLoopDDG(root: libxmljs.Element) : DDG {
    assert(root.name() === 'for');

    const ddg = new DDG();
    // go through all the declarations
    const decl_statements = root.find(".//xmlns:decl", namespaces) as libxmljs.Element[]; 
    decl_statements.forEach((decl) => {
        // TO-DO: add clause for INIT
        const type = (decl.get('./xmlns:type/xmlns:name', namespaces) as libxmljs.Element).text();
        const name = (decl.get('./xmlns:name', namespaces) as libxmljs.Element).text();
        ddg.addVertex(new DDG_Node(name, type));
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
                ddg.addEdge( new DDG_Node(target.text(), "NULL"), new DDG_Node(variable.text(), "NULL"));
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
                ddg.addEdge(new DDG_Node(target.text(), "NULL"), new DDG_Node(source.text(), "NULL"))
            });
        }

    });

    console.log(ddg.toString());

    //TO-DO: treat array indices as variables
    //TO-DO: object members

    // variable usage through the name
        // exception: method names have call parent
        // exception: data types have type parent
        // TO-DO: figure out how to isolate the name of an object without
            // the method/attribute (vector.push_back())
    return ddg;
}

function begin_parse(srcPath: string) {
    const doc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString())
    forsToDDG(doc.root())
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
