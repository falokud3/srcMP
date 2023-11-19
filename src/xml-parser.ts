import * as fs from 'fs';
import * as libxmljs from 'libxmljs2';
// import libxmljs = require('libxmljs2')
import { DDG_Node, DDG } from './ddg.js';


// TO-DO: PARSE unit tag for namespaces
const namespaces = {'xmlns': 'http://www.srcML.org/srcML/src'}

function isElement(root: libxmljs.Element | libxmljs.Node): root is libxmljs.Element {
    return (root as libxmljs.Element).childNodes !== undefined;
  }

function simpleDOALL(root: libxmljs.Element) : void {
    // retrieves every for loop
    const forLoops = root.find('//xmlns:for', namespaces)
    for (let i = 0; i < forLoops.length; i++) {
        const loopNode = forLoops[i] as libxmljs.Element
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

function begin_parse(srcPath: string) {
    const doc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString())
    simpleDOALL(doc.root())
}

// function main() {
//     if (process.argv.length < 3) {
//         console.error("Specify python source file as command-line argument.")
//     } else {
//         for (let i = 2; i < process.argv.length; i++) {
//             begin_parse(process.argv[i])
//         }
//     }
// }

function main() {
    let ddg: DDG = new DDG();
    const one: DDG_Node = new DDG_Node("1", "int");
    const two: DDG_Node = new DDG_Node("2", "int");
    const three = new DDG_Node("3", "int");
    ddg.addEdge(one, two); // change add edge; to add vertex as well
    ddg.addEdge(two, three);
    ddg.addEdge(three, one);
    console.log(ddg.toString());
    console.log(ddg.isCyclic());
}
  
main();
