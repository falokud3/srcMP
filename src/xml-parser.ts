import * as fs from 'fs';
import * as libxmljs from 'libxmljs2';
import * as LoopTools from './util/LoopTools.js'
import * as XmlTools from './util/XmlTools.js'
import * as DDFramework from './DDTFramework.js'
import * as CFG from './ControlFlowGraph.js'

// TODO: PARSE unit tag for namespaces
const namespaces = {'xmlns': 'http://www.srcML.org/srcML/src'}


  
function autoparPass(root: libxmljs.Element) : void {
    // const forLoops = root.find('//xmlns:for', namespaces) as libxmljs.Element[];
    // // TODO: Handeling of nested loops
    //    // TODO: extracting only the outermost loops
    // forLoops.forEach((forNode: libxmljs.Element) => {
    //     if (!LoopTools.isLoopEligible(forNode)) return;
    //     DDFramework.analyzeLoopForDependence(forNode);
    // });

    const test = root.find("//xmlns:function", namespaces) as libxmljs.Element[];
    for (const func of test) {
        const graph = CFG.CFGraph.buildControlFlowGraph(func);
        console.log(graph.toString());
    }
    
}

function begin_parse(srcPath: string) {
    const doc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString());
    autoparPass(doc.root());
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
