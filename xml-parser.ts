import * as fs from 'fs'
// import * as libxmljs from 'libxmljs2'
import libxmljs = require('libxmljs2')

function isElement(root: libxmljs.Element | libxmljs.Node): root is libxmljs.Element {
    return (root as libxmljs.Element).childNodes !== undefined;
  }

function simpleFor(root: libxmljs.Element) : void {

}

// TO-DO: Change general parse to individual rule based using XPATH
function parse(root: libxmljs.Element | libxmljs.Node) : void {
    if (!isElement(root)) {
        return
    }


    if (root.name() === 'for') { 
        const loopVariable = root.find("xmlns:control/xmlns:init/xmlns:decl/xmlns:name", "http://www.srcML.org/srcML/src")[0] as libxmljs.Element
        const loopBody = root.find("xmlns:block/xmlns:block_content", "http://www.srcML.org/srcML/src")[0] as libxmljs.Element
        let canMT = true
        // find all instances that i is used, if i is being modified than assume loop is unparallelizable
        const names = loopBody.find('.//xmlns:name', 'http://www.srcML.org/srcML/src')
        for (let i = 0; i < names.length; i ++) {
            // if element is i
            // if the parent expr has more than one child node then i is being modified
            if ((names[i] as libxmljs.Element).text() === 'i' 
                && (names[i].parent() as libxmljs.Element).childNodes().length > 1) {   
                canMT = false
                break
            }
        }
        if (canMT) console.log("Parallelizable For @ Line: " + root.line())
    }

    for (let i = 0; i < root.childNodes().length; i++) {
        parse(root.childNodes()[i])
    }
}

function begin_parse(srcPath: string) {
    const doc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString())
    parse(doc.root())
}

function main() {
    if (process.argv.length < 3) {
        console.error("Specify python source file as command-line argument.")
    } else {
        for (let i = 2; i < process.argv.length; i++) {
            begin_parse(process.argv[i])
        }
    }
}
  
main()
