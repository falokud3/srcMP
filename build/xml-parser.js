"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
// import * as libxmljs from 'libxmljs2'
var libxmljs = require("libxmljs2");
function isElement(root) {
    return root.childNodes !== undefined;
}
function simpleFor(root) {
}
// TO-DO: Change general parse to individual rule based using XPATH
function parse(root) {
    if (!isElement(root)) {
        return;
    }
    if (root.name() === 'for') {
        var loopVariable = root.find("xmlns:control/xmlns:init/xmlns:decl/xmlns:name", "http://www.srcML.org/srcML/src")[0];
        var loopBody = root.find("xmlns:block/xmlns:block_content", "http://www.srcML.org/srcML/src")[0];
        var canMT = true;
        // find all instances that i is used, if i is being modified than assume loop is unparallelizable
        var names = loopBody.find('.//xmlns:name', 'http://www.srcML.org/srcML/src');
        for (var i = 0; i < names.length; i++) {
            // if element is i
            // if the parent expr has more than one child node then i is being modified
            if (names[i].text() === 'i'
                && names[i].parent().childNodes().length > 1) {
                canMT = false;
                break;
            }
        }
        if (canMT)
            console.log("Parallelizable For @ Line: " + root.line());
    }
    for (var i = 0; i < root.childNodes().length; i++) {
        parse(root.childNodes()[i]);
    }
}
function begin_parse(srcPath) {
    var doc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString());
    parse(doc.root());
}
function main() {
    if (process.argv.length < 3) {
        console.error("Specify python source file as command-line argument.");
    }
    else {
        for (var i = 2; i < process.argv.length; i++) {
            begin_parse(process.argv[i]);
        }
    }
}
main();
