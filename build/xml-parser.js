"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
// import * as libxmljs from 'libxmljs2'
var libxmljs = require("libxmljs2");
// TO-DO: PARSE unit tag for namespaces
var namespaces = { 'xmlns': 'http://www.srcML.org/srcML/src' };
function isElement(root) {
    return root.childNodes !== undefined;
}
function simpleDOALL(root) {
    // retrieves every for loop
    var forLoops = root.find('//xmlns:for', namespaces);
    var _loop_1 = function (i) {
        var loopNode = forLoops[i];
        var loopVariable = loopNode.find("xmlns:control/xmlns:init/xmlns:decl/xmlns:name", namespaces)[0];
        var loopBody = loopNode.find("xmlns:block/xmlns:block_content", "http://www.srcML.org/srcML/src")[0];
        // find all instances where the loopVariable is used, if it is being modified than assume loop is unparallelizable
        var names = loopBody.find('.//xmlns:name', 'http://www.srcML.org/srcML/src').filter(function (name) {
            if (isElement(name)) {
                return name.text() === loopVariable.text();
            }
            return false;
        });
        var canMT = true;
        for (var i_1 = 0; i_1 < names.length; i_1++) {
            if (names[i_1].parent().childNodes().length > 1) {
                canMT = false;
                break;
            }
        }
        if (canMT)
            console.log("Parallelizable For @ Line: " + loopNode.line());
    };
    for (var i = 0; i < forLoops.length; i++) {
        _loop_1(i);
    }
}
function begin_parse(srcPath) {
    var doc = libxmljs.parseXmlString(fs.readFileSync(srcPath).toString());
    simpleDOALL(doc.root());
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
