#!/usr/bin/env node

import * as cuda from './cuda.js'
import * as Xml from '../Xml/Xml.js'
import { Command } from 'commander'
import { execSync } from 'child_process'


// get all the names
// find latest assignment for each name

function getLatestAssignment(variable: Xml.Element) : Xml.Element | undefined {
    // TODO: Increment/Decrement
    // TODO: Augmented Assignment
    // TODO: handeling scope

    const pointIsEarlier = (point: {line: number, col: number}, other: {line: number, col: number}) => {
        return (point.line < other.line || (point.line == other.line && other.col < other.col))
    };

    const decl = variable.find(`//xmlns:decl[xmlns:name[text()='${variable.text}']]`)
        .map((node) => node.get('xmlns:init/xmlns:expr')).filter((expr) => {
            return expr && pointIsEarlier(expr, variable);
        }).at(-1);

    const exprs = variable.find(`//xmlns:expr[xmlns:name[text()='${variable.text}' and contains(following-sibling::xmlns:operator, '=')]]`);
    const assignment = exprs.filter((expr) => {
        return Xml.hasAssignmentOperator(expr) || Xml.hasAugAssignmentOperator(expr);
    }).at(-1);
    
    if (!decl && !assignment) {
        return undefined;
    } else if (!decl) {
        return Xml.getRHSFromExpr(assignment!);
    } else if (!assignment) {
        return decl;
    }

    if (pointIsEarlier(assignment, decl)) {
        return decl;
    } else {
        return Xml.getRHSFromExpr(assignment);
    }
}

function testDependenceHierachy(xml: Xml.Element, depth: number = 0) : boolean {
    // ?: Handeling Arrays and objects
    const vars = xml.find('.//xmlns:name');
    const padding = `${" ".repeat(depth)}`;
    console.log(`${padding}"${xml.text.trim()}" depends on variables: {${vars.map((value) => value.text).join(', ')}}`)

    for (const variable of vars) {
        if (cuda.builtInVariables.includes(variable.text)) {
            console.log(`${padding}${xml.text} is dependent on built-in variable "${variable.text}"`)
            return true;
        }

        const latest = getLatestAssignment(variable);
        if (!latest) {
            console.log(`${padding}No dependence found for variable "${variable.text}"`)
            continue;
        }
        console.log(`${padding}variable "${variable.text}" depends on "${latest.text}"`);
        if (testDependenceHierachy(latest, depth + 1)) return true;
    }

    return false;
}

function runDivergenceTest(func: Xml.Element) {
    const ifStmts = func.find('.//xmlns:if')
    for (const ifStmt of ifStmts) {
        console.log(`[DivergenceTesting] Testing if at line ${ifStmt.line}`);
        if (testDependenceHierachy(ifStmt.get('xmlns:condition')!)) {
            console.log(`*Potential divergence detected*`);
        } else {
            console.log(`*No Potential divergence detected*`);
        }
        console.log();
    }
}

function runTest(xml: Xml.Element) {
    // ? Nested Functions
    const kernelFunctions = xml.find(`.//xmlns:function[xmlns:type/xmlns:name[text()='__global__']]`);

    for (const func of kernelFunctions) {
        runDivergenceTest(func);
    }
}

/**
 * Converts file contents to an Xml Object
 * * Program assumes that .xml files passed to program are srcml applied to one file
 * @param srcPath the path to the file as a string
 * @returns an xml object representing the file contents
 */
function getFileXml(srcPath: string) : Xml.Element {
    //TODO: Create SRCML interface for whole project to use
    const fileExtension = srcPath.substring(srcPath.lastIndexOf("."))
    
    if (fileExtension !== ".xml") {
        const buffer = execSync(`srcml ${srcPath} --language C++`, {timeout: 10000});
        return Xml.parseXmlString(buffer.toString());
    } else {
        return Xml.parseXmlFile(srcPath);
    }
}

/**
 * Handles command line args and begins compiler run
 * @returns exit code
 */
function main() : number {
    const program = new Command();

    program.name('divergence-tester')
        .version('0.0.1')
        .argument('<input-files...>', 'The files to be tested');

    program.parse();    

    for (const inputFile of program.args) {
        runTest(getFileXml(inputFile));
        return 1;
    }

    return 0;
}

main()