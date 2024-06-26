#!/usr/bin/env node

import * as cuda from './cuda.js'
import * as Xml from '../Facades/Xml/Xml.js'
import { Command } from 'commander'
import { execSync } from 'child_process'
import chalk from 'chalk';


// get all the names
// find latest assignment for each name

type Point = {line: number, col: number};
const NegativeInfinityPoint = {line: -Infinity, col: -Infinity};

function testMultipleBranches(variable: Xml.Element, latest: Xml.Element, depth: number) : boolean {
    const ifs = latest.get('ancestor::xmlns:if_stmt')!.find('xmlns:if');
    for (let i = ifs.length - 1; i >= 0; i--) {
        console.log()
        if (testDependenceHierachy(variable, depth++, ifs[i].nextElement!)) return true;
    }
    return false;
}


function getLatestAssignment(variable: Xml.Element, beforePoint?: Point) : [Xml.Element | undefined, Point | undefined] {
    // TODO: Increment/Decrement
    // TODO: Augmented Assignment
    // TODO: handeling scope

    const pointIsEarlier = (point: Point | undefined, other: Point | undefined) => {
        if (!point) point = NegativeInfinityPoint;
        if (!other) other = NegativeInfinityPoint;
        return (point.line < other.line || (point.line == other.line && other.col < other.col))
    };

    const unary = variable.find(`//xmlns:name[following-sibling::*[1]/text() = '++' 
        or following-sibling::*[1]/text() = '--' 
        or preceding-sibling::*[1]/text() = '++' 
        or preceding-sibling::*[1]/text() = '--']`)
        .filter((expr) => pointIsEarlier(expr, beforePoint ?? variable))
        .map((expr) => expr.parentElement ?? expr)
        .at(-1);

        // TODO: swap the filter and the map
    const decl = variable.find(`//xmlns:decl[xmlns:name[text()='${variable.text}']]`)
        .map((node) => node.get('xmlns:init')!)
        .filter((expr) => {
            return expr && pointIsEarlier(expr, beforePoint ?? variable);
        }).at(-1);

    const exprs = variable.find(`//xmlns:expr[xmlns:name[text()='${variable.text}' and contains(following-sibling::xmlns:operator, '=')]]`);
    const assignment = exprs.filter((expr) => {
        return (Xml.hasAssignmentOperator(expr) || Xml.hasAugAssignmentOperator(expr)) 
        && pointIsEarlier(expr, beforePoint ?? variable);
    }).at(-1);
    
    // TODO: Refactor this mess
    if (!decl && !assignment && !unary) {
        return [undefined, undefined];
    } 


    if (pointIsEarlier(assignment, decl) && pointIsEarlier(unary, decl)) {
        return [decl, decl?.get('ancestor::xmlns:if_stmt') ?? undefined ];
    } else if (pointIsEarlier(assignment, unary) && pointIsEarlier(decl, unary)) {
        return [unary, unary?.get('ancestor::xmlns:if_stmt') ?? undefined];
    } else {
        return Xml.hasAugAssignmentOperator(assignment!) 
        ? [assignment, assignment?.get('ancestor::xmlns:if_stmt') ?? {line: assignment!.line, col: assignment!.col - 1}] 
        : [Xml.getRHSFromExpr(assignment!), assignment?.get('ancestor::xmlns:if_stmt') ?? undefined];
    }
}

function testDependenceHierachy(xml: Xml.Element, depth: number = 0, before?: Point) : boolean {
    // ?: Handeling Arrays and objects
    const vars = xml.find('.//xmlns:name');
    const padding = `${" ".repeat(depth)}`;
    console.log(`${padding}"${xml.text.trim()}" depends on variables: {${vars.map((value) => value.text).join(', ')}}`)

    for (const variable of vars) {
        if (cuda.builtInVariables.includes(variable.text)) {
            console.log(`${padding}${xml.text} is dependent on built-in variable "${variable.text}"`)
            return true;
        }

        const [latest, beforePoint] = getLatestAssignment(variable, before);
        if (!latest) {
            console.log(`${padding}No dependence found for variable "${variable.text}"`)
            continue;
        }

        console.log(`${padding}variable "${variable.text}" depends on "${latest.text}"`);
        if (testDependenceHierachy(latest, depth + 1, beforePoint)) return true;

        if (latest.contains('ancestor::xmlns:else')) {
            console.log(`${padding}Testing other branches`);
            if (testMultipleBranches(variable, latest, depth)) return true;
        }
    }

    return false;
}

function runDivergenceTest(func: Xml.Element) {
    const ifs = func.find('.//xmlns:if');
    const divergentIfs: Xml.Element[] = [];
    for (const ifXml of ifs) {
        console.log(`[DivergenceTesting] Testing if at line ${ifXml.line}`);
        if (testDependenceHierachy(ifXml.get('xmlns:condition')!)) {
            console.log(chalk.red(`*Potential divergence detected*`));
            divergentIfs.push(ifXml);
        } else {
            console.log(chalk.green(`*No Potential divergence detected*`));
        }
        console.log();
    }

    const divergentCoverage = divergentIfs.map((ifXml) => {
        const block = ifXml.get('xmlns:block/xmlns:block_content')!;
        const startLine = Number(block.getAttribute('pos:start')!.split(':')[0]);
        const endLine = Number(block.getAttribute('pos:end')!.split(':')[0]);
        return endLine - startLine + 1;
    }).reduce((sum, curr) => sum + curr, 0);

    const block = func.get('xmlns:block/xmlns:block_content')!;
    const startLine = Number(block.getAttribute('pos:start')!.split(':')[0]);
    const endLine = Number(block.getAttribute('pos:end')!.split(':')[0]);

    const metric = divergentCoverage / (endLine - startLine) * 100;

    console.log(`Divergent Coverage: ${metric.toPrecision(4)}%`);
}

function runTest(xml: Xml.Element) {
    // ? Nested Functions
    // TODO: make language adaptable
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
        const buffer = execSync(`srcml ${srcPath} --language C++ --position`, {timeout: 10000});
        return Xml.parseXmlString(buffer.toString());
    } else {
        return Xml.parseXmlFile(srcPath);
    }
}

// TODO: interface in /bin
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

