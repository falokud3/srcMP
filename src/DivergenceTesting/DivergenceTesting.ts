#!/usr/bin/env node

import * as cuda from './cuda.js'
import * as Xml from '../Facades/Xml/Xml.js'
import { Command } from 'commander'
import { execSync } from 'child_process'
import chalk from 'chalk';
import { readFileSync } from 'fs';

type OpDict = {
    [op: string]: {
        min: number;
        max: number;
    } | undefined;
    'DEFAULT': {
        min: number;
        max: number
    }
}

function formatOP(op: string) : string {
    switch (op) {
        case '+':
        case '+=':
        case '-':
        case '-=':
            return 'ADD';
        case '*':
        case '*=':
            return 'MUL';
        case '/':
        case '/=':
            return 'DIV'
        case '<':
        case '<=':
        case '>':
        case '>=':
        case '&&':
        case '!=':
        case '==':
        case '||':
            return 'LOP'
        default:
            return op;
    }
}

function estimateClockCycles(xml: Xml.Element, dict: OpDict) : [number, number, number] {
    const ops = xml.find('.//xmlns:operator').filter((op) => ![',','.', '=',].includes(op.text));
    const reads = xml.find('.//xmlns:name[count(ancestor::xmlns:name)=0]');
    const calls = xml.find('.//xmlns:call');

    let min = 0;
    let max = 0;

    for (const op of ops.concat(calls)) {
        let cycles = dict[formatOP(op.text)];
        if (cycles === undefined) cycles = dict.DEFAULT;

        // console.log(`${op.text} -> [${cycles.min},${cycles.max}] ${dict[formatOP(op.text)] === undefined ? '(DEFAULT)' : ''}`)
        min += cycles.min;
        max += cycles.max;
    }

    for (const read of reads) {
        min += dict['MOV']?.min ?? dict.DEFAULT.min;
        max += dict['MOV']?.max ?? dict.DEFAULT.max;
    }

    return [min, max, (min + max) / 2];
}

type Point = {line: number, col: number};
const NegativeInfinityPoint = {line: -Infinity, col: -Infinity};

function testMultipleBranches(variable: Xml.Element, latest: Xml.Element, depth: number) : boolean {
    const padding = `${" ".repeat(depth)}`;
    const ifs = latest.get('ancestor::xmlns:if_stmt')!.find('xmlns:if');
    for (let i = ifs.length - 1; i >= 0; i--) {
        const otherBranchResult = getLatestAssignment(variable, ifs[i].nextElement!)
        if (!otherBranchResult[0]) continue;
        console.log(`${padding}variable "${variable.text}" may also depends on "${otherBranchResult[0].text}"`);
        if (testDependenceHierachy(otherBranchResult[0], depth + 1, otherBranchResult[1])) return true;
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
        const block = ifXml.get('xmlns:block')!;
        const startLine = Number(block.getAttribute('pos:start')!.split(':')[0]);
        const endLine = Number(block.getAttribute('pos:end')!.split(':')[0]);
        return endLine - startLine - ifXml.emptyLines + 1;
    }).reduce((sum, curr) => sum + curr, 0);

    const block = func.get('xmlns:block')!;
    const startLine = Number(block.getAttribute('pos:start')!.split(':')[0]);
    const endLine = Number(block.getAttribute('pos:end')!.split(':')[0]);
    const metric = (divergentCoverage / (endLine - startLine - block.emptyLines + 1)) * 100;

    console.log(`Divergent Coverage: ${metric.toPrecision(4)}%`);
}

function runTest(xml: Xml.Element) {
    // ? Nested Functions
    // TODO: make language adaptable
    Xml.setNamespaces(xml);
    const kernelFunctions = xml.find(`.//xmlns:function[xmlns:type/xmlns:name[text()='__global__']]`);

    const fpgaDict: OpDict = JSON.parse(readFileSync('src/DivergenceTesting/ClockCycles/hlsclockcycles.json').toString())
    const gpuDict: OpDict = JSON.parse(readFileSync('src/DivergenceTesting/ClockCycles/turingclockcycles.json').toString())
    const TITAN_RTX_FREQ_MHZ = 1350;
    const FPGA_MHZ = 50;

    for (const func of kernelFunctions) {
        // runDivergenceTest(func);
        const fpgaCycles = estimateClockCycles(func, fpgaDict);
        const gpuCycles = estimateClockCycles(func, gpuDict)
        const fpgaEst = estimateClockCycles(func, fpgaDict).map((est) => (est / FPGA_MHZ / 1000).toPrecision(3));
        const gpuEst = estimateClockCycles(func, gpuDict).map((est) => (est / TITAN_RTX_FREQ_MHZ / 1000).toPrecision(3));
        console.log(`FPA Cycles Estimate Min: ${fpgaCycles[0]} Max:${fpgaCycles[1]} Avg:${fpgaCycles[2]}`);
        console.log(`GPU Cycles Estimate Min: ${gpuCycles[0]} Max:${gpuCycles[1]} Avg:${gpuCycles[2]}`);
        console.log(`FPA Estimate Min: ${fpgaEst[0]}ms Max:${fpgaEst[1]}ms Avg:${fpgaEst[2]}ms`);
        console.log(`GPU Estimate Min: ${gpuEst[0]}ms Max:${gpuEst[1]}ms Avg:${gpuEst[2]}ms`);

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

