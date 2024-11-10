#!/usr/bin/env node

import * as ParallelProgrammingInfo from './ParallelProgrammingInfo.js'
import * as Xml from '../common/Xml/Xml.js'
import { Command } from 'commander'
import { execSync } from 'child_process'
import chalk from 'chalk';
import { readFileSync } from 'fs';

let builtInVariables: string[] = [];
let verbose: boolean = false;

type Device = {
    name: string;
    clockFrequency: number;
    affectedByDivergence: boolean;
    ops: {
        [op: string]: {
            min: number;
            max: number;
        } | undefined;
        default: {
            min: number;
            max: number;
        }
    }
}


// TODO: Try both
function formatOP(op: string) : string {
    switch (op) {
        case '+':
        case '+=':
        case '-':
        case '-=':
            return 'add';
        case '*':
        case '*=':
            return 'mul';
        case '/':
        case '/=':
            return 'div'
        case '<':
        case '<=':
        case '>':
        case '>=':
        case '&&':
        case '!=':
        case '==':
        case '||':
            return 'lop'
        default:
            return op;
    }
}

// coalesced memory access are either not based on index variables 
// if they are based on index variables, then must be linear combination
function isCoalesced(access: Xml.Element) : boolean {
    const expr = access.get('xmlns:index/xmlns:expr');
    // only testing array access that based on thread true
    if (!expr || !divergenceTest(expr)) return true; 
    return !Boolean(expr.get('.//xmlns:operator[. != "+" and . != "-"]'));
    
}

type Point = {line: number, col: number};
const NegativeInfinityPoint = {line: -Infinity, col: -Infinity};

function getLatestAssignment(variable: Xml.Element, beforePoint?: Point) : [Xml.Element | undefined, Point | undefined] {

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
    
    if (!decl && !assignment && !unary) return [undefined, undefined];

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

function multipleBranchDivergenceTest(variable: Xml.Element, latest: Xml.Element, depth: number) : boolean {
    const padding = `${" ".repeat(depth)}`;
    const ifs = latest.get('ancestor::xmlns:if_stmt')!.find('xmlns:if');
    for (let i = ifs.length - 1; i >= 0; i--) {
        const otherBranchResult = getLatestAssignment(variable, ifs[i].nextElement!)
        if (!otherBranchResult[0]) continue;
        if (verbose) console.log(`${padding}variable "${variable.text}" may also depends on "${otherBranchResult[0].text}"`);
        if (divergenceTest(otherBranchResult[0], depth + 1, otherBranchResult[1])) return true;
    }
    return false;
}

function divergenceTest(xml: Xml.Element, depth: number = 0, before?: Point) : boolean {
    // ?: Handeling Arrays and objects
    const vars = xml.find('./descendant-or-self::xmlns:name');
    const padding = `${" ".repeat(depth)}`;
    if (verbose) console.log(`${padding}"${xml.text.trim()}" depends on variables: {${vars.map((value) => value.text).join(', ')}}`)

    for (const variable of vars) {
        if (builtInVariables.includes(variable.text)) {
            if (verbose) console.log(`${padding}${xml.text} is dependent on built-in variable "${variable.text}"`)
            return true;
        }

        const [latest, beforePoint] = getLatestAssignment(variable, before);
        if (!latest) {
            if (verbose) console.log(`${padding}No dependence found for variable "${variable.text}"`)
            continue;
        }

        if (verbose) console.log(`${padding}variable "${variable.text}" depends on "${latest.text}"`);
        if (divergenceTest(latest, depth + 1, beforePoint)) return true;

        if (latest.contains('ancestor::xmlns:else')) {
            if (multipleBranchDivergenceTest(variable, latest, depth)) return true;
        }
    }

    return false;
}

function calculateDivergenceCoverage(func: Xml.Element) {
    const ifs = func.find('.//xmlns:if');
    const divergentIfs: Xml.Element[] = [];
    for (const ifXml of ifs) {
        if (verbose) console.log(`[DivergenceTesting] Testing if at line ${ifXml.line}`);
        if (divergenceTest(ifXml.get('xmlns:condition')!)) {
            if (verbose) console.log(chalk.red(`*Potential divergence detected*`));
            divergentIfs.push(ifXml);
        }
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

function estimateClockCycles(xml: Xml.Element, device: Device) : [number, number, number] {
    let min = 0;
    let max = 0;
    const ops = xml.find('.//xmlns:operator').filter((op) => ![',','.', '=',].includes(op.text));
    const calls = xml.find('.//xmlns:call/xmlns:name');

    for (const op of ops.concat(calls)) {
        let cycles = device.ops[formatOP(op.text)];
        if (cycles === undefined) cycles = device.ops.default;

        if (verbose) console.log(`${op.text} -> [${cycles.min},${cycles.max}] ${device.ops[formatOP(op.text)] === undefined ? '(DEFAULT)' : ''}`)
        min += cycles.min;
        max += cycles.max;
    }

    const reads = xml.find('.//xmlns:name[count(ancestor::xmlns:name)=0]');
    for (const read of reads) {
        min += device.ops['MOV']?.min ?? device.ops.default.min;
        max += device.ops['MOV']?.max ?? device.ops.default.max;
        if (!isCoalesced(read)) {
            if (verbose) console.log(read.text, chalk.red("Uncoalesced Access"))
            min += device.ops['MOV']?.min ?? device.ops.default.min;
            max += device.ops['MOV']?.max ?? device.ops.default.max;
        }
        
    }

    const nonDivergentIfs = xml.find(`.//xmlns:if_stmt[not(ancestor::xmlns:if_stmt) or ancestor::xmlns:if_stmt[1]/text() != "${xml.text}"]`).filter((ifStmt) => {
        return !device.affectedByDivergence || !ifStmt.find('xmlns:if/xmlns:condition').some((cond) => divergenceTest(cond));
    });


    for (const nonDivIf of nonDivergentIfs) {
        const branches = nonDivIf.find('xmlns:if')
            .concat(nonDivIf.find('xmlns:else'))
            .map((branchXml) => estimateClockCycles(branchXml, device));
        const branchMax = branches.reduce( (max, curr) => max[2] > curr[2] ? max : curr, branches[0]);
        branches.splice(branches.indexOf(branchMax), 1);
        for (const branch of branches) {
            min -= branch[0];
            max -= branch[1];
        }
    }

    // ! COMPILER OPTIMIZATION ADJUSTMENT FACTOR
    min /= 2;
    max /= 2;

    return  [Math.floor(min), Math.floor(max), Math.floor((2 * min * max) / (min + max))];
}

function getKernelFunctions(program: Xml.Element, language: ParallelProgrammingInfo.SupportedLanguages) : Xml.Element[] {
    if (language === 'cu' || language === 'hip') {
        return program.find(`.//xmlns:function[xmlns:type/xmlns:name[text()='__global__']]`);
    } else if (language === 'cl') {
        return program.find(`.//xmlns:function[xmlns:type/xmlns:name[text()='__kernel']]`);
    } else {
        console.error(chalk.yellow(`Kernel function syntax for file extension "${language}" not found. Defaulting to cuda's kernel function syntax.`));
        return program.find(`.//xmlns:function[xmlns:type/xmlns:name[text()='__global__']]`);
    }
}

function outputEstimates(xml: Xml.Element, language: ParallelProgrammingInfo.SupportedLanguages, devices: Device[]) {
    // ? Nested Functions
    builtInVariables = ParallelProgrammingInfo.cudaBuiltIns;
    Xml.setNamespaces(xml);

    const kernelFunctions = getKernelFunctions(xml, language);

    for (const func of kernelFunctions) {
        console.log(`${func.get('./xmlns:name')?.text ?? 'FunctionNameNotFound'} (line ${func.line} col ${func.col}):`);
        for (const device of devices) {
            const clockCycles = estimateClockCycles(func, device);
            const time = clockCycles.map((cycles) => (cycles / device.clockFrequency / 1000000).toPrecision(5));
            console.log(`  ${device.name} Clock Cycles Estimate Min: ${chalk.yellow(clockCycles[0])} Max: ${chalk.yellow(clockCycles[1])} Avg: ${chalk.yellow(clockCycles[2])}`);
            console.log(`  ${device.name} Time Estimate Min: ${chalk.yellow(time[0])}s Max: ${chalk.yellow(time[1])}s Avg: ${chalk.yellow(time[2])}s`);
            console.log()
        }
    }
}

/**
 * Converts file contents to an Xml Object
 * * Program assumes that .xml files passed to program are srcml applied to one file
 * @param srcPath the path to the file as a string
 * @returns an xml object representing the file contents
 */
function getFileXml(srcPath: string) : Xml.Element {
    const fileExtension = srcPath.substring(srcPath.lastIndexOf("."));
    
    if (fileExtension !== ".xml") {
        const buffer = execSync(`srcml ${srcPath} --language C++ --position`, {timeout: 10000});
        return Xml.parseXmlString(buffer.toString());
    } else {
        return Xml.parseXmlFile(srcPath);
    }
}

function setBuiltInVariables(filePath: string) : ParallelProgrammingInfo.SupportedLanguages {
    const extension = filePath.substring(filePath.lastIndexOf("."));

    if (extension === '.cu') {
        builtInVariables = ParallelProgrammingInfo.cudaBuiltIns;
        return 'cu'
    } else if (extension === '.cpp') {
        builtInVariables = ParallelProgrammingInfo.hipBuiltIns;
        return 'hip'
    } else if (extension === '.cl') {
        builtInVariables = ParallelProgrammingInfo.openclBuiltIns;
        return 'cl'
    } else {
        console.error(chalk.yellow(`Built in variables for file extension "${extension}" not found. Defaulting to cuda's variables.`));
        builtInVariables = ParallelProgrammingInfo.cudaBuiltIns;
        return 'cu'
    }
}

function parseDevices(filePaths: string[]) : Device[] {
    const setDefault = (filePath: string, device: Device, member: string, type: string, defaultValue: any) => {
        //@ts-expect-error - using indexing code smell to avoid repetition in code
        const value = device[member];
        if (value === undefined) {
            //@ts-expect-error - same as above
            device[member] = defaultValue;
            console.log(chalk.yellow(`${filePath} Device ${member} not found, defaulting to ${JSON.stringify(defaultValue)}.`))
        } else if (typeof value !== type) {
            //@ts-expect-error - same as above
            device[member] = defaultValue;
            console.log(chalk.yellow(`${filePath} Device ${member} is not the correct data type, defaulting to ${JSON.stringify(defaultValue)}.`))
        }
    };
    const devices: Device[] = [];
    for (const filePath of filePaths) {
        const device: Device = JSON.parse(readFileSync(filePath).toString());
        if (device.name === undefined) {
            device.name = `Device${devices.length + 1}`;
            console.log(chalk.yellow(`${filePath} Device name not found, defaulting to ${device.name}.`))
        }
        setDefault(filePath, device, 'name', 'string', `Device${devices.length + 1}`);
        setDefault(filePath, device, 'clockFrequency', 'number', 750);
        setDefault(filePath, device, 'affectedByDivergence', 'boolean', true);
        setDefault(filePath, device, 'ops', 'object', {default: {min:6, max: 6}})

        if (device.ops.default === undefined) {
            device.ops.default = {min: 6, max: 6}
            console.log(chalk.yellow(`${filePath} Device default op not found, defaulting to ${JSON.stringify(device.ops.default)}.`))
        }

        devices.push(device);
    } 
    
    return devices;
}

/**
 * Handles command line args and begins compiler run
 * @returns exit code
 */
function main() : number {
    const program = new Command();

    program.name('srcsim')
        .version('0.0.1')
        .argument('<input-files...>', 'The source code files whose kernels runtime are being estimated.')
        .requiredOption('-d, --devices <device-files...>', 'The files that describe the characteristics of the devices in JSON for estimation.' )
        .option('-v, --verbose', 'Displays the results of internal steps used to calculate estimation.', false);

    program.addHelpText('before', `${chalk.red('NOTE:')} This programs assumes that device clock frequency provided is in megahertz (MHz)`)

    program.parse();

    const options = program.opts();

    const devices = parseDevices(<string[]> options.devices)

    if (devices.length === 0) {
        console.error(chalk.red('Error: Could not parse any devices.'));
        return 1;
    }

    verbose = options.verbose;
    
    for (const inputFile of program.args) {
        console.log(`-- ${inputFile.substring(inputFile.lastIndexOf('/'))} --`);
        outputEstimates(getFileXml(inputFile), setBuiltInVariables(inputFile), devices);
    }

    return 0;
}

main()

