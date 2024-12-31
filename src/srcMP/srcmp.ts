#!/usr/bin/env node
import * as Xml from '../common/Xml/Xml.js';
import * as CLO from '../common/CommandLineOutput.js';
import * as DDT from './DataDependenceTesting/DataDependenceTestingPass.js';
import * as PLD from './ParallelizableLoopDetection/ParallelizableLoopDetectionPass.js';

import { Command } from 'commander';
import { Verbosity, setVerbosity } from '../common/CommandLineOutput.js';
import { writeFileSync } from 'fs';
import { getFileXml } from '../common/srcml.js';

function runCompiler(program: Xml.Element) : Xml.Element {
    Xml.setNamespaces(program);

    const programDDG = DDT.run(program);
    PLD.run(program, programDDG);

    return program;
}

/**
 * Parsing the verboisty command line argument and setting the global verbosity value
 * @param value the command argument passed in
 * @param previous unused
 */
/* eslint-disable-next-line @typescript-eslint/no-unused-vars*/ // previous is unused but mandatory to match library's function signature
function parseVerbosity(value: string, previous: number) : number {
    const num = Number(value);
    let verbosity: Verbosity = Verbosity.Basic;
    const isValidVerbosityNumber = (x: number): x is Verbosity => {
        return x in Object.values(Verbosity);
    };
    if (!isNaN(num) && isValidVerbosityNumber(num)) {
        verbosity = num;
    } else {
        CLO.warn(`Invalid verbosity input, defaulting to ${verbosity}`);
    }
    setVerbosity(verbosity);
    return verbosity;
}

/**
 * Generates the output file
 * @param inputFilePath the input file for the compilation
 * @param outputFilePath if supplied, the output file path to write to 
 * @param node the XML to write to the file
 * @param outputXml True to output as a .xml file, false to output as a source code file
 */
function printFile(inputFilePath: string, outputFilePath: string | undefined, node: Xml.Element, outputXml: boolean) {
    let outputFile: string = '';
    if (outputFilePath) {
        outputFile = outputFilePath;
    } else {
        const dirIndex = inputFilePath.lastIndexOf('/');
        const path = dirIndex !== -1 ? inputFilePath.substring(0, dirIndex + 1) : '';
        const fileName = inputFilePath.substring(dirIndex + 1);
        outputFile = `${path}mp_${fileName}`;
    }
    if (outputXml && inputFilePath.substring(inputFilePath.length - 4) !== '.xml') outputFile += '.xml';
    writeFileSync(outputFile, outputXml ? node.toString() : node.text);
}

/**
 * Program entry point
 */
function main() : number {
    const program = new Command();
    program.name('src-cetus')
        .description('A srcML based optimizing compiler')
        .version('0.0.1')
        .argument('<input-files...>', 'The files to be compiled.')
        .option('-o, --output <output-files...>', 'The paths of the output files. If none is supplied, the default naming convention is mp_<input_file>')
        .option('-v, --verbosity <number>', `Controls the verbosity level. ${JSON.stringify(Verbosity)}`, parseVerbosity, Verbosity.Basic)
        .option('--noEmit', 'Disable emitting files from a compilation.', false)
        .option('--xml', 'Output an srcML formatted XML file instead of source code.', false);
    program.parse();
    const options = program.opts();
    const outputFiles = (options.output ?? []) as string[];
    const noEmit = options.noEmit as boolean;
    const outputXml = options.xml as boolean;

    for (let i = 0; i < program.args.length; i++) {
        const inputFile = program.args[i];
        try {
            const compiledCode = runCompiler(getFileXml(inputFile));
            if (noEmit) continue;
            printFile(inputFile, outputFiles[i], compiledCode, outputXml);
        } catch (error) {
            CLO.error(`Error compiling ${inputFile}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    return 0;
}

main();
