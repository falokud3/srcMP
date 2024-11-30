#!/usr/bin/env node
import * as Xml from '../common/Xml/Xml.js';
// import * as DDT from './DataDependenceTesting/DataDependenceTestingPass.js';

import { Command } from 'commander';
import { execSync } from 'child_process';
import { Verbosity, setVerbosity } from '../common/CommandLineOutput.js';
import { getRanges } from './DataDependenceTesting/RangeAnalysis.js';
// import { writeFile } from 'fs';
// import { ControlFlowGraph } from './DataDependenceTesting/ControlFlowGraph.js';

function runCompiler(program: Xml.Element) : Xml.Element {
    Xml.setNamespaces(program);

    setVerbosity(Verbosity.Silent);
    // const programDDG = DDT.run(program);

    getRanges(program.get('.//xmlns:function')!);
    // PLD.run(program, programDDG);  //
    // TODO: OUTPUT srcml option
    return program;
}

function outputFile(content: Xml.Element, inputFilePath: string) {
    const index = Math.max(inputFilePath.lastIndexOf('/'), 0);
    // const filePath = `${inputFilePath.substring(0, index + 1)}srcmp_${inputFilePath.substring(index + 1)}`;

    // writeFile(filePath, content.text, (err) => {if (err) console.error("ERROR")}); // TODO: ERROR

}

/**
 * Converts file contents to an Xml Object
 * * Program assumes that .xml files passed to program are srcml applied to one file
 * @param srcPath the path to the file as a string
 * @returns an xml object representing the file contents
 */
function getFileXml(srcPath: string) : Xml.Element {
    
    //TODO: Create SRCML interface for whole project to use
    const fileExtension = srcPath.substring(srcPath.lastIndexOf("."));
    
    if (fileExtension !== ".xml") {
        const buffer = execSync(`srcml --position ${srcPath}`, {timeout: 10000, maxBuffer: 1024 * 1024 * 10});
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

    program.name('src-cetus')
        .description('A srcML based optimizing compiler')
        .version('0.0.1')
        .argument('<input-files...>', 'The files to be compiled');

    program.parse();

    for (const inputFile of program.args[0].split(' ').filter((arg) => arg.length > 0)) {
        // try {
            outputFile(runCompiler(getFileXml(inputFile)), inputFile);
        // } catch (error) {
            // console.error(inputFile + ": " + error.message + "(" + error.name + ")");
        // }
    }

    return 0;
}


main();
