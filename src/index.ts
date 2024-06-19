#!/usr/bin/env node
import * as Xml from './Xml/Xml.js'
import * as DDT from './DataDependenceTesting/DataDependenceTestingPass.js'

import { Command } from 'commander'
import { execSync } from 'child_process';
import * as PLD from './ParallelizableLoopDetection/ParallelizableLoopDetectionPass.js';
import { Verbosity, setVerbosity } from './CommandLineOutput.js';



function runCompiler(program: Xml.Element) {
    // TODO: PARSE unit tag for namespaces
    setVerbosity(Verbosity.Internal);
    const programDDG = DDT.run(program);
    // PLD.run(program, programDDG);
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
        const buffer = execSync(`srcml ${srcPath}`, {timeout: 10000});
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

    for (const inputFile of program.args) {
        // try {
            runCompiler(getFileXml(inputFile));
        // } catch (error) {
            // console.error(inputFile + ": " + error.message + "(" + error.name + ")");
            return 1;
        // }
    }

    return 0;
}


main();
