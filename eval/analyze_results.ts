#!/usr/bin/env node

import { Dirent, opendirSync, writeFile } from "fs";
import * as Xml from '../src/common/Xml/Xml.js';
import { execSync } from "child_process";


function loopHasForPragma(loop: Xml.Element) : boolean {
    return Boolean(loop?.prevElement?.get('./omp:directive/omp:name[text() = "for"]'));
}

function processFile(original: Xml.Element, srcmp: Xml.Element) : string[][] {

    // ! THIS ASSUMES THAT CETUS AND SRCMP DO NOT ADD OR REMOVE LOOP

    const oLoops = <Xml.ForLoop[]> original.find('//xmlns:for');

    // const cLoops = <Xml.ForLoop[]> cetus.find('//xmlns:for');
    const sLoops = <Xml.ForLoop[]> srcmp.find('//xmlns:for');


    const data: string[][] = [];

    for (let i = 0; i < oLoops.length; i++) {
        const line: (number | boolean | string)[] = [oLoops[i].line, oLoops[i].col, oLoops[i].header.text,
            loopHasForPragma(oLoops[i]), false, loopHasForPragma(sLoops[i])];
            // if (line[3] && !line[5]) {
            //     if (sLoops[i].find('./descendant::xmlns:for').some((loop) => loopHasForPragma(loop))) {
            //         line[5] = 'NESTED';
            //     }
            // }
        data.push(line.map((val) => String(val)));
    }
    return data;
}

function main() {
    const originalDir = opendirSync('eval/benchmarks/NPB-c');
    let currFile: Dirent | null = originalDir.readSync();
    const data: string[][] = [['File Name', 'Line', 'Col', 'Loop Header', 'Original Parallelized', 'Cetus Parallelized', 'SrcMP Parallelized']];
    while (currFile) {
        let original: Xml.Element;
        // let cetus: Xml.Element;
        let srcmp: Xml.Element;
        try {
            const originalPath = `${originalDir.path}/${currFile.name}`;
            // const cetusPath = `${originalDir.path}/../cetus_output/${currFile.name}`;
            const srcmpPath = `${originalDir.path}/../srcmp_output/${currFile.name}`;
            original = Xml.parseXmlString(execSync(`srcml --position ${originalPath}`, {timeout: 10000, maxBuffer: 1024 * 1024 * 10}).toString());
            // cetus = Xml.parseXmlString(execSync(`srcml --position ${cetusPath}`, {timeout: 10000, maxBuffer: 1024 * 1024 * 10}).toString());
            srcmp = Xml.parseXmlString(execSync(`srcml --position ${srcmpPath}`, {timeout: 10000, maxBuffer: 1024 * 1024 * 10}).toString());
        } catch (error) {
            console.error(String(error));
            continue;
        }

        if (data.length === 1) Xml.setNamespaces(original);
        // data.push(processFile(original, cetus, srcmp));
        const lines = processFile(original, srcmp);
        for (const line of lines) {
            line.unshift(currFile.name);
            data.push(line);
        }
        currFile = originalDir.readSync();
    }

    writeFile('benchmarks/results.csv', data.map((line) => line.join(',')).join('\n'), (err) => {if (err) console.error("ERROR");});
    originalDir.closeSync();
}

main();