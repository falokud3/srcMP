// import chalk from 'chalk';
import wrapAnsi from 'wrap-ansi';
import chalk from "chalk";
import * as Xml from './Xml/Xml.js';

export interface CLIMessage  {
    
    /**
     * 
     * @returns the message formatted as it would be on the command line
     */
    format: (verbosity: Verbosity) => string;
}

/**
 * Uses the chalk library to output compiler messages 
 */
export function output(...messages: CLIMessage[]) {
    if (verbositySetting === Verbosity.Silent) return;
    for (const message of messages) {
        const cliOutput = wrapAnsi(message.format(verbositySetting), 80, {hard: true});
        if (cliOutput.length !== 0) console.log(`${cliOutput}`);
    }
}

export function log(message: string) {
    if (verbositySetting === Verbosity.Silent) return;
    console.log(wrapAnsi(message, 80, {hard: true}));
}

export function warn(message: string) {
    if (verbositySetting === Verbosity.Silent) return;
    console.warn(wrapAnsi(chalk.yellow(message), 80, {hard: true}));
}

export function error(message: string) {
    console.warn(wrapAnsi(chalk.red(message), 80, {hard: true}));
}

export const Verbosity = {
    Silent: 0,
    Simple: 1,
    Complex: 2,
    Internal: 3,
} as const;

export type Verbosity = typeof Verbosity[keyof typeof Verbosity];

let verbositySetting : Verbosity = Verbosity.Simple;

export const setVerbosity = (verbosity: Verbosity) => {verbositySetting = verbosity;};

export const errorStart = (numIssues: number) =>  numIssues !== 0 ? 'And the' : 'Because the';
export const numFormat = (amount: number, word: string) => amount === 1 ? `1 ${word}` 
   : `${amount} ${word}s`;
export const examples = (arr: Xml.Element[], count: number = 3) => {
    let ret = '';
    for (let i = 0; i < arr.length && count < 3; i++) {
       const el = arr[i];
       ret += `\t${el.line}:${el.col}| ${el.text}`;
    }
    return ret;
 };
