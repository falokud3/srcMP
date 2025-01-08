/**
 * Interface for outputting messages to the console. Controls the verbosity
 * global variable
 */

import wrapAnsi from 'wrap-ansi';
import chalk from "chalk";
import * as Xml from './Xml/Xml.js';

export interface CLIMessage  {
    
    /**
     * @returns the message formatted as it would be on the command line
     */
    format: (verbosity: Verbosity) => string;
}

/**
 * Uses the chalk library to output compiler messages 
 */
export function output(...messages: CLIMessage[]) {
    for (const message of messages) {
        const formattedMessage = message.format(verbositySetting);
        if (formattedMessage.length === 0) continue;
        log(message.format(verbositySetting), verbositySetting);
    }
}

/**
 * Function for outputing and 
 * @param message The message to be output
 * @param verbosity Verbosity level associated with the message
 * @returns 
 */
export function log(message: string, verbosity: Verbosity = Verbosity.Basic) {
    if (verbosity > verbositySetting) return;
    console.log(wrapAnsi(message, 80, {hard: true}));
}

/**
 * Shorthand for yellow message at the Basic verboisty level
 * @param message 
 */
export function warn(message: string) {
    log(chalk.yellow(message));
}

/**
 * Shorthand for red message at the Silent verboisty level
 * @param message 
 */
export function error(message: string) {
    log(chalk.red(message), Verbosity.Silent);
}

export const Verbosity = {
    Silent: 0,
    Basic: 1,
    Internal: 2,
} as const;
export type Verbosity = typeof Verbosity[keyof typeof Verbosity];

// Global verbosity setting
let verbositySetting : Verbosity = Verbosity.Basic;

// sets the global verboisty setting
export const setVerbosity = (verbosity: Verbosity) => {verbositySetting = verbosity;};

/**
 * Functions useful for generating compiler messages
 */
export const errorStart = (numIssues: number) =>  numIssues !== 0 ? 'And the' : 'Because the';
export const numFormat = (amount: number, word: string) => amount === 1 ? `1 ${word}` 
   : `${amount} ${word}s`;
export const examples = (arr: Xml.Element[], count: number = 3) => {
    let ret = '';
    for (let i = 0; i < arr.length && i < count; i++) {
       const el = arr[i];
       ret += `${el.line}:${el.col}|${el.text}\n`;
    }
    return ret;
 };
