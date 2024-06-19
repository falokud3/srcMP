import chalk from 'chalk';

const log = console.log;

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
export function output(message: CLIMessage) {
    const cliOutput = message.format(verbositySetting);
    if (cliOutput.length === 0) return;
    console.log(`${cliOutput}`);
}

export const Verbosity = {
    Silent: 0,
    Simple: 1,
    Complex: 2,
    Internal: 3,
} as const;

export type Verbosity = typeof Verbosity[keyof typeof Verbosity];

let verbositySetting : Verbosity = Verbosity.Simple;

export const setVerbosity = (verbosity: Verbosity) => {verbositySetting = verbosity};

