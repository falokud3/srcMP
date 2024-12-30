import chalk from "chalk";
import { CLIMessage, Verbosity, examples, output } from '../../common/CommandLineOutput.js';
import { DataDependenceGraph } from "../DataDependenceTesting/DataDependenceGraph.js";
import { Direction } from "../DataDependenceTesting/DependenceVector.js";
import { extractOutermostDependenceTestEligibleLoops } from "../DataDependenceTesting/Eligibility.js";
import { collectScalarDependencies } from "../DataDependenceTesting/ScalarDependenceTest.js";
import * as Xml from '../../common/Xml/Xml.js';
import { createXml } from "../../common/srcml.js";


export function run(program: Xml.Element, programDDG: DataDependenceGraph) {
    const outerLoops = program.find('//xmlns:for[count(ancestor::xmlns:for)=0]') as Xml.ForLoop[];
    const messages: ParallelizableStatus[] = [];
    for (const outerLoop of outerLoops) {
        messages.push(...parallelizeLoopNest(outerLoop, programDDG));
    }
    removeExistingPragmas(program);
    output(...messages);    
    insertPragmas(messages);

}

function removeExistingPragmas(program: Xml.Element) : void {
    const ompPragmas = program.find('//cpp:pragma[./omp:directive]');
    for (const ompPragma of ompPragmas) {
        ompPragma.remove();
    }
}

function insertPragmas(analyzedLoops: ParallelizableStatus[]) : void {
    const parallelizedLoops = analyzedLoops.filter((loopStatus) => loopStatus.isParallelizable && !loopStatus.parallelizableOuterLoop);
    const pragmaXML = createXml('#pragma omp parallel for', 'C++')!;
    for (const loopStatus of parallelizedLoops) {
        loopStatus.loop.insertBefore(pragmaXML.copy());
        loopStatus.loop.insertBefore( loopStatus.loop.domNode.ownerDocument.createTextNode('\n'));
   }
}

function parallelizeLoopNest(outerLoop: Xml.ForLoop, ddg: DataDependenceGraph) : ParallelizableStatus[] {
    const [elligibleLoops,] = extractOutermostDependenceTestEligibleLoops(outerLoop);
    const messages: ParallelizableStatus[] = [];

    for (const elligibleLoop of elligibleLoops) {
        const nestDDG = ddg.getLoopSubGraph(elligibleLoop);

        const scheduled: Xml.ForLoop[] = [];
        for (const nestedLoop of elligibleLoop.getInnerLoopNest()) {
            const message = new ParallelizableStatus(nestedLoop);
            messages.push(message);
            
            message.parallelizableOuterLoop = scheduled.find((value) => {
                return Xml.isAncestorOf(value, nestedLoop);
            });

            // if (hasScheduledOuterLoop) continue;

            if (message.containsBreak || message.containsScalarDependencies) continue;

            const arrayDeps = new Set<Xml.Element>();
            for (const arc of nestDDG.arcs) {
                const dir = arc.dependenceVector.getDirection(nestedLoop);

                if (dir !== undefined || dir === Direction.equal || dir === Direction.nil) continue;

                const sourceSymbol = arc.source.arrayName;
                const sinkSymbol = arc.sink.arrayName;

                // TODO: Private or Reduction
                const serialize = (sourceSymbol !== sinkSymbol);

                if (serialize) {
                    arrayDeps.add(arc.source.access);
                    if (dir !== Direction.any) nestDDG.removeArc(arc);
                }

            }

            message.arrayDeps = arrayDeps;

            if (message.isParallelizable 
                && !message.parallelizableOuterLoop) scheduled.push(nestedLoop);

        }
    }
    return messages;
}

export class ParallelizableStatus implements CLIMessage {

    loop: Xml.ForLoop;
    arrayDeps: Set<Xml.Element> = new Set<Xml.Element>();
    parallelizableOuterLoop: Xml.ForLoop | undefined;

    public constructor(loop: Xml.ForLoop) {
        this.loop = loop;
    }

    get isParallelizable() : boolean {
        return this.arrayDeps.size === 0 && !this.containsBreak && !this.containsScalarDependencies;
    }

    get containsBreak() : boolean {
        return this.loop.contains('.//xmlns:break');
    }

    get containsScalarDependencies() : boolean {
        return collectScalarDependencies(this.loop).size !== 0;
    }

    get dangerousJumps() : Xml.Element[] {
        return this.loop.find('.//xmlns:return')
            .concat(this.loop.find('.//xmlns:label'))
            .concat(this.loop.find('.//xmlns:goto'));
    }

    get simpleFormat() : string {
        let output = `${this.loop.line}:${this.loop.col}| for${this.loop.header.text} `;

        if (this.isParallelizable) {
            output += `is ${chalk.green('parallelizable')} `;

            if (this.parallelizableOuterLoop) {
                output += `but it is ${chalk.yellow('nested inside a parallelizable loop')}.`;
            } else if (this.dangerousJumps.length > 0) {
                output += `but contains ${chalk.yellow('potentially dangerous jump statements')}.`;
            }
        } else {
            output += `is ${chalk.red('not parallelizable')} `;

            if (this.arrayDeps.size > 0) {
                output += 'due to loop carried array dependencies.';
            } else if (this.containsScalarDependencies) {
                output += 'due to scalar dependencies.';
            } else {
                output += 'because it contains a break statement.';
            }
        }

        return output;
    }

    get complexFormat() : string {
        let header: string, footer: string, body: string;
        const filename = this.loop.get('/xmlns:unit')?.getAttribute('filename') ?? '';
      
  
        if (this.isParallelizable) {
            // TODO: Private, Reduction Variables
            const paddingLength = 80 - (25 + 1 + filename.length);
            const padding = '+'.repeat(paddingLength > 0 ? paddingLength : 0);
            header = chalk.green(`++ PARALLELIZABLE LOOP ++${padding} ${filename}`);
            footer = chalk.green(`+`.repeat(80));

            body = `${this.loop.line}:${this.loop.col}| for${this.loop.header.text} is ${chalk.green('parallelizable')}.\n`;

            if (this.parallelizableOuterLoop) {
                body += `${chalk.yellow(`but there is an outer parallelizable for loop`)} ${this.parallelizableOuterLoop.header.text} at line ${this.parallelizableOuterLoop.line}, column ${this.parallelizableOuterLoop.col}. Parallelizing the outermost loop is typically most beneficial for performance.\n`;
            } 
            if (this.dangerousJumps.length > 0) {
                body += `but the loop contains ${chalk.yellow('potentially dangerous jump statements')} which may cause the parallelized loop to behave unexpectedly.\n`;
                body += examples(this.dangerousJumps);
            }
        } else {
            const paddingLength = 80 - (27 + 1 + filename.length);
            const padding = '-'.repeat(paddingLength > 0 ? paddingLength : 0);
            header = chalk.red(`-- UNPARALLELIZABLE LOOP --${padding} ${filename}`);
            footer = chalk.red(`-`.repeat(80));

            body = `${this.loop.line}:${this.loop.col}| for${this.loop.header.text} is ${chalk.red('not parallelizable')}.\n`;
            if (this.arrayDeps.size > 0) {
                body += 'due to loop carried array dependencies.';
                body += examples(Array.from(this.arrayDeps));
            }
            if (this.containsScalarDependencies) {
                body += 'due to scalar dependencies.';
                body += examples(Array.from(collectScalarDependencies(this.loop)));
            }
            if (this.containsBreak) {
                body += 'because it contains a break statement';
                body += examples(this.loop.find('.//xmlns:break'));
            }
        }
        return `${header}\n${body.substring(0, body.length - 1)}\n${footer}`;
    }

    format(verbosity: Verbosity) : string {
        if (verbosity === Verbosity.Basic) return this.simpleFormat;
        else return this.complexFormat;
    }
    

}