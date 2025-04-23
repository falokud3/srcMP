import chalk from "chalk";
import { CLIMessage, Verbosity, examples, output, log } from '../../common/CommandLineOutput.js';
import { DataDependenceGraph } from "../DataDependenceTesting/DataDependenceGraph.js";
import { Direction } from "../DataDependenceTesting/DependenceVector.js";
import { extractOutermostDependenceTestEligibleLoops } from "../DataDependenceTesting/Eligibility.js";
import { collectScalarDependencies } from "./ScalarDependenceTest.js";
import * as Xml from '../../common/Xml/Xml.js';
import { createXml } from "../../common/srcml.js";


export function run(program: Xml.Element, programDDG: DataDependenceGraph) {
    const startTime = performance.now();
    log('[Parallelizable Loop Detection] Start', Verbosity.Internal);

    const outerLoops = program.find('//xmlns:for[count(ancestor::xmlns:for)=0]') as Xml.ForLoop[];
    const messages: ParallelizableStatus[] = [];
    for (const outerLoop of outerLoops) {
        messages.push(...parallelizeLoopNest(outerLoop, programDDG));
    }
    // removeExistingPragmas(program);
    output(...messages);    
    // insertPragmas(program, messages);

    const endTime = performance.now();
    log(`[Parallelizable Loop Detection] End -- Duration: ${(endTime - startTime).toFixed(3)}ms`, Verbosity.Internal);
}

function removeExistingPragmas(program: Xml.Element) : void {
    const language = program.get("/xmlns:unit")!.getAttribute("language")!;
    let ompPragmas: Xml.Element[] | undefined;
    if (language === "C++") {
        ompPragmas = program.find('//cpp:pragma[./omp:directive]');
    }
    // py2srcml skips comments
    
    for (const ompPragma of (ompPragmas ?? [])) {
        ompPragma.remove();
    }
}

function insertPragmas(program: Xml.Element, analyzedLoops: ParallelizableStatus[]) : void {
    const parallelizedLoops = analyzedLoops.filter((loopStatus) => loopStatus.isParallelizable && !loopStatus.parallelizableOuterLoop);
    const language = program.get("/xmlns:unit")!.getAttribute("language")!;
    const pragmaXML = createXml('#pragma omp parallel for', language);
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

            if (message.containsBreak || message.containsScalarDependencies) continue;

            const arrayDeps = new Set<Xml.Element>();
            for (const arc of nestDDG.arcs) {
                const dir = arc.dependenceVector.getDirection(nestedLoop);

                if (dir === undefined || dir === Direction.equal || dir === Direction.nil) continue;

                arrayDeps.add(arc.source.access.parentElement!);
                if (dir !== Direction.any) nestDDG.removeArc(arc);

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
        let output = '';

        if (this.isParallelizable) {
            output = `${chalk.green('Parallelizable')}: ${this.loop.line}:${this.loop.col}|${this.loop.header.text} `;

            if (this.parallelizableOuterLoop) {
                output += `but it is ${chalk.yellow('nested inside a parallelizable loop')}.`;
            } else if (this.dangerousJumps.length > 0) {
                output += `but contains ${chalk.yellow('potentially dangerous jump statements')}.`;
            }
        } else {
            output = `${chalk.red('Unparallelizable')}: ${this.loop.line}:${this.loop.col}|${this.loop.header.text} `;

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

    get internalFormat() : string {
        let body: string;      
  
        if (this.isParallelizable) {
            // TODO: Private, Reduction Variables

            body = `${chalk.green('Parallelizable')}: ${this.loop.line}:${this.loop.col}|${this.loop.header.text}\n`;
            if (this.parallelizableOuterLoop) {
                body += `${chalk.yellow(`but there is an enclosing parallelizable loop`)} ${this.parallelizableOuterLoop.line}:${this.parallelizableOuterLoop.col}|${this.parallelizableOuterLoop.header.text}. Parallelizing the outermost loop is typically most beneficial for performance.\n`;
            } 
            if (this.dangerousJumps.length > 0) {
                body += `but the loop contains ${chalk.yellow('potentially dangerous jump statements')} which may cause the parallelized loop to behave unexpectedly.\n`;
                body += examples(this.dangerousJumps);
            }
        } else {
            body = `${chalk.red('Unparallelizable')}: ${this.loop.line}:${this.loop.col}|${this.loop.header.text}\n`;
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
        return `${body.substring(0, body.length - 1)}`;
    }

    format(verbosity: Verbosity) : string {
        if (verbosity === Verbosity.Basic) {
            return this.simpleFormat;
        } else if (verbosity === Verbosity.Internal) {
            return this.internalFormat;
        } else {
            return '';
        }
    }
    

}