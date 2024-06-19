import { DataDependenceGraph } from "../DataDependenceTesting/DataDependenceGraph";
import { Direction } from "../DataDependenceTesting/DependenceVector";
import { extractOutermostDependenceTestEligibleLoops } from "../DataDependenceTesting/Eligibility";
import { collectScalarDependencies } from "../DataDependenceTesting/ScalarDependenceTest";
import * as Xml from '../Xml/Xml.js'


export function run(program: Xml.Element, programDDG: DataDependenceGraph) {
    const loops = program.find('//xmlns:for') as Xml.ForLoop[];
    for (const loop of loops) {
        parallelizeLoopNest(loop, programDDG);
    }
}

function parallelizeLoopNest(loop: Xml.ForLoop, ddg: DataDependenceGraph) {
    const elligibleLoops = extractOutermostDependenceTestEligibleLoops(loop);
    let isParallel: boolean;

    for (const elligibleLoop of elligibleLoops) {
        const nestDDG = ddg.getLoopSubGraph(elligibleLoop);

        const scheduled: Xml.ForLoop[] = [];
        for (const nestedLoop of elligibleLoop.getInnerLoopNest()) {
            let hasScheduledOuterLoop = scheduled.some((value) => {
                return Xml.isAncestorOf(value, nestedLoop);
            });
            if (hasScheduledOuterLoop) continue;

            isParallel = true;
            if (nestedLoop.contains('.//xmlns:break')){ 
                isParallel = false;
                continue;
            }

            if (collectScalarDependencies(nestedLoop).size !== 0) {
                isParallel = false;
                continue;
            }

            const arrayDeps = new Set<string>();
            for (const arc of nestDDG.arcs) {
                const dir = arc.dependenceVector.getDirection(nestedLoop);

                if (dir !== undefined || dir === Direction.equal || dir === Direction.nil) continue;

                const sourceSymbol = arc.source.arrayName;
                const sinkSymbol = arc.sink.arrayName;

                // TODO: Private or Reduction
                const serialize = (sourceSymbol === sinkSymbol);

                if (serialize) {
                    isParallel = false;
                    arrayDeps.add(sourceSymbol);
                    if (dir !== Direction.any) nestDDG.removeArc(arc);
                }

            }

            if (isParallel) {
                if (!hasScheduledOuterLoop) scheduled.push(nestedLoop);
            }

        }
    }
}