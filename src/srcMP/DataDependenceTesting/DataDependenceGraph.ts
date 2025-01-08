import { ArrayAccess, ArrayAccessType } from "./ArrayAccess.js";
import { DependenceVector } from "./DependenceVector.js";
import * as Xml from '../../common/Xml/Xml.js';

export class DataDependenceGraph {

    private _arcs: Arc[];

    public constructor() {
        this._arcs = [];
    }

    public get arcs() : Arc[] {
        return this._arcs;
    }

    public addArcs(...arcs: Arc[]) : void {
        for (const arc of arcs) {
            if (!arc.dependenceVector.valid) continue;

            const ddgHasArc = this._arcs.some((value: Arc) => {
                return arc.equals(value);
            });
            if (!ddgHasArc) this._arcs.push(arc);
        }
    }

    public removeArc(other: Arc) : void {
        const index = this._arcs.findIndex((arc) => arc.equals(other));
        if (index !== -1) this._arcs.splice(index, 1);
    }

    /**
     * Returns a DDG consisting of all the arcs that belong to the input loop
     */
    public getLoopSubGraph(loop: Xml.ForLoop) : DataDependenceGraph {
        const loopGraph = new DataDependenceGraph();
        for (const arc of this._arcs) {
            if (arc.belongsToLoop(loop)) loopGraph.addArcs(arc);
        }
        return loopGraph;
    }

    /**
     * Outputs the Graph in DOT Language, which can be easily vizualized
     * through Graphviz
     */
    public toString() : string {
        let nodesString = '';
        let edgesString = '';
        let id = 1;
        const map = new Map<string, string>();

        for (const arc of this.arcs) {
            const source = `${arc.source.access.line}:${arc.source.access.col}|${arc.source.toString()}`;
            const sink = `${arc.sink.access.line}:${arc.sink.access.col}|${arc.sink.toString()}`;
            if (!map.has(source)) {
                map.set(source, `node${id}`);
                nodesString += `node${id} [label="${arc.source.access.line}:${arc.source.access.col}\\n${arc.source.access.text}\\n${arc.source.access_type}"]\n`;
                id += 1;
            }
            if (!map.has(sink)) {
                map.set(sink, `node${id}`);
                nodesString += `node${id} [label="${arc.sink.access.line}:${arc.sink.access.col}\\n${arc.sink.access.text}\\n${arc.sink.access_type}"]\n`;
                id += 1;
            }
            edgesString += `${map.get(source)} -> ${map.get(sink)} [label="${arc.dependenceVector.toString()}"]\n`;
        }

        return `digraph {
${nodesString}${edgesString}}`;
    }

}

type Dependence = "Flow" | "Anti" | "Ouptut" | "Input";

export class Arc {
    readonly source: ArrayAccess;
    readonly sink: ArrayAccess;
    readonly dependenceType: Dependence;
    public dependenceVector: DependenceVector;

    public constructor(expr1: ArrayAccess, expr2: ArrayAccess, depVector: DependenceVector) {
        if (depVector.isPlausibleVector) {
            this.source = expr1;
            this.sink = expr2;
            this.dependenceVector = depVector;
        } else {
            this.source = expr2;
            this.sink = expr1;
            this.dependenceVector = depVector.reverseVector;
        }

        this.dependenceType = getDependenceType(this.source.access_type, this.sink.access_type);
    }

    public equals(other: Arc) : boolean {
        return this.source.equals(other.source) 
            && this.sink.equals(other.sink)
            && this.dependenceType === other.dependenceType
            && this.dependenceVector.equals(other.dependenceVector);
    }

    public belongsToLoop(loop: Xml.ForLoop) : boolean {

        const sourceBelongs = this.source.enclosingLoop
            ?.getEnclosingLoopNest().some((sloop) => loop.equals(sloop));

        const sinkBelongs = this.sink.enclosingLoop
            ?.getEnclosingLoopNest().some((sloop) => loop.equals(sloop));

        return (sourceBelongs && sinkBelongs) ?? false;
    }

}

function getDependenceType(sourceType: ArrayAccessType, sinkType: ArrayAccessType) : Dependence {
    if (sourceType === ArrayAccess.WRITE_ACCESS 
        && sinkType === ArrayAccess.WRITE_ACCESS) {
        return "Ouptut";
    } else if (sourceType === ArrayAccess.WRITE_ACCESS 
        && sinkType === ArrayAccess.READ_ACCESS) {
        return "Flow";
    } else if (sourceType === ArrayAccess.READ_ACCESS 
        && sinkType === ArrayAccess.WRITE_ACCESS) {
        return "Anti";
    } else {
        return "Input";
    }
}