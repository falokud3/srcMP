import { ArrayAccess, ArrayAccessType } from "./ArrayAccess.js";
import { DependenceVector } from "./DependenceVector.js";


export class DataDependenceGraph {

    private dependenceArcs: Arc[];

    public constructor() {
        this.dependenceArcs = [];
    }

    public addAllArcs(other: DataDependenceGraph) : void {
        for (const arc of other.dependenceArcs) {
            this.addArc(arc);
        }
    }

    public addArc(arc: Arc) : void {
        if (!arc.dependenceVector.valid) return;

        const ddgHasArc = this.dependenceArcs.some((value: Arc) => {
            return arc.equals(value);
        });

        if (!ddgHasArc) this.dependenceArcs.push(arc)
    }

}

type Dependence = "Flow" | "Anti" | "Ouptut" | "Input";

export class Arc {
    private source: ArrayAccess;
    private sink: ArrayAccess;
    private dependenceType: Dependence;
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

        this.dependenceType = getDependenceType(this.source.getAccessType(), this.sink.getAccessType());
    }

    public equals(other: Arc) : boolean {
        return this.source.equals(other.source) 
            && this.sink.equals(other.sink)
            && this.dependenceType === other.dependenceType
            && this.dependenceVector.equals(other.dependenceVector);
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