import { ArrayAccess } from "./ArrayAccess";
import { DependenceVector } from "./DependenceVector";


export class DataDependenceGraph {

    private dependenceArcs: Arc[];

    public constructor() {
        this.dependenceArcs = [];
    }

}

type Dependence = "Flow" | "Anti" | "Ouptut" | "Input";

class Arc {
    // private source: ArrayAccess;
    // private sink: ArrayAccess;
    // private dependenceType: Dependence;
    // private dependenceVector: DependenceVector;

}