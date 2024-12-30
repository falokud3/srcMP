
import * as Xml from '../../common/Xml/Xml.js';
import { ArrayAccess } from './ArrayAccess.js';
import { ControlFlowGraph } from './ControlFlowGraph.js';

class SubscriptPair {
    private subscript1: Xml.Element;
    private subscript2: Xml.Element;

    private access1: ArrayAccess;
    private access2: ArrayAccess;

    private enclosingLoops: Xml.ForLoop[];

    constructor(subscript1: Xml.Element, subscript2: Xml.Element,
        access1: ArrayAccess, access2: ArrayAccess, loops: Xml.ForLoop[]) {
        this.subscript1 = subscript1;
        this.subscript2 = subscript2;
        this.access1 = access1;
        this.access2 = access2;
        this.enclosingLoops = loops;
    }

    public isReachable() : boolean {
        const stmt1 = this.access1.parentStatement;
        const stmt2 = this.access2.parentStatement;

        if (stmt1.equals(stmt2)) {
            return this.access1.getAccessType() === ArrayAccess.READ_ACCESS &&
                this.access2.getAccessType() === ArrayAccess.WRITE_ACCESS;
        } else {
            //TODO: use caching system
            const cfg = ControlFlowGraph.buildControlFlowGraph(this.enclosingLoops[this.enclosingLoops.length - 1]);
            return cfg.isReachable(stmt1, stmt2);
        }
    }


    public getComplexity() : number {
        let ret: number = 0;
        this.enclosingLoops.forEach((loop: Xml.ForLoop) => {
            const indexVar = loop.getLoopIndexVariableName()?.text ?? "";
            if (this.subscript1.containsName(indexVar) ||
                this.subscript2.containsName(indexVar)) ret += 1;
        });
        return ret;
    }

    public getEnclosingLoops() : Xml.ForLoop[] {
        return this.enclosingLoops;
    }

    public getSubscript1() : Xml.Element {
        return this.subscript1;
    }

    public getAccessLine(access: number) : number {
        if (access === 1) {
            return this.access1.parentStatement.line;
        } else {
            return this.access2.parentStatement.line;
        }

    }

    public getSubscript2() : Xml.Element {
        return this.subscript2;
    }

    public toString() : string {
        return `(${this.subscript1.text} ${this.subscript2.text})`;
    }
}

export {SubscriptPair};