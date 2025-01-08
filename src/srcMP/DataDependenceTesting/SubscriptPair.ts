
import * as Xml from '../../common/Xml/Xml.js';
import { ArrayAccess } from './ArrayAccess.js';
import { buildGraph } from './ControlFlowGraphBuilder.js';

class SubscriptPair {
    private static reachableCache: Map<string, boolean> =  new Map();

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

    /**
     * Returns true if the stmt2 can be reached from stmt2
     */
    public isReachable() : boolean {
        const stmt1 = this.access1.access.parentElement!;
        const stmt2 = this.access2.access.parentElement!;

        if (stmt1.equals(stmt2)) {
            return this.access1.access_type === ArrayAccess.READ_ACCESS &&
                this.access2.access_type === ArrayAccess.WRITE_ACCESS;
        } else {
            const cacheKey = this.serializePair();
            if (!SubscriptPair.reachableCache.has(cacheKey)) {
                const cfg = buildGraph(this.enclosingLoops[this.enclosingLoops.length - 1]);
                SubscriptPair.reachableCache.set(cacheKey, cfg.isReachable(stmt1, stmt2));
            }            
            return SubscriptPair.reachableCache.get(cacheKey)!;
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

    public getAccessLine(access: 1 | 2) : number {
        return access === 1 ? this.access1.access.parentElement!.line
            : this.access2.access.parentElement!.line;
    }

    public getSubscript2() : Xml.Element {
        return this.subscript2;
    }

    public toString() : string {
        return `(${this.subscript1.text} ${this.subscript2.text})`;
    }

    /**
     * Used to generate unique key for cache
     */
    private serializePair() : string {
        let ret = '';

        ret += `${this.access1.access.line}:${this.access1.access.col} ${this.access1.access.text}\n`;
        ret += `${this.access2.access.line}:${this.access2.access.col} ${this.access2.access.text}`;

        return ret;
    }
}

export {SubscriptPair};