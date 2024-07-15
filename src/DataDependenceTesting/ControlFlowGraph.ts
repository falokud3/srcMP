import * as Xml from '../Facades/Xml/Xml.js'
import { buildGraph } from './ControlFlowGraphBuilder.js';
import { RangeDomain, Range } from './RangeDomain.js';

// ! doesn't support labels and goto statements
export class ControlFlowGraph {

    public nodes: ControlFlowNode[];

    public constructor() {
        this.nodes = [];
    }

    public addNode(node: ControlFlowNode) {
        if (!this.nodes.includes(node)) this.nodes.push(node);
    }

    public toString() {
        let ret: string = "digraph G {\n"

        for (const node of this.nodes) {
            ret += node.nodeInfoToString() + "\n";
        }

        for (const node of this.nodes) {
            const edges = node.nodeEdgesToString();
            if (edges.length > 0) ret += edges  + "\n";
        }

        ret += "}"
        return ret;
    }

    public addAllNodes(node: ControlFlowNode, visited: number[]) : void {
        this.addNode(node);
        for (const adjNode of node.adjacents) {
            this.addNode(adjNode);
            if (!visited.includes(adjNode.num)) {
                visited.push(adjNode.num);
                this.addAllNodes(adjNode, visited);
            }
        }
    }

    private static rTopologicalSort(node: ControlFlowNode, visited: number[], newOrder: ControlFlowNode[]) {
        visited.push(node.num);
        for (const child of node.adjacents) {
            if (!visited.includes(child.num)) {
                this.rTopologicalSort(child, visited, newOrder);
            }
        }
        newOrder.splice(0, 0, node);
    }

    public topologicalSort(root: ControlFlowNode = this.nodes[0]) {
        const newOrder: ControlFlowNode[] = [];
        for (const node of this.nodes) {
            node.newOrder = -1;
        }

        ControlFlowGraph.rTopologicalSort(root, [], newOrder);

        for (let i = 0; i < newOrder.length; i++) {
            newOrder[i].newOrder = i; 
        }
        // NOTE: Instead of creating a new array can sort on newOrder
        this.nodes = newOrder;
    }

    public isReachable(stmt1: Xml.Element, stmt2: Xml.Element) : boolean {
        
        let from: ControlFlowNode | null = null;
        let to: ControlFlowNode | null = null;

        // ! HACKY AND BAD
        const s1 = stmt1.parentElement
        const s2 = stmt2.parentElement

        if (!s1 || !s2) return true;

        for (const node of this.nodes) {
            // ! POTENTIAL ISSUE WITH IDENTICAL STATEMENTS
            if (node.xml.text === s1.text) from = node;

            if (node.xml.text === s2.text) to = node;
        }

        if (!from || !to) throw new Error("Statement not found in CFG!");

        this.topologicalSort(from);

        return to.getOrder() > -1;
    }

    public static buildControlFlowGraph(src: Xml.Element) : ControlFlowGraph {
        return buildGraph(src);
    }

    private substituteAll(src: Xml.Element) : Xml.Element {
        throw new Error("NOT IMPLEMEnteD")
    }

    public getRangeMap(node: Xml.Element) : Map<string, RangeDomain> {
        const ret = new Map<string, RangeDomain>();

        for (const node of this.nodes) {
            const ranges = node.getRanges();
            ret.set(`${node.xml.line} ${node.xml.text}`, ranges);
            for (const subkey of ControlFlowGraph.getSubKeys(node.xml)) {
                ret.set(subkey, ranges);
            }
        }

        return ret;
    }

    public static getSubKeys(node: Xml.Element) : string[] {
        let ret: string[] = [];

        for (const child of node.childElements) {
            ret.push(`${child.line} ${child.text}`);
            ret.push(...ControlFlowGraph.getSubKeys(child));
        }

        return ret;
    }


    public static getIndexOfFirstNodeTopographically(list: ControlFlowNode[]) : number {
        let index: number = 0;
        for (let i = 1; i < list.length; i++) {
            if (list[i].getOrder() < list[index].getOrder()) {
                index = i;
            }
        }
        return index;
    }
}



export class ControlFlowNode {
    type: 'START' | 'NODE' | 'END' = 'NODE';
    private static maxID: number = 1;

    private _tail: ControlFlowNode[] = []; // used exclusively for build process then deleted
    private connectable: boolean = true;

    public xml: Xml.Element;
    

    private outEdges: ControlFlowNode[] = [];
    private inEdges: ControlFlowNode[] = [];

    private order: number = -1; // topological order

    private _idNum: number; // for toDot ouptut 

    // TODO: Refactor into one range object 
    // [in, curr, out]
    public inRanges: Map<ControlFlowNode, RangeDomain> = new Map<ControlFlowNode, RangeDomain>();
    private currRanges: RangeDomain = new RangeDomain(); // TODO: Remove
    public outRanges: Map<ControlFlowNode, RangeDomain> = new Map<ControlFlowNode, RangeDomain>();

    // TODO: Somehow move all this RangeAnalysis gunk out
    private backedge: boolean | undefined;
    public loopVariants: Set<Xml.Element> | undefined = undefined;

    public constructor(data: Xml.Element) {
        this.xml = data
        this._idNum = ControlFlowNode.maxID++;
    }

    // this extending tail
    public addAdjacent(node: ControlFlowNode) {
        if (!this.outEdges.includes(node)) this.outEdges.push(node);
        if (!node.inEdges.includes(this)) node.inEdges.push(this);
        for (const tailNode of node.getTail()) {
            if (!this._tail.includes(tailNode)) this._tail.push(tailNode);
        }
    }

    // connecting tip to tail
    public static connectNodes(from: ControlFlowNode, to: ControlFlowNode, updateTail: boolean = true) {
        // spread fixes weird bug were tail would grow when adding nodes
        const fromTail = [...from.getTail()]

        for (const tailNode of fromTail) {
            if (!tailNode.connectable) continue;
            tailNode.addAdjacent(to);
        }
        if (updateTail) from._tail = to.getTail();
    }

    public get tail() : ControlFlowNode[] {
        // TODO: Include Manual Overide for loops etc
        const tailNodes: ControlFlowNode[] = [];
        this.rGetTail(tailNodes, []);
        return tailNodes;
    }

    private rGetTail(tailNodes: ControlFlowNode[], visited: number[]) {
        // TODO: Optional worry about adding duplicates to tailNodes
        if (visited.includes(this.num)) {
            return; 
        } else {
            visited.push(this.num);
        }

        if (this.outEdges.length === 0) tailNodes.push(this);
        for (const vertex of this.outEdges) { 
            vertex.rGetTail(tailNodes, visited);
        }
    }



    // tail are all the nodes without outgoing edge
    public getTail() : ControlFlowNode[] {
        // TODO: utilize this.tail()
        return this._tail.length > 0 ? this._tail : [this];
    }

    public setTail(newTail: ControlFlowNode[]) : void {
        this._tail = newTail;
    }

    public setConnectable(val: boolean) : void {
        this.connectable = val;
    }

    public addTailNode(node: ControlFlowNode) : void {
        this._tail.push(node);
    }

    public popTailNode() : void {
        this._tail.pop();
    }

    public get adjacents() : ControlFlowNode[] {
        return this.outEdges;
    }

    public get num() : number {
        return this._idNum;
    }

    public getOrder() : number {
        return this.order;
    }

    public set newOrder(order: number) {
        this.order = order;
    }

    public get preds() : ControlFlowNode[] {
        return this.inEdges;
    }

    public get succs() : ControlFlowNode[] {
        return this.outEdges;
    }
    
    public getRanges() : RangeDomain {
        return this.currRanges;
    }

    public setRanges(newRanges: RangeDomain) : void {
        this.currRanges = newRanges;
    }

    public hasBackedge() : boolean {
        return this.backedge ?? this.setBackedge();
    }

    public setBackedge() : boolean {
        for (const pred of this.inEdges) {
            if (this.order < pred.order) {
                this.backedge = true;
                return true;
            }
        }
        return false;
    }

    public toString() : string {
        let ret: string = "";
        ret += this._idNum + " " + this.xml.name;
        return ret;
    }

    public nodeInfoToString() : string {
        let ret = "";
        ret += `node${this._idNum} [label="#${this.order}\\n<${this.xml.name}>\\n`;
        ret += `${this.xml.text.trim()}\\n`;
        ret += `${this.currRanges.toString()}\\n"]`
        return ret;
    }

    public nodeEdgesToString() : string {
        if (this.outEdges.length == 0) return "";

        let ret = `node${this._idNum}->{ `;

        for (const adj of this.outEdges) {
            ret += `node${adj._idNum} `;
        }

        ret += "};"
        return ret;
    }

}