/**
 * Control Flow Graph used to determine if an array access is reachable for
 * dependence analysis. Can also be used for range analysis.
 */

import * as Xml from '../../common/Xml/Xml.js';
import { buildGraph } from './ControlFlowGraphBuilder.js';

export class ControlFlowGraph {

    public nodes: ControlFlowNode[];

    public constructor() {
        this.nodes = [];
    }

    public addNode(node: ControlFlowNode) {
        if (!this.nodes.includes(node)) this.nodes.push(node);
    }

    /**
     * Outputs the Graph in DOT Language, which can be easily vizualized
     * through Graphviz
     */
    public toString() {
        let ret: string = "digraph {\n";

        for (const node of this.nodes) {
            ret += node.nodeInfoToString() + "\n";
        }

        for (const node of this.nodes) {
            const edges = node.nodeEdgesToString();
            if (edges.length > 0) ret += edges  + "\n";
        }

        ret += "}";
        return ret;
    }

    /**
     * Uses DFS to add all nodes connected to the input node to the CFG's
     * nodes array.
     * @param node node to traverse and add to the graph
     * @param visited nodes that have already been added or should not be added to the graph
     */
    public addAllNodes(node: ControlFlowNode, visited: number[]) : void {
        this.addNode(node);
        for (const adjNode of node.edges) {
            this.addNode(adjNode);
            if (!visited.includes(adjNode.num)) {
                visited.push(adjNode.num);
                this.addAllNodes(adjNode, visited);
            }
        }
    }

    // helper method
    private static rTopologicalSort(node: ControlFlowNode, visited: number[], newOrder: ControlFlowNode[]) {
        visited.push(node.num);
        for (const child of node.edges) {
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

    /**
     * Returns true if stmt1 can reach stmt2. Attempts to do this by finding
     * the nodes that correlate to the XML elements. It then uses topologicalSort
     * to determine reachability
     * ! NOT A VERY ROBUST APPROACH AND IS LIKELY TO HAVE ISSUES!
     */
    public isReachable(stmt1: Xml.Element, stmt2: Xml.Element) : boolean {
        
        let from: ControlFlowNode | null = null;
        let to: ControlFlowNode | null = null;

        const s1 = stmt1.parentElement;
        const s2 = stmt2.parentElement;

        if (!s1 || !s2) return true;

        for (const node of this.nodes) {
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
}

export class ControlFlowNode {
    type: 'START' | 'NODE' | 'END' = 'NODE';
    private static maxID: number = 1;

    /**
     * variables used in the build process that should not be used elsewhere. 
     * Should be refacotred to be in the ControlFlowGraphBUilder file
    */
    private _tail: ControlFlowNode[] = [];
    private connectable: boolean = true;

    public xml: Xml.Element;
    

    private _edges: ControlFlowNode[] = [];

    private order: number = -1; // topological order

    private _idNum: number; // for toDot ouptut 

    public constructor(data: Xml.Element) {
        this.xml = data;
        this._idNum = ControlFlowNode.maxID++;
    }

    // this extends tail
    public addAdjacent(node: ControlFlowNode) {
        if (!this._edges.includes(node)) this._edges.push(node);
        for (const tailNode of node.getTail()) {
            if (!this._tail.includes(tailNode)) this._tail.push(tailNode);
        }
    }

    // connecting tip to tail
    public static connectNodes(from: ControlFlowNode, to: ControlFlowNode, updateTail: boolean = true) {
        // spread fixes weird bug were tail would grow when adding nodes
        const fromTail = [...from.getTail()];

        for (const tailNode of fromTail) {
            if (!tailNode.connectable) continue;
            tailNode.addAdjacent(to);
        }
        if (updateTail) from._tail = to.getTail();
    }

    public getLeafNodes() : ControlFlowNode[] {
        const tailNodes: ControlFlowNode[] = [];
        this.rGetLeafNodes(tailNodes, []);
        return tailNodes;
    }

    // DFS
    private rGetLeafNodes(leafNodes: ControlFlowNode[], visited: number[]) {
        if (visited.includes(this.num)) {
            return; 
        } else {
            visited.push(this.num);
        }

        if (this._edges.length === 0) leafNodes.push(this);
        for (const vertex of this._edges) { 
            vertex.rGetLeafNodes(leafNodes, visited);
        }
    }


    // tail are all the nodes without outgoing edge
    public getTail() : ControlFlowNode[] {
        return this._tail.length > 0 ? this._tail : [this];
    }

    public setTail(newTail: ControlFlowNode[]) : void {
        this._tail = newTail;
    }

    public addTailNode(node: ControlFlowNode) : void {
        this._tail.push(node);
    }

    public popTailNode() : void {
        this._tail.pop();
    }

    // Sets whether new edges should be added to this node
    public setConnectable(val: boolean) : void {
        this.connectable = val;
    }

    public get edges() : ControlFlowNode[] {
        return this._edges;
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

    public nodeInfoToString() : string {
        let ret = "";
        ret += `node${this._idNum} [label="#${this.order}\\n<${this.xml.name}>\\n`;
        ret += `${this.xml.text.trim()}\\n`;
        return ret;
    }

    public nodeEdgesToString() : string {
        if (this._edges.length === 0) return "";

        let ret = `node${this._idNum}->{ `;

        for (const adj of this._edges) {
            ret += `node${adj._idNum} `;
        }

        ret += "};";
        return ret;
    }

}