import * as Xml from '../Xml/Xml.js'
import { RangeDomain, Range } from './RangeDomain.js';

// ! doesn't support labels and goto statements
export class ControlFlowGraph {

    public nodes: ControlFlowNode[];

    private static loopJumps: ControlFlowNode[] = [];
    private static labelNodes: Map<string, ControlFlowNode> = new Map<string, ControlFlowNode>();
    private static gotoJumps: ControlFlowNode[] = [];

    private constructor() {
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

    public isReachable(stmt1: Xml.XmlElement, stmt2: Xml.XmlElement) : boolean {
        
        let from: ControlFlowNode | null = null;
        let to: ControlFlowNode | null = null;

        // ! HACKY AND BAD
        const s1 = stmt1.parentElement
        const s2 = stmt2.parentElement

        if (!s1 || !s2) return false;

        for (const node of this.nodes) {
            // ! POTENTIAL ISSUE WITH IDENTICAL STATEMENTS
            if (node.xml.text === s1.text) from = node;

            if (node.xml.text === s2.text) to = node;
        }

        if (!from || !to) throw new Error("Statement not found in CFG!");

        this.topologicalSort(from);

        return to.getOrder() > -1;
    }

    public static buildControlFlowGraph(src: Xml.XmlElement) : ControlFlowGraph {
        const graph = new ControlFlowGraph();
        const newGraph: ControlFlowNode | null = ControlFlowGraph.buildGraph(src);
        if (newGraph) {
            graph.addAllNodes(newGraph, []);
        }
        return graph;
    }

    private substituteAll(src: Xml.XmlElement) : Xml.XmlElement {
        throw new Error("NOT IMPLEMEnteD")
        // for (const node of this.nodes) {
        //     // // ! NEED TO ESCAPE STRINGS IN .text

        //     const srcNodes = src.find(`.//xmlns:${node.xml.name}`, Xml.ns);
        //     let src_node: Xml.XmlElement | null = null
        //     for (const srcNode of srcNodes) {
        //         if (srcNode.text === node.xml.text) {
        //             src_node = node.xml;
        //             break;
        //         }
        //     }

        //     const var_ranges = node.getRanges()

        //     if (!src_node) return src;

        //     const src_vars = src_node.find(".//xmlns:name", Xml.ns);

        //     console.log(src_node.toString())
        //     for (const variable of src_vars) {
        //         const var_range = var_ranges.getRange(variable.text);
        //         if (var_range && var_range.isConstant) { // not LHS of Assignment

                    
        //             console.log("SUBSTITUTION")
        //             // variable.replace(replace_node);
        //         }
        //     }
        //     console.log(src_node.toString())
        //     console.log("=======")

        // }


    }

    private static buildGraph(src: Xml.XmlElement) : ControlFlowNode | null {
        let type: string = src.name;
        if (type === "function") {
            return ControlFlowGraph.buildFunction(src);
        } else if (type === "block") {
            return ControlFlowGraph.buildBlock(src);
        } else if (type === "unit") {
            return ControlFlowGraph.buildUnit(src);
        } else if (type === "condition") { 
            return new ControlFlowNode(src);
        } else if (type === "if_stmt") {
            return ControlFlowGraph.buildIf(src);
        } else if (type === "switch") {
            return ControlFlowGraph.buildSwitch(src);
        } else if (type === "while") {
            return ControlFlowGraph.buildWhile(src);
        } else if (type === "do") {
            return ControlFlowGraph.buildDo(src);
        } else if (type === "for") {
            return ControlFlowGraph.buildFor(src);
        } else if (type == "init") {
            return new ControlFlowNode(src);
        } else if (type == "incr") {
            return new ControlFlowNode(src);
        } else if (type === "break" || type === "continue") {
            const ret = new ControlFlowNode(src);
            ret.setConnectable(false);
            ControlFlowGraph.loopJumps.push(ret);
            return ret;
        } else if (type === "return") {
            const ret = new ControlFlowNode(src);
            ret.setConnectable(false);
            return ret;
        } else if (type === "case" || type === "default") {
            return ControlFlowGraph.buildCase(src);
        } else if (type.includes("stmt") || type === "expr" || type == "decl") {
            return new ControlFlowNode(src);
        } else if (type === "label") {
            return ControlFlowGraph.buildLabel(src);
        } else if (type === "goto") {
            return ControlFlowGraph.buildGoto(src);
        } else if (type === "comment") {
            return null;
        } else {
            console.error("Unexpected Element: " + type);
            return null;
        }
    }

    private static buildFunction(func: Xml.XmlElement) : ControlFlowNode | null {
        const block = func.get("./xmlns:block", Xml.ns);
        return ControlFlowGraph.buildBlock(block!);
    }

    private static buildBlock(block: Xml.XmlElement) : ControlFlowNode {
        const blockContent = block.get("./xmlns:block_content", Xml.ns)!;
        let ret: ControlFlowNode | null = null;
        const children = blockContent.elementChildren

        for (const child of children) {
            // TODO: Skippable Nodes Refactor
            if (child.name === "function") continue;

            let childnode = ControlFlowGraph.buildGraph(child);

            if (!childnode) continue;

            if (!ret) {
                ret = childnode;
                continue;
            }

            if (child.name != "break" && child.name != "continue" 
                && child.name != "return" && child.name != "goto") {
                ControlFlowNode.connectNodes(ret, childnode);
            } else {
                ControlFlowNode.connectNodes(ret, childnode, false);
                break; // any other nodes would be unreachable code
            }
        }
        return ret ? ret : new ControlFlowNode(blockContent);
    }

    private static buildUnit(unit: Xml.XmlElement) : ControlFlowNode {
        const children = unit.elementChildren;
        let ret: ControlFlowNode | null = null;

        for (const child of children) {
            // TODO: Skippable Nodes Refactor
            if (child.name === "function") continue;

            let childnode = ControlFlowGraph.buildGraph(child);

            if (!childnode) continue;

            if (!ret) {
                ret = childnode;
                continue;
            }

            if (child.name != "break" && child.name != "continue" 
                && child.name != "return" && child.name != "goto") {
                ControlFlowNode.connectNodes(ret, childnode);
            } else {
                ControlFlowNode.connectNodes(ret, childnode, false);
                break; // any other nodes would be unreachable code
            }
        }

        return ret ? ret : new ControlFlowNode(unit);
    }

    // the chosen approach may mess up the tails on the internal conds, but
    // those aren't relevant to the build process and tails should not be used
    // for tree traversal
    private static buildIf(ifStmt: Xml.XmlElement) : ControlFlowNode {
        // initial if
        const tail: ControlFlowNode[] = [];
        let ifXml: Xml.XmlElement | null = ifStmt.elementChildren[0];

        const firstCondXml = ifXml.get("./xmlns:condition", Xml.ns)!;
        const firstCond = ControlFlowGraph.buildGraph(firstCondXml)!;
        const firstBlockXml = ifXml.get("./xmlns:block", Xml.ns)!;
        const firstBlock = ControlFlowGraph.buildBlock(firstBlockXml);

        ControlFlowNode.connectNodes(firstCond, firstBlock);
        tail.push(...firstBlock.getTail());

        let cond = firstCond;
        while (ifXml = ifXml.nextElement) {
            const blockXml = ifXml.get("./xmlns:block", Xml.ns)!;
            const blockNode = ControlFlowGraph.buildBlock(blockXml);

            //const ifXml = ifStmt.childNodes()[i]; // <if> or <else>
            const condXml = ifXml.get("./xmlns:condition", Xml.ns);
            if (condXml) { // <else> has no <cond>
                const newCond = ControlFlowGraph.buildGraph(condXml)!;
                cond.addAdjacent(newCond);
                cond = newCond;
            }
            cond.addAdjacent(blockNode);
            tail.push(...blockNode.getTail());
        }

        if (!ifStmt.contains("./xmlns:else", Xml.ns)) tail.push(cond);

        firstCond.setTail(tail); 
        return firstCond;
    }
    
    private static buildCase(caseStmt: Xml.XmlElement) : ControlFlowNode {
        const caseNode = new ControlFlowNode(caseStmt);

        let curr: Xml.XmlElement | null = caseStmt;
        while (curr = curr.nextElement) {
            if (curr.name === "case" || curr.name === "default") break;
            // TODO: Skippable Nodes Refactor
            if (curr.name === "function") continue;
            const currNode = ControlFlowGraph.buildGraph(curr);
            if (!currNode) continue;
            ControlFlowNode.connectNodes(caseNode, currNode);
            if (curr.name === "break") {
                currNode.setConnectable(true);
                ControlFlowGraph.loopJumps.pop();
                break;
            }
        }
        return caseNode;
    }

    private static buildSwitch(switchStmt: Xml.XmlElement) : ControlFlowNode {
        const condXml = switchStmt.get("./xmlns:condition", Xml.ns)!;
        const cond = ControlFlowGraph.buildGraph(condXml)!;

        let hasDefaultCase: boolean = false;
        const casesXml = switchStmt.get("./xmlns:block/xmlns:block_content", Xml.ns)!
            .elementChildren
            .filter((node: Xml.XmlElement) => {
                if (node.name == "case") return true;
                if (node.name == "default") {
                    hasDefaultCase = true;
                    return true;
                }
                return false;
            });
    
        let prevCase: ControlFlowNode | null = null;
        for (const caseXml of casesXml) {
            let caseNode: ControlFlowNode = ControlFlowGraph.buildCase(caseXml);
            let hasBreak: boolean = caseNode.getTail()[0].xml.name == "break";

            if (prevCase) ControlFlowNode.connectNodes(prevCase, caseNode);
            cond.addAdjacent(caseNode);

            if (!hasBreak) cond.popTailNode();

            prevCase = hasBreak ? null : caseNode;  
        }

        if (!hasDefaultCase) cond.addTailNode(cond);

        return cond;
    }

    private static buildWhile(whileStmt: Xml.XmlElement) : ControlFlowNode {

        const condition = whileStmt.get("./xmlns:condition", Xml.ns)!;
        const condNode = ControlFlowGraph.buildGraph(condition)!;
    
                
        const blockXml = whileStmt.get("./xmlns:block", Xml.ns)!;
        const blockNode = ControlFlowGraph.buildBlock(blockXml);
        ControlFlowNode.connectNodes(condNode, blockNode);
        ControlFlowNode.connectNodes(blockNode, condNode, false);

        condNode.setTail([]); 

        condNode.loopVariants = whileStmt.defSymbols;

        ControlFlowGraph.resolveLoopJumps(condNode, condNode);

        return condNode;
    }

    // ! Assume for loop always has two semicolons
    private static buildFor(forstmt: Xml.XmlElement) : ControlFlowNode {

        const initXml = forstmt.get("./xmlns:control/xmlns:init", Xml.ns)!;
        const initNode = ControlFlowGraph.buildGraph(initXml)!;
        
        // condition
        const condition = forstmt.get("./xmlns:control/xmlns:condition", Xml.ns)!;
        const condNode = ControlFlowGraph.buildGraph(condition)!;
        ControlFlowNode.connectNodes(initNode, condNode);

        // body
        const blockXml = forstmt.get("./xmlns:block", Xml.ns)!;
        const blockNode = ControlFlowGraph.buildBlock(blockXml)!;
        ControlFlowNode.connectNodes(initNode, blockNode);

        const incrXML = forstmt.get("./xmlns:control/xmlns:incr", Xml.ns)!;
        const incrNode = ControlFlowGraph.buildGraph(incrXML)!;
        ControlFlowNode.connectNodes(blockNode, incrNode);
        ControlFlowNode.connectNodes(incrNode, condNode);


        initNode.setTail([condNode]);

        ControlFlowGraph.resolveLoopJumps(incrNode, initNode);

        condNode.loopVariants = forstmt.defSymbols;

        return initNode;
    }

    private static buildDo(doStmt: Xml.XmlElement) : ControlFlowNode {

        const blockXml = doStmt.get("./xmlns:block", Xml.ns)!;
        const blockNode = ControlFlowGraph.buildBlock(blockXml)!;

        // condition
        const conditionXML = doStmt.get("./xmlns:condition", Xml.ns)!;
        const condNode = ControlFlowGraph.buildGraph(conditionXML)!;

        condNode.loopVariants = doStmt.defSymbols;

        ControlFlowNode.connectNodes(blockNode, condNode);
        ControlFlowNode.connectNodes(condNode, blockNode, false);

        ControlFlowGraph.resolveLoopJumps(condNode, blockNode);

        return blockNode;
    }

    private static resolveLoopJumps(enterNode: ControlFlowNode, exitNode: ControlFlowNode) {
        for (const jumpNode of ControlFlowGraph.loopJumps) {
            jumpNode.setConnectable(true);
            if (jumpNode.xml.name == "continue") {
                ControlFlowNode.connectNodes(jumpNode, enterNode);
            // break
            } else {
                exitNode.addTailNode(jumpNode);
            }
        }
    }

    private static buildLabel(labelStmt: Xml.XmlElement) : ControlFlowNode {
        const labelNode = new ControlFlowNode(labelStmt);
        const labelNameXml = labelStmt.get("./xmlns:name", Xml.ns)!;
        this.labelNodes.set(labelNameXml.text, labelNode);

        for (let i = 0; i < ControlFlowGraph.gotoJumps.length; i++) {
            const gotoNode = ControlFlowGraph.gotoJumps[i];
            const nodeLabelXml = gotoNode.xml.get("./xmlns:name", Xml.ns)!;
            if (nodeLabelXml.text === labelNameXml.text) {
                gotoNode.addAdjacent(labelNode);
                ControlFlowGraph.gotoJumps.splice(i, 1);
                i--;
            }
        }
        return labelNode;
    }

    private static buildGoto(gotoStmt: Xml.XmlElement) : ControlFlowNode {
        const gotoNode = new ControlFlowNode(gotoStmt);
        const labelNameXml = gotoStmt.get("./xmlns:name", Xml.ns)!;
        const labelNode = ControlFlowGraph.labelNodes.get(labelNameXml.text);
        if (labelNode) {
            gotoNode.addAdjacent(labelNode);
        } else {
            ControlFlowGraph.gotoJumps.push(gotoNode);
        }
        gotoNode.setConnectable(false);
        return gotoNode;
    }

    public getRanges() : void {

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
    private static maxID: number = 1;


    private tail: ControlFlowNode[]; // used exclusively for build process then deleted
    private connectable: boolean = true;

    private _xml: Xml.XmlElement;

    private outEdges: ControlFlowNode[];
    private inEdges: ControlFlowNode[];

    private order: number = -1; // topological order

    private _idNum: number; // for toDot ouptut 

    public inRanges: Map<ControlFlowNode, RangeDomain> = new Map<ControlFlowNode, RangeDomain>();
    private currRanges: RangeDomain;
    public outRanges: Map<ControlFlowNode, RangeDomain> = new Map<ControlFlowNode, RangeDomain>();

    private backedge: boolean | undefined;
    public loopVariants: Set<Xml.XmlElement> | undefined = undefined;

    public constructor(data: Xml.XmlElement) {
        this._xml = data
        this.outEdges = [];
        this.inEdges = [];
        this.tail = [];
        this._idNum = ControlFlowNode.maxID++;
        this.currRanges = new RangeDomain();
    }

    // this extending tail
    public addAdjacent(node: ControlFlowNode) {
        if (!this.outEdges.includes(node)) this.outEdges.push(node);
        if (!node.inEdges.includes(this)) node.inEdges.push(this);
        for (const tailNode of node.getTail()) {
            if (!this.tail.includes(tailNode)) this.tail.push(tailNode);
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
        if (updateTail) from.tail = to.getTail();
    }

    // tail are all the nodes without outgoing edge
    public getTail() : ControlFlowNode[] {
        return this.tail.length > 0 ? this.tail : [this];
    }

    public setTail(newTail: ControlFlowNode[]) : void {
        this.tail = newTail;
    }

    public setConnectable(val: boolean) : void {
        this.connectable = val;
    }

    public addTailNode(node: ControlFlowNode) : void {
        this.tail.push(node);
    }

    public popTailNode() : void {
        this.tail.pop();
    }

    public get adjacents() : ControlFlowNode[] {
        return this.outEdges;
    }

    public get xml() : Xml.XmlElement {
        return this._xml;
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
        ret += this._idNum + " " + this._xml.name;
        return ret;
    }

    public nodeInfoToString() : string {
        let ret = "";
        ret += `node${this._idNum} [label="#${this.order}\\n<${this._xml.name}>\\n`;
        ret += `${this._xml.text.trim()}\\n`;
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