import * as xml from 'libxmljs2';
import * as XmlTools from './util/XmlTools.js'
import { assert } from 'console';
import { copyFile } from 'fs';

// ! doesn't support labels and goto statements
export class CFGraph {

    private nodes: CFNode[];

    private static loopJumps: CFNode[] = [];
    private static labelNodes: Map<string, CFNode> = new Map<string, CFNode>();
    private static gotoJumps: CFNode[] = [];

    public constructor() {
        this.nodes = [];
    }

    public addNode(node: CFNode) {
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

    public addAllNodes(node: CFNode, visited: number[]) : void {
        this.addNode(node);
        for (const adjNode of node.adjacents) {
            this.addNode(adjNode);
            if (!visited.includes(adjNode.num)) {
                visited.push(adjNode.num);
                this.addAllNodes(adjNode, visited);
            }
        }
    }

    private static rTopologicalSort(node: CFNode, visited: number[], newOrder: CFNode[]) {
        visited.push(node.num);
        for (const child of node.adjacents) {
            if (!visited.includes(child.num)) {
                this.rTopologicalSort(child, visited, newOrder);
            }
        }
        newOrder.splice(0, 0, node);
    }

    public topologicalSort(graph: CFGraph) {
        const newOrder: CFNode[] = [];
        CFGraph.rTopologicalSort(graph.nodes[0], [], newOrder);

        for (let i = 0; i < newOrder.length; i++) {
            newOrder[i].newOrder = i; 
        }
        graph.nodes = newOrder;
    }

    public static buildControlFlowGraph(src: xml.Element) : CFGraph {
        const graph = new CFGraph();
        const newGraph = CFGraph.buildGraph(src);
        graph.addAllNodes(newGraph, []);
        graph.topologicalSort(graph);
        return graph;
    }

    private static buildGraph(src: xml.Element) : CFNode {
        let copy = src.clone();
        let type: string = copy.name();
        if (type === "function") {
            return CFGraph.buildFunction(copy);
        } else if (type === "block") {
            return CFGraph.buildBlock(copy);
        } else if (type === "block") {
            return CFGraph.buildUnit(copy);
        } else if (type === "condition") { 
            return new CFNode(copy);
        } else if (type === "if_stmt") {
            return CFGraph.buildIf(copy);
        } else if (type === "switch") {
            return CFGraph.buildSwitch(copy);
        } else if (type === "while") {
            return CFGraph.buildWhile(copy);
        } else if (type === "do") {
            return CFGraph.buildDo(copy);
        } else if (type === "for") {
            return CFGraph.buildFor(copy);
        } else if (type == "init") {
            return new CFNode(copy);
        } else if (type == "incr") {
            return new CFNode(copy);
        } else if (type === "break" || type === "continue") {
            const ret = new CFNode(copy);
            ret.setConnectable(false);
            CFGraph.loopJumps.push(ret);
            return ret;
        } else if (type === "return") {
            const ret = new CFNode(copy);
            ret.setConnectable(false);
            return ret;
        } else if (type === "case" || type === "default") {
            return CFGraph.buildCase(copy);
        } else if (type.includes("stmt") || type === "expr" || type == "decl") {
            return new CFNode(copy);
        } else if (type === "label") {
            return CFGraph.buildLabel(copy);
        } else if (type === "goto") {
            return CFGraph.buildGoto(copy);
        } else {
            throw new Error("Unexpected Element: " + type);
            // return null;
        }
    }

    private static buildFunction(func: xml.Element) : CFNode {
        return CFGraph.buildBlock(func.get("./xmlns:block", XmlTools.ns));
    }

    private static buildBlock(block: xml.Element) : CFNode {
        const blockContent = <xml.Element> block.get("./xmlns:block_content", XmlTools.ns);
        let ret: CFNode = null;
        const children = <xml.Element[]> blockContent.childNodes().filter((xmlNode: xml.Node) => {
            return xmlNode.type() === "element";
        });
        for (const child of children) {
            let childnode = CFGraph.buildGraph(child);
            if (!ret) {
                ret = childnode;
                continue;
            }

            if (child.name() != "break" && child.name() != "continue" 
                && child.name() != "return" && child.name() != "goto") {
                CFNode.connectNodes(ret, childnode);
            } else {
                CFNode.connectNodes(ret, childnode, false);
                break; // any other nodes would be unreachable code
            }
        }
        return ret ? ret : new CFNode(blockContent);
    }

    private static buildUnit(unit: xml.Element) : CFNode {
        const children = <xml.Element[]> unit.childNodes().filter((xmlNode: xml.Node) => {
            // NOTE: UNTESTED
            return xmlNode.type() === "element" && xmlNode.namespaces.length === 1;
        });
        let ret: CFNode = null;

        for (const child of children) {
            let childnode = CFGraph.buildGraph(child);
            if (!ret) {
                ret = childnode;
                continue;
            }

            if (child.name() != "break" && child.name() != "continue" 
                && child.name() != "return" && child.name() != "goto") {
                CFNode.connectNodes(ret, childnode);
            } else {
                CFNode.connectNodes(ret, childnode, false);
                break; // any other nodes would be unreachable code
            }
        }

        return ret ? ret : new CFNode(unit);
    }

    // the chosen approach may mess up the tails on the internal conds, but
    // those aren't relevant to the build process and tails should not be used
    // for tree traversal
    private static buildIf(ifStmt: xml.Element) : CFNode {
        // initial if
        const tail: CFNode[] = [];
        let ifXml = <xml.Element> ifStmt.childNodes()[0];

        const firstCondXml = <xml.Element> ifXml.get("./xmlns:condition", XmlTools.ns);
        const firstCond = CFGraph.buildGraph(firstCondXml);
        const firstBlockXml = <xml.Element> ifXml.get("./xmlns:block", XmlTools.ns);
        const firstBlock = CFGraph.buildBlock(firstBlockXml);

        CFNode.connectNodes(firstCond, firstBlock);
        tail.push(firstBlock);

        let cond = firstCond;
        while (ifXml = ifXml.nextElement()) {
            const blockXml = <xml.Element> ifXml.get("./xmlns:block", XmlTools.ns);
            const blockNode = CFGraph.buildBlock(blockXml);

            //const ifXml = <xml.Element> ifStmt.childNodes()[i]; // <if> or <else>
            const condXml = <xml.Element> ifXml.get("./xmlns:condition", XmlTools.ns);
            if (condXml) { // <else> has no <cond>
                const newCond = CFGraph.buildGraph(condXml);
                cond.addAdjacent(newCond);
                cond = newCond;
            }
            cond.addAdjacent(blockNode);
            tail.push(blockNode);
        }

        if (!XmlTools.contains(ifStmt, "./xmlns:else", XmlTools.ns)) tail.push(cond);

        firstCond.setTail(tail); 
        return firstCond;
    }
    
    private static buildCase(caseStmt: xml.Element) : CFNode {
        const caseNode = new CFNode(caseStmt);

        let curr = caseStmt;
        while (curr = curr.nextElement()) {
            if (curr.name() === "case" || curr.name() === "default") break;
            const currNode = CFGraph.buildGraph(curr);
            CFNode.connectNodes(caseNode, currNode);
            if (curr.name() === "break") {
                currNode.setConnectable(true);
                CFGraph.loopJumps.pop();
                break;
            }
        }
        return caseNode;
    }

    private static buildSwitch(switchStmt: xml.Element) : CFNode {
        const condXml = <xml.Element> switchStmt.get("./xmlns:condition", XmlTools.ns);
        const cond = CFGraph.buildGraph(condXml);

        let hasDefaultCase: boolean = false;
        const casesXml = <xml.Element[]> (switchStmt.get("./xmlns:block/xmlns:block_content", XmlTools.ns) as xml.Element)
            .childNodes()
            .filter((node: xml.Node) => {
                if (node.type() !== "element") return false;
                if ((node as xml.Element).name() == "case") return true;
                if ((node as xml.Element).name() == "default") {
                    hasDefaultCase = true;
                    return true;
                }
            });
    
        let prevCase: CFNode = null;
        for (const caseXml of casesXml) {
            let caseNode: CFNode = CFGraph.buildCase(caseXml);
            let hasBreak: boolean = caseNode.getTail()[0].xml.name() == "break";

            if (prevCase) CFNode.connectNodes(prevCase, caseNode);
            cond.addAdjacent(caseNode);

            if (!hasBreak) cond.popTailNode();

            prevCase = hasBreak ? null : caseNode;  
        }

        if (!hasDefaultCase) cond.addTailNode(cond);

        return cond;
    }

    private static buildWhile(whileStmt: xml.Element) : CFNode {

        const condition = <xml.Element> whileStmt.get("./xmlns:condition", XmlTools.ns);
        const condNode = CFGraph.buildGraph(condition);
    
                
        const blockXml = <xml.Element> whileStmt.get("./xmlns:block", XmlTools.ns);
        const blockNode = CFGraph.buildBlock(blockXml);
        CFNode.connectNodes(condNode, blockNode);
        CFNode.connectNodes(blockNode, condNode, false);

        condNode.setTail([]); 

        CFGraph.resolveLoopJumps(condNode, condNode);

        return condNode;
    }

    // ! Assume for loop always has two semicolons
    private static buildFor(forstmt: xml.Element) : CFNode {

        const initXml = <xml.Element> forstmt.get("./xmlns:control/xmlns:init", XmlTools.ns);
        const initNode = CFGraph.buildGraph(initXml);
        
        // condition
        const condition = <xml.Element> forstmt.get("./xmlns:control/xmlns:condition", XmlTools.ns);
        const condNode = CFGraph.buildGraph(condition);
        CFNode.connectNodes(initNode, condNode);

        // body
        const blockXml = <xml.Element> forstmt.get("./xmlns:block", XmlTools.ns);
        const blockNode = CFGraph.buildBlock(blockXml);
        CFNode.connectNodes(initNode, blockNode);

        const incrXML = <xml.Element> forstmt.get("./xmlns:control/xmlns:incr", XmlTools.ns);
        const incrNode = CFGraph.buildGraph(incrXML);
        CFNode.connectNodes(blockNode, incrNode);
        CFNode.connectNodes(incrNode, condNode);


        initNode.setTail([condNode]);

        CFGraph.resolveLoopJumps(incrNode, initNode);

        return initNode;
    }

    private static buildDo(doStmt: xml.Element) : CFNode {

        const blockXml = <xml.Element> doStmt.get("./xmlns:block", XmlTools.ns);
        const blockNode = CFGraph.buildBlock(blockXml);

        // condition
        const conditionXML = <xml.Element> doStmt.get("./xmlns:condition", XmlTools.ns);
        const condNode = CFGraph.buildGraph(conditionXML);

        CFNode.connectNodes(blockNode, condNode);
        CFNode.connectNodes(condNode, blockNode, false);

        CFGraph.resolveLoopJumps(condNode, blockNode);

        return blockNode;
    }

    private static resolveLoopJumps(enterNode: CFNode, exitNode: CFNode) {
        for (const jumpNode of CFGraph.loopJumps) {
            jumpNode.setConnectable(true);
            if (jumpNode.xml.name() == "continue") {
                CFNode.connectNodes(jumpNode, enterNode);
            // break
            } else {
                exitNode.addTailNode(jumpNode);
            }
        }
    }

    private static buildLabel(labelStmt: xml.Element) : CFNode {
        const labelNode = new CFNode(labelStmt);
        const labelNameXml = <xml.Element> labelStmt.get("./xmlns:name", XmlTools.ns);
        this.labelNodes.set(labelNameXml.text(), labelNode);

        for (let i = 0; i < CFGraph.gotoJumps.length; i++) {
            const gotoNode = CFGraph.gotoJumps[i];
            const nodeLabelXml = <xml.Element> gotoNode.xml.get("./xmlns:name", XmlTools.ns);
            if (nodeLabelXml.text() === labelNameXml.text()) {
                gotoNode.addAdjacent(labelNode);
                CFGraph.gotoJumps.splice(i, 1);
                i--;
            }
        }
        return labelNode;
    }

    private static buildGoto(gotoStmt: xml.Element) : CFNode {
        const gotoNode = new CFNode(gotoStmt);
        const labelNameXml = <xml.Element> gotoStmt.get("./xmlns:name", XmlTools.ns);
        const labelNode = CFGraph.labelNodes.get(labelNameXml.text());
        if (labelNode) {
            gotoNode.addAdjacent(labelNode);
        } else {
            CFGraph.gotoJumps.push(gotoNode);
        }
        gotoNode.setConnectable(false);
        return gotoNode;
    }
}

class CFNode {
    private data: xml.Element;
    private outEdges: CFNode[];
    private inEdges: CFNode[];
    private tail: CFNode[]; // used exclusively for build process then deleted
    private order: number = -1;
    private idNum: number;
    private connectable: boolean = true;

    private static maxID: number = 1;

    public constructor(data: xml.Element) {
        this.data = data
        this.outEdges = [];
        this.inEdges = [];
        this.tail = [];
        this.idNum = CFNode.maxID++;
    }

    // this extending tail
    public addAdjacent(node: CFNode) {
        if (!this.outEdges.includes(node)) this.outEdges.push(node);
        if (!node.inEdges.includes(this)) node.inEdges.push(this);
        for (const tailNode of node.getTail()) {
            if (!this.tail.includes(tailNode)) this.tail.push(tailNode);
        }
    }

    // connecting tip to tail
    public static connectNodes(from: CFNode, to: CFNode, updateTail: boolean = true) {
        // spread fixes weird bug were tail would grow when adding nodes
        const fromTail = [...from.getTail()]

        for (const tailNode of fromTail) {
            if (!tailNode.connectable) continue;
            tailNode.addAdjacent(to);
        }
        if (updateTail) from.tail = to.getTail();
    }

    // tail are all the nodes without outgoing edge
    public getTail() : CFNode[] {
        return this.tail.length > 0 ? this.tail : [this];
    }

    public setTail(newTail: CFNode[]) : void {
        this.tail = newTail;
    }

    public setConnectable(val: boolean) : void {
        this.connectable = val;
    }

    public addTailNode(node: CFNode) : void {
        this.tail.push(node);
    }

    public popTailNode() : void {
        this.tail.pop();
    }

    public get adjacents() : CFNode[] {
        return this.outEdges;
    }

    public get xml() : xml.Element {
        return this.data;
    }

    public get num() : number {
        return this.idNum;
    }

    public set newOrder(order: number) {
        this.order = order;
    }

    public toString() : string {
        let ret: string = "";
        ret += this.idNum + " " + this.data.name();
        return ret;
    }

    public nodeInfoToString() : string {
        let ret = "";
        ret += `node${this.idNum} [label="#${this.order}\\n<${this.data.name()}>\\n`;
        ret += `${this.data.text().trim()}\\n"]`
        return ret;
    }

    public nodeEdgesToString() : string {
        if (this.outEdges.length == 0) return "";

        let ret = `node${this.idNum}->{ `;

        for (const adj of this.outEdges) {
            ret += `node${adj.idNum} `;
        }

        for (const inNodes of this.inEdges) {
            console.log(`node${inNodes.order} -> node${this.order}`);
        }

        ret += "};"
        return ret;
    }

}