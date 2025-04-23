/**
 * Build control flow graph from srcML elements
 */

import * as Xml from '../../common/Xml/Xml.js';
import { ControlFlowGraph, ControlFlowNode } from './ControlFlowGraph.js';

let loopJumps: ControlFlowNode[];
let labelNodes: Map<string, ControlFlowNode>; 
let gotoJumps: ControlFlowNode[];

export function buildGraph(src: Xml.Element) : ControlFlowGraph {
    loopJumps = [];
    labelNodes = new Map<string, ControlFlowNode>();
    gotoJumps = [];
    
    const graph = new ControlFlowGraph();

    const node = buildNode(src);
    if (node) {
        graph.addAllNodes(node, []);
        node.type = 'START';
        for (const endNode of node.getLeafNodes()) {
            endNode.type = 'END';
        }
    }

    return graph;
}

function buildNode(src: Xml.Element) : ControlFlowNode | null {
    const type: string = src?.name;
    if (!type) return null;
    if (type === "function") {
        return buildFunction(src);
    } else if (type === "block") {
        return buildBlock(src);
    } else if (type === "unit") {
        return buildUnit(src);
    } else if (type === "condition") { 
        return new ControlFlowNode(src);
    } else if (type === "if_stmt") {
        return buildIf(src);
    } else if (type === "switch") {
        return buildSwitch(src);
    } else if (type === "while") {
        return buildWhile(src);
    } else if (type === "do") {
        return buildDo(src);
    } else if (type === "for") {
        return buildFor(src);
    } else if (type === "init") {
        return new ControlFlowNode(src);
    } else if (type === "incr") {
        return new ControlFlowNode(src);
    } else if (type === "break" || type === "continue") {
        const ret = new ControlFlowNode(src);
        ret.setConnectable(false);
        loopJumps.push(ret);
        return ret;
    } else if (type === "return") {
        const ret = new ControlFlowNode(src);
        ret.setConnectable(false);
        return ret;
    } else if (type === "case" || type === "default") {
        return buildCase(src);
    } else if (type.includes("stmt") || type === "expr" || type === "decl" || type === "range" || type === "call") {
        return new ControlFlowNode(src);
    } else if (type === "label") {
        return buildLabel(src);
    } else if (type === "goto") {
        return buildGoto(src);
    } else if (type === "comment") {
        return null;
    } else {
        console.error("Unexpected Element: " + type);
        return null;
    }
}

function buildFunction(func: Xml.Element) : ControlFlowNode | null {
    const block = func.get("./xmlns:block");
    return buildBlock(block!);
}

function buildBlock(block: Xml.Element) : ControlFlowNode {
    const blockContent = block.get("./xmlns:block_content")!;
    let ret: ControlFlowNode | null = null;
    const children = blockContent.childElements;

    for (const child of children) {
        // TODO: Skippable Nodes Refactor
        if (child.name === "function") continue;

        const childnode = buildNode(child);

        if (!childnode) continue;

        if (!ret) {
            ret = childnode;
            continue;
        }

        if (child.name !== "break" && child.name !== "continue" 
            && child.name !== "return" && child.name !== "goto") {
            ControlFlowNode.connectNodes(ret, childnode);
        } else {
            ControlFlowNode.connectNodes(ret, childnode, false);
            break; // any other nodes would be unreachable code
        }
    }
    return ret ? ret : new ControlFlowNode(blockContent);
}

function buildUnit(unit: Xml.Element) : ControlFlowNode | null {
    const children = unit.childElements;
    let ret: ControlFlowNode | null = null;

    for (const child of children) {
        // TODO: Skippable Nodes Refactor
        if (child.name === "function") continue;

        const childnode = buildNode(child);

        if (!childnode) continue;

        if (!ret) {
            ret = childnode;
            continue;
        }

        if (child.name !== "break" && child.name !== "continue" 
            && child.name !== "return" && child.name !== "goto") {
            ControlFlowNode.connectNodes(ret, childnode);
        } else {
            ControlFlowNode.connectNodes(ret, childnode, false);
            break; // any other nodes would be unreachable code
        }
    }

    return ret ;
}

// the chosen approach may mess up the tails on the internal conds, but
// those aren't relevant to the build process and tails should not be used
// for tree traversal
function buildIf(ifStmt: Xml.Element) : ControlFlowNode {
    // initial if
    const tail: ControlFlowNode[] = [];
    let ifXml: Xml.Element | null = ifStmt.childElements[0];

    const firstCondXml = ifXml.get("./xmlns:condition")!;
    const firstCond = buildNode(firstCondXml)!;
    const firstBlockXml = ifXml.get("./xmlns:block")!;
    const firstBlock = buildBlock(firstBlockXml);

    ControlFlowNode.connectNodes(firstCond, firstBlock);
    tail.push(...firstBlock.getTail());

    let cond = firstCond;
    ifXml = ifXml.nextElement;
    while (ifXml) {
        const blockXml = ifXml.get("./xmlns:block")!;
        const blockNode = buildBlock(blockXml);

        //const ifXml = ifStmt.childNodes()[i]; // <if> or <else>
        const condXml = ifXml.get("./xmlns:condition");
        if (condXml) { // <else> has no <cond>
            const newCond = buildNode(condXml)!;
            cond.addAdjacent(newCond);
            cond = newCond;
        }
        cond.addAdjacent(blockNode);
        tail.push(...blockNode.getTail());
        ifXml = ifXml.nextElement;
    }

    if (!ifStmt.contains("./xmlns:else")) tail.push(cond);

    firstCond.setTail(tail); 
    return firstCond;
}

function buildCase(caseStmt: Xml.Element) : ControlFlowNode {
    const caseNode = new ControlFlowNode(caseStmt);

    let curr: Xml.Element | null = caseStmt;
    while (curr) {
        if (curr.name === "case" || curr.name === "default") break;
        // TODO: Skippable Nodes Refactor
        if (curr.name === "function") continue;
        const currNode = buildNode(curr);
        if (!currNode) continue;
        ControlFlowNode.connectNodes(caseNode, currNode);
        if (curr.name === "break") {
            currNode.setConnectable(true);
            loopJumps.pop();
            break;
        }
        curr = curr.nextElement;
    }
    return caseNode;
}

function buildSwitch(switchStmt: Xml.Element) : ControlFlowNode {
    const condXml = switchStmt.get("./xmlns:condition")!;
    const cond = buildNode(condXml)!;

    let hasDefaultCase: boolean = false;
    const casesXml = switchStmt.get("./xmlns:block/xmlns:block_content")!
        .childElements
        .filter((node: Xml.Element) => {
            if (node.name === "case") return true;
            if (node.name === "default") {
                hasDefaultCase = true;
                return true;
            }
            return false;
        });

    let prevCase: ControlFlowNode | null = null;
    for (const [index, caseXml] of casesXml.entries()) {
        const caseNode: ControlFlowNode = buildCase(caseXml);

        if (prevCase) ControlFlowNode.connectNodes(prevCase, caseNode);
        cond.addAdjacent(caseNode);

        const hasBreak: boolean = caseNode.getTail()[0].xml.name === "break";
        if (!hasBreak && index !== casesXml.length - 1) cond.popTailNode();

        prevCase = hasBreak ? null : caseNode;  
    }

    if (!hasDefaultCase) cond.addTailNode(cond);

    return cond;
}

function buildWhile(whileStmt: Xml.Element) : ControlFlowNode {

    const condition = whileStmt.get("./xmlns:condition")!;
    const condNode = buildNode(condition)!;

            
    const blockXml = whileStmt.get("./xmlns:block")!;
    const blockNode = buildBlock(blockXml);
    ControlFlowNode.connectNodes(condNode, blockNode);
    ControlFlowNode.connectNodes(blockNode, condNode, false);

    condNode.setTail([]); 

    resolveLoopJumps(condNode, condNode);

    return condNode;
}

// ! Assume for loop always has two semicolons
function buildFor(forstmt: Xml.Element) : ControlFlowNode {
    if ((forstmt as Xml.ForLoop).type === "RANGE") {
        const initXml = forstmt.get("./xmlns:control/xmlns:init")!;
        const initNode = buildNode(initXml)!;

        const blockXml = forstmt.get("./xmlns:block")!;
        const blockNode = buildBlock(blockXml);
        ControlFlowNode.connectNodes(initNode, blockNode);
        ControlFlowNode.connectNodes(blockNode, initNode, false);

        initNode.setTail([]);
        resolveLoopJumps(initNode, initNode);
        return initNode;
    }

    const initXml = forstmt.get("./xmlns:control/xmlns:init")!;
    const initNode = buildNode(initXml)!;
    
    // condition
    const condition = forstmt.get("./xmlns:control/xmlns:condition")!;
    const condNode = buildNode(condition)!;
    ControlFlowNode.connectNodes(initNode, condNode);

    // body
    const blockXml = forstmt.get("./xmlns:block")!;
    const blockNode = buildBlock(blockXml);
    ControlFlowNode.connectNodes(initNode, blockNode);

    const incrXML = forstmt.get("./xmlns:control/xmlns:incr")!;
    const incrNode = buildNode(incrXML)!;
    ControlFlowNode.connectNodes(blockNode, incrNode);
    ControlFlowNode.connectNodes(incrNode, condNode);


    initNode.setTail([condNode]);

    resolveLoopJumps(incrNode, initNode);

    return initNode;
}

function buildDo(doStmt: Xml.Element) : ControlFlowNode {

    const blockXml = doStmt.get("./xmlns:block")!;
    const blockNode = buildBlock(blockXml);

    // condition
    const conditionXML = doStmt.get("./xmlns:condition")!;
    const condNode = buildNode(conditionXML)!;

    ControlFlowNode.connectNodes(blockNode, condNode);
    ControlFlowNode.connectNodes(condNode, blockNode, false);

    resolveLoopJumps(condNode, blockNode);

    return blockNode;
}

function resolveLoopJumps(enterNode: ControlFlowNode, exitNode: ControlFlowNode) {
    for (const jumpNode of loopJumps) {
        jumpNode.setConnectable(true);
        if (jumpNode.xml.name === "continue") {
            ControlFlowNode.connectNodes(jumpNode, enterNode);
        // break
        } else {
            exitNode.addTailNode(jumpNode);
        }
    }
}

function buildLabel(labelStmt: Xml.Element) : ControlFlowNode {
    const labelNode = new ControlFlowNode(labelStmt);
    const labelNameXml = labelStmt.get("./xmlns:name")!;
    labelNodes.set(labelNameXml.text, labelNode);

    for (let i = 0; i < gotoJumps.length; i++) {
        const gotoNode = gotoJumps[i];
        const nodeLabelXml = gotoNode.xml.get("./xmlns:name")!;
        if (nodeLabelXml.text === labelNameXml.text) {
            gotoNode.addAdjacent(labelNode);
            gotoJumps.splice(i, 1);
            i--;
        }
    }
    return labelNode;
}

function buildGoto(gotoStmt: Xml.Element) : ControlFlowNode {
    const gotoNode = new ControlFlowNode(gotoStmt);
    const labelNameXml = gotoStmt.get("./xmlns:name")!;
    const labelNode = labelNodes.get(labelNameXml.text);
    if (labelNode) {
        gotoNode.addAdjacent(labelNode);
    } else {
        gotoJumps.push(gotoNode);
    }
    gotoNode.setConnectable(false);
    return gotoNode;
}