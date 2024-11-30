import { assert } from 'console';
import * as Xml from '../../common/Xml/Xml.js';
import { RangeDomain, Range } from './RangeDomain.js';
import { ControlFlowGraph as CFG, ControlFlowNode as CFNode } from './ControlFlowGraph.js';
import { execSync } from 'child_process';
import * as ComplexMath from '../../common/ComputerAlgebraSystem.js';
import { createXml } from '../../common/srcml.js';

// NOTE: Using Xml.Element.toString() as the key, because using the object
// * JS would store a reference not the value
const rangeDomains = new Map<string, RangeDomain>();

export function getRanges(root: Xml.Element) : Map<string, RangeDomain> {
    // TO

    // TODO : ALIAS AnALYSIS

    const cfg = CFG.buildControlFlowGraph(root);

    cfg.topologicalSort();
    iterateToFixpoint(cfg, true);
    console.log(cfg.toString());
    // iterateToFixpoint(cfg, false);

    // TODO: FILTER UNSAFE RANGES
    return cfg.getRangeMap(root);

}

export function query(stmt: Xml.Element) : RangeDomain | null {
    let ret = rangeDomains.get(`${stmt.line} ${stmt.text}`);
    if (!ret) {
        getRanges(stmt.enclosingFunction).forEach((rangeDomain, key) => {
            rangeDomains.set(key, rangeDomain);
        });
        ret = rangeDomains.get(`${stmt.line} ${stmt.text}`);
    }
    return ret ?? null;
}

// TODO: Review
function iterateToFixpoint(graph: CFG, widen: boolean = false) : void {
    const worklist: CFNode[] = [];

    if (widen) {
        worklist.push(graph.nodes[0]);
    } else {
        for (const node of graph.nodes) {
            if (node.hasBackedge()) worklist.push(node);
        }
    }

    while (worklist.length > 0) {
        const index = CFG.getIndexOfFirstNodeTopographically(worklist);
        const node = worklist.splice(index, 1)[0];

        const currRanges = new RangeDomain();
        for (const inData of node.inRanges) {
            if (inData[1].isEmpty()) continue;
            currRanges.unionRangeDomains(inData[1]);
        }
        
        const nodePrevRanges = node.getRanges();
        if (!nodePrevRanges.isEmpty() && node.hasBackedge()) {
            if (widen) {
                const widener = node.loopVariants!;
                if (widener.size > 0) {
                    currRanges.widenAffectedRanges(nodePrevRanges, widener);
                } else {
                    currRanges.widenAffectedRanges(nodePrevRanges);
                }
            } else {
                currRanges.narrowRanges(nodePrevRanges);
            }
        }

        // TODO: Handle Scope
        // TODO: PRAGMA

        if (nodePrevRanges.isEmpty() || !nodePrevRanges.equals(currRanges)) {
            node.setRanges(currRanges);
            updateRanges(node);
            for (const outData of node.outRanges) {
                worklist.push(outData[0]);
            }
        }
    }
}

// TODO: ALL OTHER CASES
function updateRanges(node: CFNode) : void {

    // TODO: Count visits and only do this once
    processIncrements('PRE', node);

    // TODO: InterProceduralAnalysis and Function Calls
    if ((node.xml.name === "decl_stmt" && node.xml.get("./xmlns:decl/xmlns:init")) || (node.xml.get("./xmlns:expr/xmlns:operator") 
        && [...node.xml.get("./xmlns:expr/xmlns:operator")!.text].filter((char) => char === '=' ).length === 1)
        || (node.xml.name === "init") || (node.xml.name === "incr")) {
        // TODO: SPLIT BASED ON NODE TYPE
        updateAssignment(node);
    } else if (node.xml.name === "condition" && node.xml.contains("parent::switch")) {
        // update switch
        updateSwtich(node);
    } else if (node.xml.name === "condition") {
        updateCondtion(node);
        // TODO: UPDATE CONDITION TO SPECIFY A TRUE AND A FALSE BRANCH
    } else if (node.xml.name === 'goto') {
        updateUnsafeNode(node);
    } else {
        updateSafeNode(node);
    }
    // TODO: unsafeNodes
        // TODO: update unsafeNode if it contains a side effect


    processIncrements('POST', node);
}

function updateAssignment(node: CFNode, expression?: string) : void {
    type AssignmentDirection = "normal" | "nochange" | "kill" | "recurrence" 
        | "invertible";

    // TODO: Handle Increment
    
    // TODO: Handle multiple assignment (ie multiple to's)
    const to = node.xml.get(".//xmlns:name[not(parent::xmlns:type)]");
    if (!to) return;
    let from: Xml.Element;
    let direction: AssignmentDirection;
    if (node.xml.name === "decl_stmt") {
        from = node.xml.get(".//xmlns:init/xmlns:expr")!;
    } else if (node.xml.name === "expr_stmt" || node.xml.name === "init" || node.xml.name === 'incr' || node.xml.name === 'expr') {
        // TODO: replace with Xml.Expression.getRHS
        const expr = node.xml.name !== 'expr' ? node.xml.get(".//xmlns:expr") : node.xml;
        if (!expr) throw new Error("IMPROPER SRCML");
        const assignOpIndex = expr.childElements.findIndex( (child: Xml.Element) => {
            return [...child.text].filter((char) => char === '=' ).length === 1;
        });
        let rhsString: string = "";
        for (let i = assignOpIndex + 1; i < expr.childElements.length; i++) {
            rhsString += expr.child(i)!.text;
        }

        const language = node.xml.get("/xmlns:unit")?.getAttribute("language");
        if (!language) throw new Error("IMPROPER SRCML");

        const buffer = execSync(`srcml --language ${language} --text "${rhsString}"`, {timeout: 10000});

        const rhsRoot = Xml.parseXmlString(buffer.toString());

        from = expr.copy();
        const kiddos = Array.from(from.domNode.childNodes);
        const aopi = kiddos.findIndex((node) => {
            return [...(node.textContent ?? "")].filter((char) => char === '=' ).length === 1;
        });
        for (let i = 0; i <= aopi; i++) {
            from.domNode.removeChild(kiddos[i]);
        }

    }

    if (from! === undefined) return;

 
    // simplify expr
    const simplifiedFrom = ComplexMath.simplifyXml(from)!;

   // TODO: DEREFERENCE

   let inverted;
    if (!isTractableType(to)) {
        direction = "nochange";
    } else if (!isTractableRange(from, simplifiedFrom)) {
        direction = "kill";
    } else if (!from.contains(`.//xmlns:name[text()='${to.text}']`)
        || !simplifiedFrom.contains(`.//xmlns:name[text()='${to.text}']`)) {
        direction = "normal";
    } else {
         inverted = ComplexMath.invertExpression(to, simplifiedFrom);
        if (!inverted) {
            direction = "recurrence";
        // } else if (inverted.contains(".//xmlns:name/xmlns:index")) {
        } else if (inverted.text.includes("[")) {
            direction = "kill";
        } else {
            direction = "invertible";
        }
    }

    const currRanges = node.getRanges();
    const outRanges = currRanges.copy();
    
    if (!isSimpleIdentifier(to)) {
        outRanges.removeRange(to.text);
    } else if (direction !== "nochange") {
        outRanges.killArraysWith(to);
        const replacer = direction === "invertible" ? inverted 
            : outRanges.getRange(to.text);
        
        let expandedFrom: string;
        if (direction === "invertible" 
            || direction === "recurrence") {
            expandedFrom = outRanges.getExpandedExpression(simplifiedFrom, to) ?? simplifiedFrom.text;
            outRanges.expandRangeExpressions(to);
        }

        outRanges.replaceSymbol(to, replacer);
        outRanges.removeRecurrence();

        if (direction === "kill") {
            outRanges.removeRange(to.text);
        } else {

            outRanges.setRange(new Range(to.text, expandedFrom! ?? simplifiedFrom.text,
                expandedFrom! ?? simplifiedFrom.text));
        }
        
    }

    for (const successor of node.succs) {
        // TODO: Refactor
        node.outRanges.set(successor, outRanges);
        successor.inRanges.set(node, outRanges);
    }

}

function updateSwtich(node: CFNode) : void {
    // TODO: update unsafeNode if it contains a side effect

    // // extract range from each case condtion == case value
    // for (const caseNode of node.outRanges) {
    //     if (caseNode[0].xml.name === 'case') {
            
    //         extractRanges(`${node.xml.get('xmlns:expr')} == ${caseNode[0].xml.get('xmlns:expr')!.text}`,
    //             node.xml.get("/xmlns:unit")!.getAttribute("language")!)
    //     }
    //     // TODO: Default
    // }

    // // intersect ranges
    // for (const caseNode of node.outRanges) {
    //     const newOutRange = Object.assign({}, node.getRanges()); // TODO: use this instead of .clone/.copy implementations
    //     newOutRange.intersectRanges(caseNode[1]);

    //     // As a result of the intersect, ranges that don't intersect are removed (ie [2,2] [3,Infinity)])
    //     // not intersected implies that the value cannot exist and that the path is infeasible
    //     if (caseNode[1].size > newOutRange.size) {
    //         caseNode[0].inRanges.delete(caseNode[0])
    //         node.outRanges.delete(caseNode[0]);
    //     } else { 
    //         node.outRanges.set(caseNode[0], newOutRange)
    //         caseNode[0].outRanges.set(caseNode[0], newOutRange)

    //     }
    // }
}

function updateCondtion(node: CFNode) : void {
    // throw new Error("UNIMPLEMENTED")
}

function updateSafeNode(node: CFNode) : void {
    // ? What about node's outRanges
    for (const successor of node.succs) {
        node.outRanges.set(successor, node.getRanges());
        successor.inRanges.set(node, node.getRanges());
    }
}

function updateUnsafeNode(node: CFNode) : void {
    // ? What about node's outRanges
    for (const successor of node.succs) {
        node.outRanges.set(successor, node.getRanges());
        successor.inRanges.set(node, new RangeDomain());
    }
}

function isTractableType(variable: Xml.Element) : boolean {
    assert(variable.name === "name");
    // find the variable's declaration
    // ! this won't work for python esque languages

    const decl = variable.get(`//xmlns:decl[./xmlns:name[text()='${variable.text}']]`);

    if (!decl) return false;

    // get type
    // NOTE: COULD IMPROVE BY USING DECL_STMT TO GET TYPE INSTEAD OF LOOP
    let type : Xml.Element = decl.get("./xmlns:type")!;
    if (!type) return false;
    while (type.getAttribute("ref") === "prev") {
        type = decl.prevElement!.get("./xmlns:type")!;
    }

    // check if type is int/
    return type.text.includes("int") && !type.text.includes("unsigned");
    
}

function isTractableRange(expression: Xml.Element, simplifiedExpr: Xml.Element) : boolean {
    // check if all number literals are within range
    const nums = expression.find(".//xmlns:literal[@type='number']");
    for (const num of nums) {
        const value = parseInt(num.text);
        // 4 seems arbitrary
        if (value >= Number.MAX_SAFE_INTEGER / 4 ||
            value <= Number.MIN_SAFE_INTEGER / 4
        ) {
            return false;
        }
    }

    // check that all nodes are tractable class
    // NOTE: NOT PRACTICAL

    // check that all names are tractable type
    const names = expression.find(".//xmlns:name");
    for (const name of names) {
        if (!isTractableType(name) && simplifiedExpr.text.includes(name.text)) {
            return false;
        }
    }

    // check that all ops are tractable
    const ops = expression.find(".//xmlns:operator");
    for (const op of ops) {
        if(!["+", "-", "/", "*", "%"].includes(op.text)) return false;
    }
    return true;
}

// returns true if not object member or array access
function isSimpleIdentifier(name: Xml.Element) : boolean {
    assert(name.name === "name");
    return !name.contains("./xmlns:name");
}

function processIncrements(type: 'PRE' | 'POST', node: CFNode) : void {

    const increments = node.xml.find(`.//xmlns:name[${type === 'PRE' ? 'preceding-sibling' : 'following-sibling'}::*[1]/text() = '++' 
        or ${type === 'PRE' ? 'preceding-sibling' : 'following-sibling'}::*[1]/text() = '--']`);
    if (increments.length < 1) return;
    const originalXML = node.xml.copy();
    const language = originalXML.get("/xmlns:unit")?.getAttribute("language") ?? ""; // TODO: fix

    for (const incr of increments) {
        const op = type === 'PRE' ? incr.prevElement!.text : incr.nextElement!.text;
        const tempXML = createXml(`${incr.text} = ${incr.text} ${op.charAt(0)} 1`, language);
        if (!tempXML) continue;
        node.xml.replace(tempXML);
        node.xml = tempXML;
        updateAssignment(node);
    }
    node.xml.replace(originalXML);
    node.xml = originalXML;

}

// function extractRanges(expr: string, language: string) : RangeDomain {
//     const ret = new RangeDomain();
//     const xml = createXml(expr, language);
//     if (!xml) return ret;
//     const ops = xml.find('xmlns:operator');
//     if (ops.length !== 1) // TODO: Handle more complex situations
// }