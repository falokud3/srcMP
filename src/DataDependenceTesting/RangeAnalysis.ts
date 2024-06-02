import { assert } from 'console';
import * as Xml from '../Xml/Xml.js'
import { RangeDomain, Range } from './RangeDomain.js';
import { ControlFlowGraph as CFG, ControlFlowNode as CFNode } from './ControlFlowGraph.js';
import { execSync } from 'child_process';
import * as ComplexMath from '../ComputerAlgebraSystem.js'

// NOTE: Using Xml.Element.toString() as the key, because using the object
// * JS would store a reference not the value
const rangeDomains = new Map<string, RangeDomain>();

export function getRanges(stmt: Xml.XmlElement) {
    assert(stmt.name === "function" || stmt.name === "unit");

    // TODO : ALIAS AnALYSIS

    const cfg = CFG.buildControlFlowGraph(stmt);
    // console.log(cfg.toString());

    cfg.topologicalSort();
    iterateToFixpoint(cfg, true);
    console.log(cfg.toString());
    iterateToFixpoint(cfg, false);

    // getRangeMap
    // console.log(cfg.toString());


    // return rangeMap

}

export function query(stmt: Xml.XmlElement) : RangeDomain {
    return new RangeDomain();
    // let ret = rangeDomains.get(stmt.toString());
    // if (!ret) {
    //     getRanges(stmt.enclosingFunction)
    // }
    // return ret;
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
            currRanges.unionRanges(inData[1]);
        }
        
        const nodePrevRanges = node.getRanges();
        if (!nodePrevRanges.isEmpty() && node.hasBackedge()) {
            if (widen) {
                const widener = node.loopVariants!
                if (widener.size > 0) {
                    currRanges.widenAffectedRanges(nodePrevRanges, widener);
                } else {
                    currRanges.widenAffectedRanges(nodePrevRanges);
                }
            } else {
                currRanges.narrowRanges(nodePrevRanges)
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

    // TODO: InterProceduralAnalysis and Function Calls

    // TODO: DECL
    if ((node.xml.name === "decl_stmt" && node.xml.get("./xmlns:decl/xmlns:init")) || (node.xml.get("./xmlns:expr/xmlns:operator") 
        && [...node.xml.get("./xmlns:expr/xmlns:operator")!.text].filter((char) => char === '=' ).length === 1)) {
        updateAssignment(node);
    } else if (node.xml.name === "condition" && node.xml.contains("parent::switch")) {
        // update switch
    } else if (node.xml.name === "condition") {
        // update condition
    } else {
        updateSafeNode(node);
    }
}

function updateAssignment(node: CFNode) : void {
    type AssignmentDirection = "normal" | "nochange" | "kill" | "recurrence" 
        | "invertible";

    // TODO: Handle multiple assignment (ie multiple to's)
    const to = node.xml.get(".//xmlns:name[not(parent::xmlns:type)]");
    if (!to) return;
    let from: Xml.XmlElement;
    let direction: AssignmentDirection;
    if (node.xml.name === "decl_stmt") {
        from = node.xml.get(".//xmlns:init/xmlns:expr")!;
    } else if (node.xml.name === "expr_stmt") {
        const expr = node.xml.get("./xmlns:expr");
        if (!expr) throw new Error("IMPROPER SRCML");
        let assignOpIndex = expr.elementChildren.findIndex( (child: Xml.XmlElement) => {
            return [...child.text].filter((char) => char === '=' ).length === 1
        });
        let rhsString: string = "";
        for (let i = assignOpIndex + 1; i < expr.elementChildren.length; i++) {
            rhsString += expr.child(i)!.text;
        }

        const language = node.xml.get("/xmlns:unit")?.getAttribute("language");
        if (!language) throw new Error("IMPROPER SRCML");

        const buffer = execSync(`srcml --language ${language} --text "${rhsString}"`, {timeout: 10000});

        const rhsRoot = Xml.parseXmlString(buffer.toString());

        from = rhsRoot.get("./xmlns:expr")!;
    } else if (node.xml.name === "init") {
        from = node.xml.get("./xmlns:expr")!;
    }

    if (from! === undefined) return;

    // simplify expr
    // const simplifeidFrom = ComplexMath.simplifyXml(from)
    const simplifeidFrom = from;

   // TODO: DEREFERENCE

   let inverted;
    if (!isTractableType(to)) {
        direction = "nochange";
    } else if (!isTractableRange(from, simplifeidFrom)) {
        direction = "kill";
    } else if (!from.contains(`.//xmlns:name[text()='${to.text}']`)
        || !simplifeidFrom.contains(`.//name[text()='${to.text}']`)) {
        direction = "normal";
    } else {
         inverted = ComplexMath.invertExpression(to, simplifeidFrom);
        if (!inverted) {
            direction = "recurrence"
        } else if (inverted.contains(".//xmlns:name/xmlns:index")) {
            direction = "kill"
        } else {
            direction = "invertible"
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
            expandedFrom = outRanges.getExpandedExpression(simplifeidFrom, to) ?? simplifeidFrom.text;
            outRanges.expandRangeExpressions(to);
        }

        

        outRanges.replaceSymbol(to, replacer);
        outRanges.removeRecurrence();

        if (direction === "kill") {
            outRanges.removeRange(to.text);
        } else {
            outRanges.setRange(to.text, expandedFrom! ?? simplifeidFrom.text,
                expandedFrom! ?? simplifeidFrom.text);
        }
        
    }

    for (const successor of node.succs) {
        node.outRanges.set(successor, outRanges);
        successor.inRanges.set(node, outRanges);
    }

}

function updateSwtich(node: CFNode) : void {

}

function updateCondtion(node: CFNode) : void {

}

function updateSafeNode(node: CFNode) : void {
    // ? What about node's outRanges
    for (const successor of node.succs) {
        node.outRanges.set(successor, node.getRanges())
        successor.inRanges.set(node, node.getRanges());
    }
}

function updateUnsafeNode(node: CFNode) : void {
    // ? What about node's outRanges
    for (const successor of node.succs) {
        node.outRanges.set(successor, node.getRanges())
        successor.inRanges.set(node, new RangeDomain());
    }
}

function isTractableType(variable: Xml.XmlElement) : boolean {
    assert(variable.name === "name")
    // find the variable's declaration
    // ! this won't work for python esque languages

    const decl = variable.get(`//xmlns:decl[./xmlns:name[text()='${variable.text}']]`);

    if (!decl) return false;

    // get type
    // NOTE: COULD IMPROVE BY USING DECL_STMT TO GET TYPE INSTEAD OF LOOP
    let type : Xml.XmlElement = decl.get("./xmlns:type")!;
    if (!type) return false;
    while (type.getAttribute("ref") === "prev") {
        type = decl.prevElement!.get("./xmlns:type")!;
    }

    // check if type is int/
    return type.text.includes("int") && !type.text.includes("unsigned");
    
}

function isTractableRange(expression: Xml.XmlElement, simplifiedExpr: Xml.XmlElement) : boolean {
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
function isSimpleIdentifier(name: Xml.XmlElement) : boolean {
    assert(name.name === "name");
    return !name.contains("./xmlns:name");
}