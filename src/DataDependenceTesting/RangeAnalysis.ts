import { assert } from 'console';
import * as Xml from '../Facades/Xml/Xml.js'
import { RangeDomain, Range } from './RangeDomain.js';
import { ControlFlowGraph as CFG, ControlFlowNode as CFNode } from './ControlFlowGraph.js';
import { execSync } from 'child_process';
import * as ComplexMath from '../Facades/ComputerAlgebraSystem.js'

// NOTE: Using Xml.Element.toString() as the key, because using the object
// * JS would store a reference not the value
const rangeDomains = new Map<string, RangeDomain>();

export function getRanges(root: Xml.Element) : Map<string, RangeDomain> {
    assert(root.name === "function" || root.name === "unit");

    // TODO : ALIAS AnALYSIS

    const cfg = CFG.buildControlFlowGraph(root);

    cfg.topologicalSort();
    iterateToFixpoint(cfg, true);
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
    if ((node.xml.name === "decl_stmt" && node.xml.get("./xmlns:decl/xmlns:init")) || (node.xml.get("./xmlns:expr/xmlns:operator") 
        && [...node.xml.get("./xmlns:expr/xmlns:operator")!.text].filter((char) => char === '=' ).length === 1)
        || (node.xml.name === "init")) {
        // TODO: SPLIT BASED ON NODE TYPE
        updateAssignment(node);
    // } else if (node.xml.name === "incr") {

        // basically update assignment but handle ++ --
    // } else if (node.xml.name === "condition" && node.xml.contains("parent::switch")) {
        // update switch
    // } else if (node.xml.name === "condition") {
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
    let from: Xml.Element;
    let direction: AssignmentDirection;
    if (node.xml.name === "decl_stmt") {
        from = node.xml.get(".//xmlns:init/xmlns:expr")!;
    } else if (node.xml.name === "expr_stmt" || node.xml.name === "init") {
        // TODO: replace with Xml.Expression.getRHS
        const expr = node.xml.get("./xmlns:expr");
        if (!expr) throw new Error("IMPROPER SRCML");
        let assignOpIndex = expr.childElements.findIndex( (child: Xml.Element) => {
            return [...child.text].filter((char) => char === '=' ).length === 1
        });
        let rhsString: string = "";
        for (let i = assignOpIndex + 1; i < expr.childElements.length; i++) {
            rhsString += expr.child(i)!.text;
        }

        const language = node.xml.get("/xmlns:unit")?.getAttribute("language");
        if (!language) throw new Error("IMPROPER SRCML");

        const buffer = execSync(`srcml --language ${language} --text "${rhsString}"`, {timeout: 10000});

        const rhsRoot = Xml.parseXmlString(buffer.toString());

        from = expr.copy()
        const kiddos = Array.from(from.domNode.childNodes);
        const aopi = kiddos.findIndex((node) => {
            return [...(node.textContent ?? "")].filter((char) => char === '=' ).length === 1
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
            direction = "recurrence"
        // } else if (inverted.contains(".//xmlns:name/xmlns:index")) {
        } else if (inverted.text.includes("[")) {
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
            expandedFrom = outRanges.getExpandedExpression(simplifiedFrom, to) ?? simplifiedFrom.text;
            outRanges.expandRangeExpressions(to);
        }

        outRanges.replaceSymbol(to, replacer);
        outRanges.removeRecurrence();

        if (direction === "kill") {
            outRanges.removeRange(to.text);
        } else {
            outRanges.setRange(to.text, expandedFrom! ?? simplifiedFrom.text,
                expandedFrom! ?? simplifiedFrom.text);
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

function isTractableType(variable: Xml.Element) : boolean {
    assert(variable.name === "name")
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