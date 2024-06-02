
import { assert } from "console";

import { ArrayAccess } from '../DataDependenceTesting/ArrayAccess.js';
import { XmlElement } from "./Element.js";
import * as Xml from './Xml.js'


export class ForLoop extends XmlElement {

    public constructor(domElement: Element) {
        assert(domElement.tagName === "for")
        super(domElement);
    }

    public static createLoop(element: XmlElement) {
        return new ForLoop(element.domElement);
    }

    // gets the loopnest that surrounds this loop (includes loop)
    public getEnclosingLoopNest() : ForLoop[] {
        return <ForLoop[]> this.find("ancestor-or-self::xmlns:for", Xml.ns);
    }

    // gets the loops nested within this loop (includes input loop)
    public getInnerLoopNest() : ForLoop[] {
        return <ForLoop[]> this.find("descendant-or-self::xmlns:for", Xml.ns);
    }

    // returns the intersection of two enclosing loop nests
    public getCommonEnclosingLoopNest(otherLoop: ForLoop) : ForLoop[] {
        const loopNest = this.getEnclosingLoopNest();
        const other_loopNest = otherLoop.getEnclosingLoopNest();
        return ForLoop.getCommonLoops(loopNest, other_loopNest);
    }

    public get initialization() : XmlElement {
        const init = this.get("./xmlns:control/xmlns:init");
        if (!init) throw new Error("SrcML is not properly formatted");
        return init;
    }

    public get condition() : XmlElement {
        const cond = this.get("./xmlns:control/xmlns:condition");
        if (!cond) throw new Error("SrcML is not properly formatted");
        return cond;
    }

    public get increment() : XmlElement {
        const incr = this.get("./xmlns:control/xmlns:incr");
        if (!incr) throw new Error("SrcML is not properly formatted");
        return incr;
    }

    public get body() : XmlElement {
        const body = this.get("./xmlns:block/xmlns:block_content");
        if (!body) throw new Error("SrcML is not properly formatted");
        return body;
    }

    public getLoopIndexVariableName() : XmlElement | null {
        assert(this.name === "for")
        return this.get("xmlns:control/xmlns:incr/xmlns:expr/xmlns:name", Xml.ns);
    }

    /**
     * Returns a map containing all the array access of a loop where the
     * key is the array name and the value is a list of each array acess
     * ! Assumes <index> is associated with an array not an object
     */
    public getArrayAccesses() : Map<string, ArrayAccess[]> {   
        //const test: Xml.Element[] = loopNode.find(".//xmlns:name/xmlns:index[1]/..", Xml.ns);
        const array_access_map = new Map<string, ArrayAccess[]>();
        const indexNodes: XmlElement[] = this.find(".//xmlns:index", Xml.ns);
        indexNodes.forEach( (indexNode: XmlElement) => {
            // ! BAD: LAZY BAD BAD BAD
            // TODO: FIX THIS (I JUST COULDN'T BE BOTHERED AT THE TIME)
            const access_expr = indexNode.parentElement ?? indexNode;
            const parent_stmt = access_expr.parentElement ?? indexNode;
            const enclosing_loops = parent_stmt.find("ancestor::xmlns:for", Xml.ns);
            const enclosing_loop = <Xml.ForLoop> enclosing_loops[enclosing_loops.length - 1];

            // ? use filter() before forEach instead of continue
            // declarations are not array acceses
            if (parent_stmt.name === "decl")  return;

            // edge case: multi-dimenstional arrays
            // need to avoid double counting
            if (access_expr.get("xmlns:index", Xml.ns) != indexNode) return;

            let access_type: string;
            // parent_stmt.child(0) is the LHS of an assigment expression
            if (Xml.hasAssignmentOperator(parent_stmt) 
                && parent_stmt.child(0) === access_expr) {
                access_type = ArrayAccess.write_access;

            } else {
                access_type = ArrayAccess.read_access;
            }

            const array_access = new ArrayAccess(access_type, access_expr,
                enclosing_loop, parent_stmt);

            const array_name = (access_expr.child(0) as XmlElement).text;
            let array_accesses = array_access_map.get(array_name);
            if (array_accesses === undefined) {
                array_accesses = [];
                array_access_map.set(array_name, array_accesses);
            } 

            array_accesses.push(array_access);

            // NOTE: there may be a better way to structure this
                // maybe change hasAssignmentOperator to getOP
            if (Xml.hasAugAssignmentOperator(parent_stmt) 
                && parent_stmt.child(0) === access_expr) {
                const aug_access = new ArrayAccess(ArrayAccess.read_access,
                    access_expr, enclosing_loop, parent_stmt);
                array_accesses.push(aug_access);
                // REMOVE
                console.log(aug_access.toString());
            }

        });

        return array_access_map;
    }

    // returns the intersection of two loop nests
    public static getCommonLoops(loopNest: ForLoop[], other_loopNest: ForLoop[]) : ForLoop[] {
        return loopNest.filter((node: ForLoop) => {
            return other_loopNest.some((inner) => {
                return node.text === inner.text
            });
        });
    }
}