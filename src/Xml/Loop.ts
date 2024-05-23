import assert from "assert";
import { Element } from "./Element.js";
import * as libxmljs2 from 'libxmljs2';

import { ArrayAccess } from '../DataDependenceTesting/ArrayAccess.js';
import * as Xml from './Xml.js'


export class Loop extends Element {

    public constructor(libxml: libxmljs2.Element) {
        assert(libxml.name() === "for")
        super(libxml);
    }

    public static createLoop(element: Element) {
        return new Loop(element.libraryXmlObject);
    }

    // gets the loopnest that surrounds this loop (includes loop)
    public getEnclosingLoopNest() : Loop[] {
        return <Loop[]> this.find("ancestor-or-self::xmlns:for", Xml.ns);
    }

    // gets the loops nested within this loop (includes input loop)
    public getInnerLoopNest() : Loop[] {
        return <Loop[]> this.find("descendant-or-self::xmlns:for", Xml.ns);
    }

    // returns the intersection of two enclosing loop nests
    public getCommonEnclosingLoopNest(otherLoop: Loop) : Loop[] {
        const loopNest = this.getEnclosingLoopNest();
        const other_loopNest = otherLoop.getEnclosingLoopNest();
        return Loop.getCommonLoops(loopNest, other_loopNest);
    }

    public getInitialization() : Xml.Element {
        assert(this.name === "for")
        return this.get("xmlns:control/xmlns:init", Xml.ns);
    }

    public getLoopIndexVariableName() : Xml.Element {
        assert(this.name === "for")
        const index_var: Xml.Element = this.get("xmlns:control/xmlns:incr/xmlns:expr/xmlns:name", Xml.ns);
        return index_var;
    }

    /**
     * Returns a map containing all the array access of a loop where the
     * key is the array name and the value is a list of each array acess
     * ! Assumes <index> is associated with an array not an object
     */
    public getArrayAccesses() : Map<string, ArrayAccess[]> {   
        //const test: Xml.Element[] = loopNode.find(".//xmlns:name/xmlns:index[1]/..", Xml.ns);
        const array_access_map = new Map<string, ArrayAccess[]>();
        const indexNodes: Xml.Element[] = this.find(".//xmlns:index", Xml.ns);
        indexNodes.forEach( (indexNode: Xml.Element) => {
            const access_expr = indexNode.parent;
            const parent_stmt = <Xml.Expression> access_expr.parent;
            const enclosing_loops = parent_stmt.find("ancestor::xmlns:for", Xml.ns);
            const enclosing_loop = <Xml.Loop> enclosing_loops[enclosing_loops.length - 1];

            // ? use filter() before forEach instead of continue
            // declarations are not array acceses
            if (parent_stmt.name === "decl")  return;

            // edge case: multi-dimenstional arrays
            // need to avoid double counting
            if (access_expr.get("xmlns:index", Xml.ns) != indexNode) return;

            let access_type: string;
            // parent_stmt.child(0) is the LHS of an assigment expression
            if (parent_stmt.hasAssignmentOperator() 
                && parent_stmt.child(0) === access_expr) {
                access_type = ArrayAccess.write_access;

            } else {
                access_type = ArrayAccess.read_access;
            }

            const array_access = new ArrayAccess(access_type, access_expr,
                enclosing_loop, parent_stmt);

            const array_name = (access_expr.child(0) as Xml.Element).text;
            let array_accesses = array_access_map.get(array_name);
            if (array_accesses === undefined) {
                array_accesses = [];
                array_access_map.set(array_name, array_accesses);
            } 

            array_accesses.push(array_access);

            // NOTE: there may be a better way to structure this
                // maybe change hasAssignmentOperator to getOP
            if (parent_stmt.hasAugAssignmentOperator() 
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
    public static getCommonLoops(loopNest: Loop[], other_loopNest: Loop[]) : Loop[] {
        return loopNest.filter((node: Loop) => {
            return other_loopNest.some((inner) => {
                return node.text === inner.text
            });
        });
    }
}