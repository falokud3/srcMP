import * as xml from 'libxmljs2';
import { ArrayAccess } from './ArrayAccess.js';
import * as ExprTools from './ExprTools.js'
import * as XmlTools from './XmlTools.js'

// TODO: Write Loop Eligibility in LoopTools.ts
function isLoopEligible(loopNode: xml.Element) : boolean {
    return true;
} 

// gets the loopnest that surrounds this loop (includes loop)
function getEnclosingLoopNest(loopNode: xml.Element) : xml.Element[] {
    return loopNode.find("ancestor-or-self::xmlns:for", XmlTools.ns);
}

// gets the loops nested within this loop (does not include input loop)
function getInnerLoopNest(loopNode: xml.Element) : xml.Element[] {
    return loopNode.find("descendant::xmlns:for", XmlTools.ns);
}

// returns the intersection of two enclosing loop nests
function getCommonEnclosingLoopNest(loop: xml.Element, other_loop: xml.Element) : xml.Element[] {
    const loopNest = getEnclosingLoopNest(loop);
    const other_loopNest = getEnclosingLoopNest(other_loop);
    return getCommonLoops(loopNest, other_loopNest);
}

// returns the intersection of two loop nests
function getCommonLoops(loopNest: xml.Element[], other_loopNest: xml.Element[]) : xml.Element[] {
    return loopNest.filter((loopNode: xml.Element) => {
        return other_loopNest.includes(loopNode);
    });
}

/**
 * Returns a map containing all the array access of a loop where the
 * key is the array name and the value is a list of each array acess
 * ! Assumes <index> is associated with an array not an object
 * @param loopNode 
 */
function getArrayAccesses(loopNode: xml.Element) : Map<String, ArrayAccess[]> {   
    //const test: xml.Element[] = loopNode.find(".//xmlns:name/xmlns:index[1]/..", XmlTools.ns);
    const array_access_map = new Map<String, ArrayAccess[]>();
    const indexNodes: xml.Element[] = loopNode.find("//xmlns:index", XmlTools.ns);
    indexNodes.forEach( (indexNode: xml.Element) => {
        const access_expr = <xml.Element> indexNode.parent();
        const parent_stmt = <xml.Element> access_expr.parent();
        const enclosing_loop = loopNode.clone();

        // ? use filter() before forEach instead of continue
        // declarations are not array acceses
        if (parent_stmt.name() === "decl")  return;

        // edge case: multi-dimenstional arrays
        // need to avoid double counting
        if (access_expr.get("xmlns:index", XmlTools.ns) != indexNode) return;

        let access_type: string;
        // parent_stmt.child(0) is the LHS of an assigment expression
        if (ExprTools.hasAssignmentOperator(parent_stmt) 
            && parent_stmt.child(0) === access_expr) {
            access_type = ArrayAccess.write_access;

        } else {
            access_type = ArrayAccess.read_access;
        }

        const array_access = new ArrayAccess(access_type, access_expr,
            enclosing_loop, parent_stmt);

        const array_name = (access_expr.child(0) as xml.Element).text();
        let array_accesses = array_access_map.get(array_name);
        if (array_accesses === undefined) {
            array_accesses = [];
            array_access_map.set(array_name, array_accesses);
        } 

        array_accesses.push(array_access);

        // NOTE: there may be a better way to structure this
            // maybe change hasAssignmentOperator to getOP
        if (ExprTools.hasAugAssignmentOperator(parent_stmt) 
            && parent_stmt.child(0) === access_expr) {
            const aug_access = new ArrayAccess(ArrayAccess.read_access,
                access_expr, enclosing_loop, parent_stmt);
            array_accesses.push(aug_access);
        }
    });

    console.log(array_access_map);

    return array_access_map;
}

export {isLoopEligible, getArrayAccesses, getEnclosingLoopNest, getInnerLoopNest, getCommonEnclosingLoopNest, getCommonLoops};
