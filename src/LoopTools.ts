import * as xml from 'libxmljs2';
import { ArrayAccess } from './ArrayAccess.js';
import * as ExprTools from './ExprTools.js'

const ns = {'xmlns': 'http://www.srcML.org/srcML/src'}

/**
 * Returns a map containing all the array access of a loop where the
 * key is the array name and the value is a list of each array acess
 * ! Assumes <index> is associated with an array not an object
 * TODO: add check to insure <index> is with an array; (may need symbol table
 * TODO: Test Nested Array Access
 * @param loopNode 
 */
function getArrayAccesses(loopNode: xml.Element) : Map<String, ArrayAccess[]> {    
    const array_access_map = new Map<String, ArrayAccess[]>();
    // ! index should not be nested in a <decl>
    // TODO: Check that index is not nested in a decl
    const indexNodes: xml.Element[] = loopNode.find(".//xmlns:index", ns);
    indexNodes.forEach( (indexNode: xml.Element) => {
        const access_expr = <xml.Element> indexNode.parent();
        const parent_stmt = <xml.Element> access_expr.parent();
        const enclosing_loop = loopNode.clone();

        let access_type: number;
        // parent_stmt.child(0) is the LHS of an assigment expression
        if (ExprTools.hasAssignmentOperator(parent_stmt) 
            && parent_stmt.child(0) == access_expr) {
            access_type = ArrayAccess.write_access;
        } else {
            access_type = ArrayAccess.read_access;
        }

        const array_access = new ArrayAccess(access_type, access_expr,
            enclosing_loop, parent_stmt);
        const array_name = (access_expr.child(0) as xml.Element).text();

        let array_accesses = array_access_map.get(array_name);
        if (array_accesses === undefined) {
            array_accesses = []; //initialize
            array_access_map.set(array_name, array_accesses);
        } 
        array_accesses.push(array_access);
        // array_accesses.set(array_name, )
    });

    console.log(array_access_map)
    return array_access_map;
}

export {getArrayAccesses};
