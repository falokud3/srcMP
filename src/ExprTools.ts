import * as xml from 'libxmljs2';
import * as XmlTools from './XmlTools.js'

// should include =, +=, etc. but exclude ==, ===, or !=
function hasAssignmentOperator(stmt: xml.Element) : boolean {
    if (!(XmlTools.contains(stmt, "xmlns:operator", XmlTools.ns))) {
        return false;
    }

    const op = (stmt.get("xmlns:operator", XmlTools.ns) as xml.Element).text();
    
    // ensure it's not a boolean operator
    if (op.includes(">") || op.includes("<") || op.includes("!")) {
        return false;
    }

    try {
        return op.match(/=/g).length === 1;
    } catch {
        // if there are no matches then attempting to read the
        // length will throw an exception
        return false;
    }
}

export {hasAssignmentOperator}