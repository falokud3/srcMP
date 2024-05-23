import * as libxmljs2 from 'libxmljs2';
import * as Xml from './Xml.js'
import { Element } from "./Element.js";
import assert from "assert";


export class Expression extends Element {

    public constructor(libxml: libxmljs2.Element) {
        assert(libxml.name() === "expr")
        super(libxml);
    }

    // TODO: Convert to simple list of assignment ops and augmentated assignment ops
    public hasAssignmentOperator() : boolean {
        if (!(this.contains("xmlns:operator", Xml.ns))) {
            return false;
        }
    
        const op = this.get("xmlns:operator", Xml.ns).text;
        
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
    
    
    
    public hasAugAssignmentOperator() : boolean {
        if (!this.hasAssignmentOperator()) return false;
    
        const op = this.get("xmlns:operator", Xml.ns).text;
        return op.length > 1;
    }
}