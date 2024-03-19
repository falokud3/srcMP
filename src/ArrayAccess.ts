import * as xml from 'libxmljs2';
import * as XmlTools from './util/XmlTools.js';

class ArrayAccess {

    // ? should this be an enum??

    public static readonly write_access: string = "Write";
    public static readonly read_access: string = "Read";

    private access_type: string;
    private access_expr: xml.Element;
    private enclosing_loop: xml.Element;
    private parent_stmt: xml.Element;
     
    constructor(access_type: string, access_expr: xml.Element,
        enclosing_loop: xml.Element, parent_stmt: xml.Element) {
        this.access_type = access_type;
        this.access_expr = access_expr;
        this.enclosing_loop = enclosing_loop;
        this.parent_stmt = parent_stmt;
    }
    
    public getArrayDimensionality() : number {
        return this.access_expr.find("xmlns:index", XmlTools.ns).length;
    }

    public getDimension(dimension: number) : xml.Element {
        return this.access_expr.get(`xmlns:index[${dimension}]`, XmlTools.ns);
    }

    // TODO: getters & setters
    public getAccessType() : string {
        return this.access_type;
    }

    public getEnclosingLoop() : xml.Element {
        return this.enclosing_loop;
    }

    public get parentStatement() : xml.Element {
        return this.parent_stmt;
    }
    // mayble: get loop nest of enclosing list

    public toString(): string {
        let ret: string = "[Array Access] ";
        ret += "Access Expression: " + this.access_expr.text() + " ";
        ret += "Access Type: " + (this.access_type == ArrayAccess.write_access ? "Write" : "Read") + " ";
        return ret;
    }

}

export {ArrayAccess}