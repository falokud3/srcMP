import * as xml from 'libxmljs2';

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
    
    // TODO: getters & setters
    public getAccessType() : string {
        return this.access_type;
    }

    public getEnclosingLoop() : xml.Element {
        return this.enclosing_loop;
    }

    // mayble: get loop nest of enclosing list

    public toString(): string {
        let ret: string = "";
        ret += "Access Expression: " + this.access_expr.toString() + " ";
        ret += "Access Type: " + (this.access_type == ArrayAccess.write_access ? "Write" : "Read") + " ";
        return ret;
    }

}

export {ArrayAccess}