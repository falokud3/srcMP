import * as Xml from '../Xml/Xml.js'

class ArrayAccess {

    // ? should this be an enum??

    public static readonly write_access: string = "Write";
    public static readonly read_access: string = "Read";

    private access_type: string;
    private access_expr: Xml.Element;
    private enclosing_loop: Xml.Loop;
    private parent_stmt: Xml.Element;
     
    constructor(access_type: string, access_expr: Xml.Element,
        enclosing_loop: Xml.Loop, parent_stmt: Xml.Element) {
        this.access_type = access_type;
        this.access_expr = access_expr;
        this.enclosing_loop = enclosing_loop;
        this.parent_stmt = parent_stmt;
    }
    
    public getArrayDimensionality() : number {
        return this.access_expr.find("xmlns:index", Xml.ns).length;
    }

    public getDimension(dimension: number) : Xml.Element {
        return this.access_expr.get(`xmlns:index[${dimension}]`, Xml.ns);
    }

    // TODO: getters & setters
    public getAccessType() : string {
        return this.access_type;
    }

    public getEnclosingLoop() : Xml.Loop {
        return this.enclosing_loop;
    }

    public get parentStatement() : Xml.Element {
        return this.parent_stmt;
    }
    // mayble: get loop nest of enclosing list

    public toString(): string {
        let ret: string = "[Array Access] ";
        ret += "Access Expression: " + this.access_expr.text + " ";
        ret += "Access Type: " + (this.access_type == ArrayAccess.write_access ? "Write" : "Read") + " ";
        return ret;
    }

}

export {ArrayAccess}