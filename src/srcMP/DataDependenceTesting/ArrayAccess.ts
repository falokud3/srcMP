import * as Xml from '../../common/Xml/Xml.js'

export type ArrayAccessType = "Write" | "Read";

export class ArrayAccess {

    public static readonly WRITE_ACCESS: ArrayAccessType = "Write";
    public static readonly READ_ACCESS: ArrayAccessType = "Read";

    private access_type: ArrayAccessType;
    readonly access: Xml.Element;
     
    constructor(access: Xml.Element, accessType: ArrayAccessType) {
        this.access = access;
        this.access_type = accessType;
    }

    get arrayName() : string {
        return this.access.get('xmlns:name')!.text;
    }

    get enclosingLoop() : Xml.ForLoop | undefined {
        const enclosing_loops = <Xml.ForLoop[]> this.access.find("ancestor::xmlns:for");
        return enclosing_loops.at(-1);
    }

    public get parentStatement() : Xml.Element {
        return this.access.parentElement!;
    }
    
    public getArrayDimensionality() : number {
        return this.access.find("xmlns:index").length;
    }

    public getDimension(dimension: number) : Xml.Element | null {
        return this.access.get(`xmlns:index[${dimension}]`);
    }

    // TODO: getters & setters
    public getAccessType() : ArrayAccessType {
        return this.access_type;
    }

    public equals(other: ArrayAccess) : boolean {
        return this.access.equals(other.access)
            && this.access_type === other.access_type;
    }

    
    // mayble: get loop nest of enclosing list

    public toString(): string {
        let ret: string = "[Array Access] ";
        ret += "Access Expression: " + this.access.text + " ";
        ret += "Access Type: " + (this.access_type == ArrayAccess.WRITE_ACCESS ? "Write" : "Read") + " ";
        return ret;
    }

}
