import * as Xml from '../../common/Xml/Xml.js';

export type ArrayAccessType = "Write" | "Read";

export class ArrayAccess {

    public static readonly WRITE_ACCESS: ArrayAccessType = "Write";
    public static readonly READ_ACCESS: ArrayAccessType = "Read";

    readonly access_type: ArrayAccessType;
    readonly access: Xml.Element;
     
    constructor(access: Xml.Element, accessType: ArrayAccessType) {
        this.access = access;
        this.access_type = accessType;
    }

    get arrayName() : string {
        return this.access.get('xmlns:name')!.text;
    }

    /**
     * The loop that the array resides in or undefined if the array access is not 
     * within an array
     */
    get enclosingLoop() : Xml.ForLoop | undefined {
        const enclosing_loops = <Xml.ForLoop[]> this.access.find("ancestor::xmlns:for");
        return enclosing_loops.at(-1);
    }
    
    /**
     * Returns the number of dimensions an array has
     * Examples:
     * * a[x] = 1
     * * a[x][x] = 2
     */
    public getArrayDimensionality() : number {
        return this.access.find("xmlns:index").length;
    }

    /**
     * Returns the <index> element of a specifc array dimension or null if the
     * array does not have that dimension
     */
    public getDimension(dimension: number) : Xml.Element | null {
        return this.access.get(`xmlns:index[${dimension}]`);
    }

    public equals(other: ArrayAccess) : boolean {
        return this.access.equals(other.access)
            && this.access_type === other.access_type;
    }

    public toString(): string {
        return `"${this.access.text}" (${(this.access_type === ArrayAccess.WRITE_ACCESS ? "Write" : "Read")})`;
    }

}
