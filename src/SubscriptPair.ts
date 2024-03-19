import * as xml from 'libxmljs2';
import * as LoopTools from './util/LoopTools.js'
import * as XmlTools from './util/XmlTools.js'
import { ArrayAccess } from './ArrayAccess.js';

class SubscriptPair {
    private subscript1: xml.Element;
    private subscript2: xml.Element;

    private access1: ArrayAccess;
    private access2: ArrayAccess;

    private enclosingLoops: xml.Element[];

    constructor(subscript1: xml.Element, subscript2: xml.Element,
        access1: ArrayAccess, access2: ArrayAccess, loops: xml.Element[]) {
        this.subscript1 = subscript1;
        this.subscript2 = subscript2;
        this.access1 = access1;
        this.access2 = access2;
        this.enclosingLoops = loops;
    }

    public isReachable() : boolean {
        const stmt1 = this.access1.parentStatement;
        const stmt2 = this.access2.parentStatement;

        if (stmt1 == stmt2) {
            return this.access1.getAccessType() == ArrayAccess.read_access &&
                this.access2.getAccessType() == ArrayAccess.write_access;
        }

        throw new Error("isReachable unfinished")

    }


    public getComplexity() : number {
        let ret: number = 0;
        this.enclosingLoops.forEach((loop: xml.Element) => {
            const indexVar = LoopTools.getLoopIndexVariable(loop).text();
            if (XmlTools.containsName(this.subscript1, indexVar) ||
                XmlTools.containsName(this.subscript2, indexVar)) ret += 1;
        });
        return ret;
    }

    public getEnclosingLoops() : xml.Element[] {
        return this.enclosingLoops;
    }

    public getSubscript1() : xml.Element {
        return this.subscript1;
    }

    public getSubscript2() : xml.Element {
        return this.subscript2;
    }

    public toString() : string {
        let ret: string = '[SubscriptPair] ';
        ret += "Sub1: " + this.subscript1.text() + " ";
        ret += "Sub2: " + this.subscript2.text();
        return ret;
    }
}

export {SubscriptPair}