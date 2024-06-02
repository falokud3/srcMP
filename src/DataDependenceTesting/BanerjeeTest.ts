import { DependenceVector } from './DependenceVector';
import { SubscriptPair } from './SubscriptPair.js';
import * as Xml from '../Xml/Xml.js';



class BanerjeeTest {
    private subscript1: Xml.XmlElement;
    private subscript2: Xml.XmlElement;
    private loopnest: Xml.XmlElement[];
    private pair: SubscriptPair;
    private bounds: Map<string, number[]>;
    private const1;
    private const2;

    static readonly  LB = 0;
    static readonly  UB = 4;

    static readonly  LB_any = 0;
    static readonly  LB_less = 1;
    static readonly  LB_equal = 2;
    static readonly  LB_greater = 3;

    static readonly  UB_any = 4;
    static readonly  UB_less = 5;
    static readonly  UB_equal = 6;
    static readonly  UB_greater = 7;


    public constructor(pair: SubscriptPair) {
        this.pair = pair;
        this.subscript1 = pair.getSubscript1();
        this.subscript2 = pair.getSubscript2();
        this.loopnest = pair.getEnclosingLoops();

        this.bounds = new Map<string, number[]>()

        // get loop index variables
        const idList = []
        for (const loop of this.loopnest) {
            const decl = loop.get("./xmlns:control/xmlns:init/xmlns:decl", Xml.ns);
            // const coeff_node = <Xml.Element> decl.get("./xmlns:init/xmlns:expr/xmlns:literal", Xml.ns);
            // const coeff = parseInt(coeff_node.text);
            const index_node = decl!.get("./xmlns:name", Xml.ns);
            idList.push(index_node!.text)

        }

        // get constant coefficients
        this.const1 = this.getConstCoeff(this.subscript1);
        this.const2 = this.getConstCoeff(this.subscript2);

        // compute bounds for the loop
        // TODO: LOOPS

        for (let i = 0; i < this.loopnest.length; i++) {

            // !: FIX
            const A = 1;
            const B = 1;

            const up_node = this.loopnest[i].get("./xmlns:control/xmlns:condition/xmlns:expr/xmlns:literal", Xml.ns);
            const U = parseInt(up_node!.text);
            
            const decl = this.loopnest[i].get("./xmlns:control/xmlns:init/xmlns:decl", Xml.ns);
            const coeff_node = decl!.get("./xmlns:init/xmlns:expr/xmlns:literal", Xml.ns);
            const L = parseInt(coeff_node!.text);


            // ! FIX
            const N = 1;

            const bounds: number[] = []

            Math.min(A, 0);


            bounds.splice(BanerjeeTest.LB_any, 0,
                (Math.min(A, 0) - Math.max(B, 0))*(U-L) +
                (A-B)*L);
            bounds.splice(BanerjeeTest.LB_less, 0,
                    Math.min(Math.min(A, 0)-B)*(U-L-N) +
                    (A-B)*L - B*N, 0);
            bounds.splice(BanerjeeTest.LB_equal, 0, 
                    Math.min(A-B, 0)*(U-L) + (A-B)*L);
            bounds.splice(BanerjeeTest.LB_greater, 0,
                    Math.min(A-Math.max(B))*(U-L-N) +
                    (A-B)*L + A*N, 0);
            bounds.splice(BanerjeeTest.UB_any, 0,
                    (Math.max(A, 0) - Math.min(B, 0))*(U-L) +
                    (A-B)*L);
            bounds.splice(BanerjeeTest.UB_less, 0,
                    Math.max(Math.max(A, 0)-B, 0)*(U-L-N) +
                    (A-B)*L - B*N);
            bounds.splice(BanerjeeTest.UB_equal, 0,
                    Math.max(A-B, 0)*(U-L) + (A-B)*L);
            bounds.splice(BanerjeeTest.UB_greater, 0,
                    Math.max(A-Math.min(B, 0), 0)*(U-L-N) +
                    (A-B)*L + A*N);

            this.bounds.set(this.loopnest[i].text, bounds);
        }

    }

    public getConstCoeff(subscript: Xml.XmlElement) : number {
        const lit = subscript.get("./xmlns:expr/xmlns:literal", Xml.ns);
        if (lit) {
            const number = parseInt(lit.text)
            return number
        }
        return 0;
    }

    public get subscriptPair() {
        return this.pair;
    }

    public pairIsElligible() : boolean {
        return true;
    }

    public testDependence(dv: DependenceVector) : boolean {
        let LB = 0;
        let UB = 0;
        const diff = this.const2 - this.const1;

        for (const loop of this.loopnest) {
            const dir = dv.getDirection(loop, true);
            const loopBounds = this.bounds.get(loop.text);

            LB += loopBounds![BanerjeeTest.LB];
            UB += loopBounds![BanerjeeTest.UB];
        }

        // console.log(this.pair.toString())
        if (diff < LB || diff > UB) {
            // console.log("No Dependence found")
            return false;
        } else {
            // console.log("Dependence found")
            return true;
        }
    }

}

export {BanerjeeTest}