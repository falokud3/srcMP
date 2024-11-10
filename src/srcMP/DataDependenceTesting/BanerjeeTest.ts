import { DependenceVector } from './DependenceVector';
import { SubscriptPair } from './SubscriptPair.js';
import * as Xml from '../../common/Xml/Xml.js';
import * as ComputerAlgebraSystem from '../../common/ComputerAlgebraSystem.js'


export class BanerjeeTest {
    private loopnest: Xml.ForLoop[];
    private banerjeeBounds: Map<string, number[]>;
    private const1 : string;
    private const2 : string;
    private pair: SubscriptPair;

    private isTestEligible: boolean = true

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
        this.loopnest = pair.getEnclosingLoops();

        this.banerjeeBounds = new Map<string, number[]>()

        // get loop index variables
        const idList: string[] = [];
        for (const loop of this.loopnest) {
            const index_node = Xml.getCanonicalIndexVariable(loop);
            if (index_node) {
                idList.push(index_node.text);
            }

        }

        // get constant coefficients
        // TODO: check if literal
        this.const1 = this.getConstantCoefficient(pair.getSubscript1(), idList);
        this.const2 = this.getConstantCoefficient(pair.getSubscript2(), idList);

        // compute bounds for the loop

        // TODO: ZIP() instead of i 
        for (let i = 0; i < this.loopnest.length; i++) {


            const id = idList[i];
            const loop = this.loopnest[i];

            if (!Xml.isLoopInvariant(loop, this.const1) 
                || !Xml.isLoopInvariant(loop, this.const1)) {
                this.isTestEligible = false;
                break;
            }

            // TODO: Change Subscript pair to have the expression not the index
            const coeff1 = this.getCoefficient(pair.getSubscript1().get('xmlns:expr')!.text, id);
            const coeff2 = this.getCoefficient(pair.getSubscript2().get('xmlns:expr')!.text, id);

            if (typeof coeff1 !== 'number' || typeof coeff2 !== 'number') {
                this.isTestEligible = false;
                break;
            }

            const A = coeff1;
            const B = coeff2;

            let U = loop.upperboundExpression
            let L = loop.lowerboundExpression;
            let N = Xml.getCanonicalIncrementValue(loop);

            if (typeof U !== 'number' || typeof L !== 'number' || typeof N !== 'number') {
                this.isTestEligible = false;
                break;
            }

            const bounds: number[] = []

            if ( N >= 0) {
                bounds.splice(BanerjeeTest.LB_any, 0,
                    (getNegativePart(A) - getPositivePart(B))*(U-L) +
                    (A-B)*L);
                bounds.splice(BanerjeeTest.LB_less, 0,
                    getNegativePart(getNegativePart(A)-B)*(U-L-N) +
                    (A-B)*L - B*N);
                bounds.splice(BanerjeeTest.LB_equal, 0, 
                        getNegativePart(A-B)*(U-L) + (A-B)*L);
                bounds.splice(BanerjeeTest.LB_greater, 0,
                        getNegativePart(A-getPositivePart(B))*(U-L-N) +
                        (A-B)*L + A*N);
                bounds.splice(BanerjeeTest.UB_any, 0,
                        (getPositivePart(A) - getNegativePart(B))*(U-L) +
                        (A-B)*L);
                bounds.splice(BanerjeeTest.UB_less, 0,
                        getPositivePart(getPositivePart(A)-B)*(U-L-N) +
                        (A-B)*L - B*N);
                bounds.splice(BanerjeeTest.UB_equal, 0,
                        getPositivePart(A-B)*(U-L) + (A-B)*L);
                bounds.splice(BanerjeeTest.UB_greater, 0,
                        getPositivePart(A-getNegativePart(B))*(U-L-N) +
                        (A-B)*L + A*N);
            } else {
                [U, L] = [L, U];
                N *= -1;

                bounds.splice(BanerjeeTest.LB_any, 0,
                    (getNegativePart(A) - getPositivePart(B))*(U-L) +
                    (A-B)*L);
                bounds.splice(BanerjeeTest.LB_less, 0,
                        getNegativePart(A - getNegativePart(B))*(U-L-N) +
                        (A-B)*L + A*N);
                bounds.splice(BanerjeeTest.LB_equal, 0, 
                        getNegativePart(A-B)*(U-L) + (A-B)*L);
                bounds.splice(BanerjeeTest.LB_greater, 0,
                        getNegativePart(getNegativePart(A)-B)*(U-L-N) +
                        (A-B)*L - B*N);
                bounds.splice(BanerjeeTest.UB_any, 0,
                        (getPositivePart(A)-getNegativePart(B))*(U-L) +
                        (A-B)*L);
                bounds.splice(BanerjeeTest.UB_less, 0,
                        getPositivePart(A-getNegativePart(B))*(U-L-N) +
                        (A-B)*L + A*N);
                bounds.splice(BanerjeeTest.UB_equal, 0,
                        getPositivePart(A-B)*(U-L) + (A-B)*L);
                bounds.splice(BanerjeeTest.UB_greater, 0,
                        getPositivePart(getPositivePart(A)-B)*(U-L-N) +
                        (A-B)*L - B*N);

            }
            this.banerjeeBounds.set(this.loopnest[i].toString(), bounds);
        }

    }

    public get subscriptPair() : SubscriptPair {
        return this.pair;
    }

    public getConstantCoefficient(subscript: Xml.Element, idList: string[]) : string {
        const expression = subscript.get('xmlns:expr')!.text;
        // TODO: use Range Analysis here
        let ret: string = expression;
        // TODO: use Algebrite filter instead
        for (const id of idList) {
            // maybe use filter instead
            ret = ComputerAlgebraSystem.simplify(`coeff(${ret}, ${id}, 0)`);
        }
        return ret;
    }

    public getCoefficient(expression: string, id: string) : string | number {
        const ret = ComputerAlgebraSystem.simplify(`coeff(${expression}, ${id}, 1)`);
        const num = Number(ret);
        return !Number.isNaN(num) ? num : ret;
    }

    public pairIsElligible() : boolean {
        return this.isTestEligible;
    }

    public testDependence(dv: DependenceVector) : boolean {
        let banerjeeLB = 0;
        let banerjeeUB = 0;
        const exprDiff = ComputerAlgebraSystem.simplify(`${this.const2} - (${this.const1})`);
        const diff = Number(exprDiff);

        if (Number.isNaN(diff)) {
            return true;
        }

        for (const loop of this.loopnest) {
            const dir = dv.getDirection(loop, true)!;
            const loopBounds = this.banerjeeBounds.get(loop.toString());

            banerjeeLB += loopBounds![dir + BanerjeeTest.LB];
            banerjeeUB += loopBounds![dir + BanerjeeTest.UB];
        }

        // console.log(this.pair.toString())
        if (diff < banerjeeLB || diff > banerjeeUB) {
            // console.log("No Dependence found")
            return false;
        } else {
            // console.log("Dependence found")
            return true;
        }
    }

}

function getNegativePart(num: number) : number {
    return Math.min(num, 0);
}

function getPositivePart(num: number) : number {
    return Math.max(num, 0);
}