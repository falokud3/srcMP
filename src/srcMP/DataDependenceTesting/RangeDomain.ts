import { assert } from 'console';
import * as Xml from '../../common/Xml/Xml.js';
import * as CAS from '../../common/ComputerAlgebraSystem.js';

export class RangeDomain {
    private ranges: Map<string, Range>;

    public constructor() {
        this.ranges = new Map<string, Range>();
    }

    get size() : number {
        return this.ranges.keys.length;
    }

    public copy() : RangeDomain {
        const copy = new RangeDomain();
        for (const range of this.ranges) {
            copy.ranges.set(range[0], range[1].copy());
        }
        return copy;
    }

    public getRange(variable: string) : Range | undefined {
        const range = this.ranges.get(variable);
        return range;
    }

    public removeRange(variable: string) : boolean {
        return this.ranges.delete(variable);
    }

    // TODO: Remove
    public setRange(range: Range) {
        this.ranges.set(range.variable, range);
    }

    public killArraysWith(symbol: Xml.Element) {
        assert(symbol.name === "name");
        const symbolName = symbol.text;
        for (const range of this.ranges) {
            if (range[0].includes("[") 
                && (range[1].lowerbound.includes(symbolName) 
                    || range[1].upperbound.includes(symbolName))) {
                this.ranges.delete(range[0]);
            }
        }
    }

    public removeRecurrence() : void{
        for (const range of this.ranges) {
            if (range[1].lowerbound.includes(range[0]) 
                || range[1].upperbound.includes(range[0])) {
                this.ranges.delete(range[0]);
            }
        }
    }

    public replaceSymbol(original: Xml.Element, replacement: Xml.Element | Range | undefined | null) {
        if (!replacement) return;

        for (const range of this.ranges) {
            if (replacement instanceof Range) {
                if (replacement.isConstant) {
                    range[1].lowerbound = Range.substitute(range[1].lowerbound, original.text, replacement.lowerbound);
                    range[1].upperbound = Range.substitute(range[1].upperbound, original.text, replacement.upperbound);
                }
            } else {
                range[1].lowerbound = Range.substitute(range[1].lowerbound, original.text, replacement.text);
                range[1].upperbound = Range.substitute(range[1].upperbound, original.text, replacement.text);

            }
        }
    }

    public substituteForward(expression: Xml.Element) : string | number {
        const vars = expression.find("./descendant-or-self::xmlns:name[not(parent::xmlns:name)]");
        let ret = expression.text;
        for (const variable of vars) {
            const expanded = this.getExpandedExpression(ret, variable.text);
            if (expanded) ret = expanded;
        }
        const number = Number(ret);
        return number ? number : ret;
    }

    // returns the expression with all occurances of the specified variable replaced with
    // their value ranges
    public getExpandedExpression(expression: Xml.Element | string, variable: Xml.Element | string) : string | null {

        if (expression instanceof Xml.Element && variable instanceof Xml.Element) {
            const range = this.getRange(variable.text);
            
            return range && range.isConstant ? CAS.simplify(Range.substitute(expression.text, variable.text, range.lowerbound)) 
                : null;
        } else {
            const expr = <string> expression;
            const vari = <string> variable;
            const range = this.getRange(vari);

            
            return range && range.isConstant ? CAS.simplify(Range.substitute(expr, vari, range.lowerbound)) 
                : null;
        }


    }

    // updates the ranges by replacing the variable with its range 
    public expandRangeExpressions(variable: Xml.Element) : void {
        if (!this.ranges.has(variable.text)) return;

        const clone = this.copy();
        for (const variable of clone.ranges) {
            const expand = clone.expandRangeExpression(variable);
            if (expand) this.ranges.set(variable[0], new Range(variable[0], expand, expand));
        } 

    }

    public expandRangeExpression(range: [string, Range]) {
        return range[1].isConstant 
            ? this.getExpandedExpression(range[1].lowerbound, range[0]) : null;
    }

    public isEmpty() : boolean {
        return this.ranges.size === 0;
    }

    public equals(other: RangeDomain) : boolean {
        for (const variable of this.ranges.keys()) {
            if (!this.getRange(variable)!.equals(other.getRange(variable) ?? null)) {
                return false;
            }
        }
        return true;
    }

    public narrowRanges(other: RangeDomain) {
        for (const variable of other.ranges.keys()) {
            const result = RangeDomain.narrowRange(other.getRange(variable)!, this.getRange(variable)!);
            if (result?.isOmega ?? true) {
                this.removeRange(variable);
            } else {
                this.setRange(result);
            }
        }

    }

    public widenAffectedRanges(other: RangeDomain, vars: Set<Xml.Element> = new Set<Xml.Element>()) {
        const affected = new Set<string>();
        for (const var_range of other.ranges.keys()) {
            for (const varIn of vars) {
                if (this.getRange(var_range)!.variable === varIn.text) {
                    affected.add(var_range);
                }
            }
        }
        
        for (const val of vars) {
            affected.add(val.text);
        }

        for (const affect of affected) {
            const result = RangeDomain.widenRange(this, other.getRange(affect), this.getRange(affect));
            if (result.isOmega) {
                this.removeRange(affect);
            } else {
                this.setRange(result);
            }
        }
    }

    public unionRangeDomains(otherRange: RangeDomain) : void {
        const vars = new Set(this.ranges.keys());
        for (const key of otherRange.ranges.keys()) {
            vars.add(key);
        }
        for (const variable of vars) {
            const result = Range.unionRanges(this.getRange(variable), 
                otherRange.getRange(variable));
            if (result.isOmega) {
                this.removeRange(variable);
            } else {
                this.setRange(result);
            }
        }
    }

    intersectRanges(other: RangeDomain) : void {
        for (const range of other.ranges) {
            if (!this.ranges.has(range[0])) this.setRange(range[1]);
        }

        for (const range of this.ranges) {
            const newRange = Range.intersectRanges(range[1], other.getRange(range[0]));
        }
    }

    public static widenRange(rd: RangeDomain, e?: Range, widen?: Range) : Range {

        if (!e && !widen) return new Range('NULL', String(-Infinity), String(Infinity));

        if (!e) return widen!.copy();
        else if (!widen) return e.copy();

        if (e.isOmega || widen.isOmega){
            // TODO: CHECK EXPECTED BEHAVIOR
            return e.copy();
        }

        const ret = e.copy();
        if (e.lowerbound !== widen.lowerbound) {
            ret.lowerbound = String(-Infinity);
        }

        if (e.upperbound !== widen.upperbound) {
            ret.upperbound = String(Infinity);
        }

        return ret;
    }


    // TODO REVIEW
    public static narrowRange(e: Range, narrow: Range) : Range {
        if (narrow.isOmega) return e.copy();
        else if (e.isOmega) return narrow.copy();

        const ret = e.copy();
        if (Math.abs(Number(ret.lowerbound)) === Infinity) ret.lowerbound = narrow.lowerbound;
        if (Math.abs(Number(ret.upperbound)) === Infinity) ret.upperbound = narrow.upperbound;
        return ret;
    }

    public toString() : string {
        let ret = "[";
        for (const variable of this.ranges.keys()) {
            ret += this.getRange(variable)!.toString() + ", ";
        }
        if (!this.isEmpty()) ret = ret.substring(0, ret.length - 2);
        ret += "]";
        return ret;
    }



}

export class Range {
    
    private lb: string;
    private ub: string;
    private stride: string;
    public variable: string;

    public constructor(variable: string, lowerbound: string, upperbound: string, stride?: string) {
        this.variable = variable;
        this.lb = lowerbound;
        this.ub = upperbound;
        this.stride = stride ?? '1';
    }

    public copy() : Range {
        return new Range(this.variable, this.lb, this.ub, this.stride);
    }

    public get lowerbound() : string {
        return this.lb;
    }

    public set lowerbound(lowerbound: string) {
        this.lb = lowerbound;
    }

    public get upperbound() : string {
        return this.ub;
    }

    public set upperbound(upperbound: string) {
        this.ub = upperbound;
    }
    
    public get isConstant() : boolean {
        return this.lb === this.ub;
    }

    // TODO: MERGE isBounded and isOmega
    public get isOmega() {
        return Number(this.lb) === -Infinity && Number(this.ub) === Infinity;
    }

    // /**
    //  * Returns true if the range is narrower than (-inf, +inf)
    //  * This is oppositive of the concept of the ordinal number omega
    //  */
    // public get isBounded() : boolean {
    //     return this.lowerbound > -Infinity || this.upperbound < Infinity;
    // }

    public equals(other: Range | null) : boolean {
        if (other === null) return false;
        return this.lb === other.lb &&
            this.ub === other.ub;
    }

    public toString() : string {
        return this.isConstant ? `${this.variable} = ${this.lb}`
            :`${this.lb} < ${this.variable} < ${this.ub}`;
    }

    static substitute(inputString: string, original: string, replacement: string) : string {
        return inputString.replace(new RegExp("\\b"+original+"\\b"), replacement);
    }

    public static unionRanges(r1?: Range, r2?: Range) : Range {

        if (!r1 && !r2) return new Range('NULL', String(-Infinity), String(Infinity));
        else if (!r1) {
            return r2!;
        } else if (!r2) {
            return r1;
        }
 
        const lbRelation = CAS.compare(r1.lowerbound, r2.lowerbound);
        const ubRelation = CAS.compare(r1.upperbound, r2.upperbound);
        
        const ret = new Range(r1.variable, String(-Infinity), String(Infinity));

        if (lbRelation === '=' || lbRelation === '>' || lbRelation === '>=') ret.lowerbound = r1.lowerbound;
        else if (lbRelation === '<') ret.lowerbound = r2.lowerbound;

        if (ubRelation === '=' || ubRelation === '>' || ubRelation === '>=') ret.upperbound = r1.upperbound;
        else if (ubRelation === '<') ret.upperbound = r2.upperbound;

        return ret;
    }

    static intersectRanges(r1?: Range, r2?: Range) : Range {
        if (!r1 && !r2) return new Range('NULL', String(-Infinity), String(Infinity));
        else if (!r1) {
            return r2!;
        } else if (!r2) {
            return r1;
        }

        const lbRelation = CAS.compare(r1.lowerbound, r2.lowerbound);
        const ubRelation = CAS.compare(r1.upperbound, r2.upperbound);

        const ret = new Range(r1.variable, String(-Infinity), String(Infinity));

        if (lbRelation === '=' || lbRelation === '>' || lbRelation === '>=') ret.lowerbound = r1.lowerbound;
        else if (lbRelation === '<') ret.lowerbound = r2.lowerbound;

        if (ubRelation === '=' || ubRelation === '>' || ubRelation === '>=') ret.upperbound = r2.upperbound;
        else if (ubRelation === '<') ret.upperbound = r1.upperbound;

        return ret;

    }
}

