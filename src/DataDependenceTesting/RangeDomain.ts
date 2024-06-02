import { assert } from 'console';
import * as Xml from '../Xml/Xml.js'

export class RangeDomain {
    private ranges: Map<string, Range>;

    public constructor() {
        this.ranges = new Map<string, Range>();
    }

    public copy() : RangeDomain {
        const copy = new RangeDomain();
        for (const range of this.ranges) {
            copy.ranges.set(range[0], range[1].copy());
        }
        return copy;
    }

    public getRange(variable: string) : Range {
        const range = this.ranges.get(variable);
        if (!range) throw new Error("Range not found");
        return range;
    }

    public setRange(variable: string, lowerbound: string, upperbound: string) {
        this.ranges.set(variable, new Range(variable, lowerbound, upperbound));
    }

    public removeRange(variable: string) : boolean {
        return this.ranges.delete(variable);
    }

    public setVarRange(variable: string, varrange: Range) {
        this.ranges.set(variable, varrange);
    }

    public killArraysWith(symbol: Xml.XmlElement) {
        assert(symbol.name === "name")
        const symbolName = symbol.text
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

    public replaceSymbol(original: Xml.XmlElement, replacement: Xml.XmlElement | Range | undefined | null) {
        if (!replacement) return;

        for (const range of this.ranges) {
            if (replacement instanceof Range) {
                if (replacement.isConstant) {
                    range[1].lowerbound = range[1].lowerbound.replace(original.text, replacement.lowerbound)
                    range[1].upperbound = range[1].upperbound.replace(original.text, replacement.upperbound)
                }
            } else {
                range[1].lowerbound = range[1].lowerbound.replace(original.text, replacement.text)
                range[1].upperbound = range[1].upperbound.replace(original.text, replacement.text)
            }
        }
    }

    // returns the expression with all occurances of the specified variable replaced with
    // their value ranges
    public getExpandedExpression(expression: Xml.XmlElement | string, variable: Xml.XmlElement | string) : string | null {

        if (expression instanceof Xml.XmlElement && variable instanceof Xml.XmlElement) {
            const range = this.getRange(variable.text);
            return range && range.isConstant ? expression.text.replace(variable.text, range.lowerbound) 
                : null;
        } else {
            const expr = <string> expression;
            const vari = <string> variable;
            return expr.replace(vari, expr);
        }


    }

    // updates the ranges by replacing the variable with its range 
    public expandRangeExpressions(variable: Xml.XmlElement) : void {
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
        return this.ranges.size == 0;
    }

    public equals(other: RangeDomain) : boolean {
        for (const variable of this.ranges.keys()) {
            if (!this.getRange(variable).equals(other.getRange(variable))) {
                return false;
            }
        }
        return true;
    }

    public narrowRanges(other: RangeDomain) {
        for (const variable of other.ranges.keys()) {
            const result = RangeDomain.narrowVarRanges(other.getRange(variable), this.getRange(variable), this);
            if (result.isOmega) {
                this.removeRange(variable);
            } else {
                this.setVarRange(variable, result);
            }
        }

    }

    public widenAffectedRanges(other: RangeDomain, vars: Set<Xml.XmlElement> = new Set<Xml.XmlElement>()) {
        const affected = new Set<string>();
        for (const var_range of other.ranges.keys()) {
            for (const varIn of vars) {
                if (this.getRange(var_range).variable === varIn.text) {
                    affected.add(var_range);
                }
            }
        }
        
        for (const val of vars) {
            affected.add(val.text);
        }

        for (const affect of affected) {
            const result = RangeDomain.widenRange(other.getRange(affect), this.getRange(affect), this);
            if (result.isOmega) {
                this.removeRange(affect)
            } else {
                this.setVarRange(affect, result)
            }
        }
    }

    public unionRange(variable: string, varRange: Range) : void {
        const result = RangeDomain.unionVarRanges(variable, this.getRange(variable), 
            varRange);
        if (result.isOmega) {
            this.removeRange(variable);
        } else {
            this.setVarRange(variable, result);
        }
    }

    public unionRanges(otherRange: RangeDomain) : void {
        const vars = new Set(this.ranges.keys());
        for (const key of otherRange.ranges.keys()) {
            vars.add(key);
        }
        for (const variable of vars) {
            const result = RangeDomain.unionVarRanges(variable, this.getRange(variable), 
                otherRange.getRange(variable));
            if (result.isOmega) {
                this.removeRange(variable)
            } else {
                this.setVarRange(variable, result);
            }
        }
    }

    public static widenRange(e: Range, widen: Range, rd: RangeDomain) : Range {
        if (e.isOmega || widen.isOmega){
            // TODO: CHECK EXPECTED BEHAVIOR
            return e;
        }
        // ! isEmpty

        if (e.lowerbound !== widen.lowerbound) {
            e.lowerbound = String(-Infinity);
        }

        if (e.upperbound !== widen.upperbound) {
            e.upperbound = String(Infinity);
        }

        return e;
    }

    public static unionVarRanges(variable: string, r1: Range, r2: Range) : Range {
        if (r1 === undefined) {
            return r2;
        } else if (r2 === undefined) {
            return r1;
        }

        // const LB = Math.min(r1.lb, r2.lb);
        // const UB = Math.max(r1.ub, r2.ub);
        throw new Error("NOT IMPLEMENTED")
        // return new Range(variable, "", "");
    }

    // TODO REVIEW
    public static narrowVarRanges(e: Range, narrow: Range, rd: RangeDomain) : Range {
        if (narrow.isOmega) {
            return e;
        } else if (e.isOmega) {
            return narrow;
        }

        if (Number(e.lowerbound) === -Infinity) {
            e.lowerbound = narrow.lowerbound;
        }

        if (Number(e.upperbound) === Infinity) {
            e.upperbound = narrow.upperbound;
        }

        return e;
    }

    public toString() : string {
        let ret = "[";
        for (const variable of this.ranges.keys()) {
            ret += this.getRange(variable).toString() + ", "
        }
        if (!this.isEmpty()) ret = ret.substring(0, ret.length - 2);
        ret += "]";
        return ret;
    }

}

export class Range {
    
    // TODO: FLOAT
    private lb: string;
    private ub: string;
    public variable: string;

    public constructor(variable: string, lowerbound: string, upperbound: string) {
        this.variable = variable;
        this.lb = lowerbound;
        this.ub = upperbound;
    }

    public copy() : Range {
        return new Range(this.variable, this.lb, this.ub);
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

    public equals(other: Range) : boolean {
        return this.lb === other.lb &&
            this.ub === other.ub;
    }

    public toString() : string {
        return this.isConstant ? `${this.variable} = ${this.lb}`
            :`${this.lb} < ${this.variable} < ${this.ub}`;
    }
}

