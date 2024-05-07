
export class RangeDomain {
    private ranges: Map<string, VariableRange>;

    public constructor() {
        this.ranges = new Map<string, VariableRange>();
    }

    public getRange(variable: string) : VariableRange {
        return this.ranges.get(variable);
    }

    public setRange(variable: string, lowerbound: number, upperbound: number) {
        this.ranges.set(variable, new VariableRange(variable, lowerbound, upperbound));
    }

    public setVarRange(variable: string, varrange: VariableRange) {
        this.ranges.set(variable, varrange);
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

    public unionRange(variable: string, varRange: VariableRange) : void {
        const result = RangeDomain.unionVarRanges(variable, this.getRange(variable), 
            varRange);
        this.setVarRange(variable, result);
    }

    public unionRanges(otherRange: RangeDomain) : void {
        const vars = new Set(this.ranges.keys());
        for (const key of otherRange.ranges.keys()) {
            vars.add(key);
        }
        for (const variable of vars) {
            const result = RangeDomain.unionVarRanges(variable, this.getRange(variable), 
                otherRange.getRange(variable));
            this.setVarRange(variable, result);
        }
    }

    public static unionVarRanges(variable: string, r1: VariableRange, r2: VariableRange) : VariableRange {
        if (r1 === undefined) {
            return r2;
        } else if (r2 === undefined) {
            return r1;
        }

        const LB = Math.min(r1.lowerbound, r2.lowerbound);
        const UB = Math.max(r1.upperbound, r2.upperbound);
        return new VariableRange(variable, LB, UB);
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

export class VariableRange {
    // TODO: FLOAT
    public lowerbound: number;
    public upperbound: number;
    public variable: string;

    public constructor(variable: string, lowerbound: number, upperbound: number) {
        this.variable = variable;
        this.lowerbound = lowerbound;
        this.upperbound = upperbound;
    }

    public getLowerBound() : number {
        return this.lowerbound;
    }
    
    public isConstant() : boolean {
        return this.lowerbound === this.upperbound;
    }

    public equals(other: VariableRange) : boolean {
        return this.lowerbound === other.lowerbound &&
            this.upperbound === other.upperbound;
    }

    public toString() : string {
        return `${this.lowerbound} < ${this.variable} < ${this.upperbound}`;
    }
}

