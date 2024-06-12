import * as Xml from '../Xml/Xml.js'

class DependenceVector {

    //TODO: remove Xml.Element version
    private directionVector: Map<Xml.Element, number> = new Map<Xml.Element, number>();
    private dirVec: Map<string, number> = new Map<string, number>();

    private _valid: boolean = true;

    public constructor(loop_nest: Xml.Element[] = []) {
        loop_nest.forEach( (loop: Xml.Element) => {
            this.directionVector.set(loop, Direction.any);
            this.dirVec.set(loop.text, Direction.any)
        });
    }

    public get valid() : boolean {
        return this._valid;
    }

    public set valid(value: boolean) {
        this._valid = value;
    }

    public get loops() : Xml.ForLoop[] {
        return Array.from(this.directionVector.keys()) as Xml.ForLoop[];
    }

    public get isPlausibleVector() : boolean {
        for (const vector of this.directionVector) {
            if (vector[1] === Direction.greater) return false;
            else if (vector[1] === Direction.less) return true;
        }
        return true;
    }

    public get reverseVector() : DependenceVector {
        const ret = this.clone();
        for (const vector of this.directionVector) {
            if (vector[1] === Direction.less) {
                ret.setDirection(vector[0], Direction.greater);
            } else if (vector[1] === Direction.greater) {
                ret.setDirection(vector[0], Direction.less);
            }
        }
        return ret;
    }

    public setDirection(loop: Xml.Element, dir: Direction) {
        this.directionVector.set(loop, dir);
        this.dirVec.set(loop.text, dir)
    }

    public getDirection(loop: Xml.Element, string_version: boolean = true) : Direction | undefined {
        if (string_version) {
            return this.dirVec.get(loop.text);
        } else {
            return this.directionVector.get(loop);
        }
    }

    public containsDirection(keyDir: Direction) : boolean {
        for (const dir of this.directionVector.values()) {
            if (dir == keyDir) return true;
        }
        return false;
    }

    public isAllEqual() : boolean {
        for (const dir of this.directionVector.values()) {
            if (dir != Direction.equal) return false;
        }
        return true;
    }

    public clone(): DependenceVector {
        const clone = new DependenceVector();
        clone.directionVector = new Map(this.directionVector);
        clone.dirVec = new Map(this.dirVec);
        clone._valid = this._valid;
        return clone;
    }

    public equals(other: DependenceVector) : boolean {
        if (this._valid !== other._valid) return false;

        for (const entry of this.dirVec) {
            if (other.dirVec.get(entry[0]) !== entry[1]) return false;
        }

        return true;
    }

    public toString() : string {
        for (const loop of this.directionVector.keys()) {
            console.log(this.directionVector.get(loop));
        }
        return "";
    }

    public mergeWith(other: DependenceVector) : void {
        let newDir: Direction;
        for (const loop of other.loops) {
            if (!this.dirVec.has(loop.toString())) {
                this.directionVector.set(loop, other.getDirection(loop)!)
                this.dirVec.set(loop.toString(), other.getDirection(loop)!);
                continue;
            }
            
            const thisDir = this.getDirection(loop) ?? Direction.nil;
            const thatDir = other.getDirection(loop) ?? Direction.nil; 

            newDir = thisDir !== Direction.nil ? cartesianProduct[thisDir][thatDir] : Direction.nil;
            if (newDir === Direction.nil) this.valid = false;
            this.directionVector.set(loop, other.getDirection(loop)!)
            this.dirVec.set(loop.toString(), other.getDirection(loop)!);
        }
    }
}

export function mergeVectorSets(dvs: DependenceVector[], other: DependenceVector[]) : void {
    if (dvs.length === 0) {
        dvs.push(...other);
        return;
    }

    let size = dvs.length;
    // TODO: INVESTIGATE ThIS ALGORITHM
    while (size-- > 0) {
        const dv = dvs.shift()!;
        for (let i = 0; i < other.length; i++) {
            const mergedDV = dv.clone()
            mergedDV.mergeWith(other[i]);
            if (mergedDV.valid) dvs.push(mergedDV);
        }
    }
}

// TODO: Refactor
enum Direction {
    nil = -1,
    any = 0, 
    less, 
    equal, 
    greater
}

const cartesianProduct: number[][] = [
    [Direction.any, Direction.less, Direction.equal, Direction.greater],
    [Direction.less, Direction.less, Direction.nil, Direction.nil],
    [Direction.equal, Direction.nil, Direction.equal, Direction.nil],
    [Direction.greater, Direction.nil, Direction.nil, Direction.greater],
];

export {DependenceVector, Direction as DependenceDir}