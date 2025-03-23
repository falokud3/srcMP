import * as Xml from '../../common/Xml/Xml.js';

class DependenceVector {

    //TODO: Refactor to only use one - there's a reason I did this, but I forget (regardless it seems very poorly written)
    private directionVector: Map<Xml.Element, number> = new Map<Xml.Element, number>();

    public valid: boolean = true;

    public constructor(loop_nest: Xml.Element[] = []) {
        loop_nest.forEach( (loop: Xml.Element) => {
            this.directionVector.set(loop, Direction.any);
        });
    }

    /**
     * Returns true if the first non NIL, *, = direction is <
     * If there are no < or >, it allows returns true.
     */
    public get isPlausibleVector() : boolean {
        for (const vector of this.directionVector.values()) {
            if (vector === Direction.greater.valueOf()) return false;
            else if (vector === Direction.less.valueOf()) return true;
        }
        return true;
    }

    /** 
     * Returns a version of the dependece vector where 
     * all the > to < and all the < to > in the direction vector
     */
    public get reverseVector() : DependenceVector {
        const ret = this.clone();
        for (const vector of this.directionVector) {
            if (vector[1] === Direction.less.valueOf()) {
                ret.setDirection(vector[0], Direction.greater);
            } else if (vector[1] === Direction.greater.valueOf()) {
                ret.setDirection(vector[0], Direction.less);
            }
        }
        return ret;
    }

    public setDirection(loop: Xml.Element, dir: Direction) {
        for (const key of this.directionVector.keys()) {
            if (key.text === loop.text) {
                this.directionVector.set(key, dir);
                return;
            }
        }
        this.directionVector.set(loop, dir);
    }

    public getDirection(loop: Xml.Element) : Direction | undefined {
        for (const key of this.directionVector.keys()) {
            if (key.text === loop.text) {
                return this.directionVector.get(key);
            }
        }
        return undefined;
    }

    public containsDirection(keyDir: Direction) : boolean {
        for (const dir of this.directionVector.values()) {
            if (dir === keyDir.valueOf()) return true;
        }
        return false;
    }

    public isAllEqual() : boolean {
        for (const dir of this.directionVector.values()) {
            if (dir !== Direction.equal.valueOf()) return false;
        }
        return true;
    }

    public clone(): DependenceVector {
        const clone = new DependenceVector();
        clone.directionVector = new Map(this.directionVector);
        clone.valid = this.valid;
        return clone;
    }

    public equals(other: DependenceVector) : boolean {
        if (this.valid !== other.valid) return false;

        for (const entry of this.directionVector) {
            if (other.getDirection(entry[0])?.valueOf() !== entry[1]) return false;
        }

        return true;
    }

    public toString() : string {
        const ret: string[] = [];
        for (const dir of this.directionVector.values()) {
            ret.push(directionToString(dir));
        }
        return ret.join(',');
    }
    
    private hasKey(searchKey: Xml.Element) : boolean {
        for (const key of this.directionVector.keys()) {
            if (key.text === searchKey.text) {
                return true;
            }
        }
        return false;
    }

    public mergeWith(other: DependenceVector) : void {
        let newDir: Direction;
        const loops =  Array.from(this.directionVector.keys()) as Xml.ForLoop[];
        for (const loop of loops) {
            if (!this.hasKey(loop)) {
                this.directionVector.set(loop, other.getDirection(loop)!);
                continue;
            }
            
            const thisDir = this.getDirection(loop) ?? Direction.nil;
            const thatDir = other.getDirection(loop) ?? Direction.nil; 

            newDir = thisDir !== Direction.nil ? cartesianProduct[thisDir][thatDir] : Direction.nil;
            if (newDir === Direction.nil) this.valid = false;
            this.directionVector.set(loop, other.getDirection(loop)!);
        }
    }
}

export function mergeVectorSets(dvs: DependenceVector[], other: DependenceVector[]) : void {
    if (dvs.length === 0) {
        dvs.push(...other);
        return;
    }

    let size = dvs.length;
    while (size-- > 0) {
        const dv = dvs.shift()!;
        for (let i = 0; i < other.length; i++) {
            const mergedDV = dv.clone();
            mergedDV.mergeWith(other[i]);
            if (mergedDV.valid) dvs.push(mergedDV);
        }
    }
}

export enum Direction {
    nil = -1,
    any = 0, 
    less, 
    equal, 
    greater,
}
function directionToString(dir: Direction) : string {
    switch (dir) {
        case Direction.nil:
            return 'NIL';
        case Direction.any:
            return '*';
        case Direction.less:
            return '<';
        case Direction.equal:
            return '=';
        case Direction.greater:
            return '>';
    }
}

const cartesianProduct: number[][] = [
    [Direction.any, Direction.less, Direction.equal, Direction.greater],
    [Direction.less, Direction.less, Direction.nil, Direction.nil],
    [Direction.equal, Direction.nil, Direction.equal, Direction.nil],
    [Direction.greater, Direction.nil, Direction.nil, Direction.greater],
];

export {DependenceVector, Direction as DependenceDir};