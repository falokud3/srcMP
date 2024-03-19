import * as xml from 'libxmljs2'

class DependenceVector {


    private directionVector: Map<xml.Element, number> = new Map<xml.Element, number>();

    public constructor(loop_nest: xml.Element[] = []) {
        loop_nest.forEach( (loop: xml.Element) => {
            this.directionVector.set(loop, Direction.any);
        });
    }

    public setDirection(loop: xml.Element, dir: Direction) {
        this.directionVector.set(loop, dir);
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
        return clone;
    }
}

enum Direction {
    any = 0, 
    less, 
    equal, 
    greater
}

export {DependenceVector, Direction as DependenceDir}