import * as xml from 'libxmljs2'

class DependenceVector {


    private directionVector: Map<xml.Element, number> = new Map<xml.Element, number>();
    private dirVec: Map<string, number> = new Map<string, number>();

    public constructor(loop_nest: xml.Element[] = []) {
        loop_nest.forEach( (loop: xml.Element) => {
            this.directionVector.set(loop, Direction.any);
            this.dirVec.set(loop.text(), Direction.any)
        });
    }

    public setDirection(loop: xml.Element, dir: Direction) {
        this.directionVector.set(loop, dir);
        this.dirVec.set(loop.text(), dir)
    }

    public getDirection(loop: xml.Element, string_version: boolean) {
        if (string_version) {
            return this.dirVec.get(loop.text());
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
        return clone;
    }

    public toString() : string {
        for (const loop of this.directionVector.keys()) {
            console.log(this.directionVector.get(loop));
        }
        return "";
    }
}

enum Direction {
    any = 0, 
    less, 
    equal, 
    greater
}

export {DependenceVector, Direction as DependenceDir}