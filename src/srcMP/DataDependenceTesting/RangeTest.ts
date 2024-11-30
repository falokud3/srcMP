import { DependenceVector } from './DependenceVector';
import { SubscriptPair } from './SubscriptPair';

class RangeTest {

    private f;
    private g;
    private pair: SubscriptPair;

    public constructor(pair: SubscriptPair) {
        this.f = pair.getSubscript1();
        this.g = pair.getSubscript2();
        this.pair = pair;
        
        throw new Error("DONE!");
    }

    public pairIsElligible() : boolean {
        throw new Error('Method not implemented.');
    }

    public testDependence(dv: DependenceVector) : boolean {
        const ret: boolean = true;
        
        return ret;
    }

    private solve() {

    }
    
    public get subscriptPair() {
        return this.pair;
    }
}

export {RangeTest};