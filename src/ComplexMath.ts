/**
 * Abstract interface for complex athematical operations to reduce coupling
 */
import * as cortex from '@cortex-js/compute-engine';

let Engine = new cortex.ComputeEngine();

/**
 * Represents Mathematical Expressions
 */
export class Expression {
    private cortexExpression: cortex.BoxedExpression; 

    public constructor(cortexExpression: cortex.BoxedExpression) {
        this.cortexExpression = cortexExpression;
    }

    /**
     * Return true if the expression can be determined to be 0, false if it can
     * be determined to not be 0, and null if the value of the expression cannot
     * be determined
     */
    public get isZero() : boolean {
        return this.cortexExpression.isZero ? this.cortexExpression.isZero : null;
    }
}

/**
 * The engine associates values to symoblic variables. Reseting the engine removes
 * those asssociations
 */
export function resetEngine() {
    Engine = new cortex.ComputeEngine();
}

/**
 * Takes a mathematical expression represented as a string and returns an 
 * expression object representing it
 * @param expression
 */
export function simplify(expression: string) : Expression {
    return new Expression(Engine.parse(expression).simplify()); 
}