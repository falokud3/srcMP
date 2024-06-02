

/**
 * Typescript wrapper for the Computer Algebra System library Algebrite.
 * The wrapper does not include all functionality from the library; functions
 * and classes can be added as necessary
 * 
 * @see http://algebrite.org/docs/1.4.0/reference.html
 */
declare module 'algebrite' {

    /**
     * Class that represents the epxressions returned by algebrite functions.
     * Unsure what mathematical concept U stands for, potentially the universal
     * set.
     */
    export class U {
        
        /**
         * Human readable version of the expression
         */
        toString() : string;

        /**
         * Latex Version of the expression
         */
        toLatexString() : string;
    }

    /**
     * Returns a simplified version of the input expression
     * @param expression the expression to be simplified as a string
     */
    export function simplify(expression: string) : U;

    export function subst(newValue: string, oldValue: string, expression: string) : U;


}