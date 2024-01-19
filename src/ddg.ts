import * as libxmljs from 'libxmljs2'
import * as XmlTools from './XmlTools.js'
import assert from 'assert';


// Directed Dependency Graph


class DirectedDependencyGraph {
    private nodes : Map<string, Node>;

    public constructor() {
        this.nodes = new Map<string, Node>();
    }

    /**
     * Add a node to the graph if a node with the same is not already in the
     * graph
     * @param node the node to be added to the Graph
     * @return true if a vertex is sucessfully added
     */
    public addVertex(node: Node) : boolean {
        if (!this.nodes.has(node.getName())) {
            this.nodes.set(node.getName(), node);
            return true;
        }
        return false;
    }

    /**
     * Finds a node by using a string name as a key
     * @param nodeName the string used as a key
     * @returns the node associated with that name or undefined if not found
     */
    public findNode(nodeName: string) : Node {
        return this.nodes[nodeName];
    }

    /**
     * Adds a directed edge from the source node to the target node; This method also
     * adds nodes to the graph if they are not already in the graph
     * @param source the source node
     * @param target the target node
     */
    public addEdge(source: Node, target: Node) : void {
        this.addVertex(source);
        this.addVertex(target);
        this.nodes.get(source.getName()).addChild(target);
    }

    /**
     * Uses a depth-frist search approach to determine if there are cycles in 
     * the graph
     * @returns true if a cycle exists, false otherwise
     */
    public isCyclic() : boolean {
        const vertices = this.nodes.size;
        let visitedNodes : string[] = [];
        let recursiveStack : string[] = [];

        for (const node of this.nodes.values()) {
            if (!visitedNodes.includes(node.getName()) 
            && this.isCyclicUtil(node, visitedNodes, recursiveStack)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Recursive helper method used to determine if the graph is cyclic
     * @param vertex the current node being visted
     * @param visitedNodes an array of previously visited nodes
     * @param currentPath the current path of nodes taken
     * @returns true if a cycle exists, false otherwise
     */
    private isCyclicUtil(vertex: Node, visitedNodes: string[], currentPath: string[]) : boolean {
        const nodeName : string = vertex.getName();

        if (currentPath.includes(nodeName)) return true;

        if (visitedNodes.includes(nodeName)) return false;

        visitedNodes.push(nodeName);
        currentPath.push(nodeName);

        let children = vertex.getChildren();

        for (const node of children) {
            if (this.isCyclicUtil(node, visitedNodes, currentPath)) {
                return true;
            }
        }

        currentPath.pop();
        return false;

    }

    /**
     * Returns a string representation of the graph in the form:
     * node -> adjacent nodes
     * @returns 
     */
    public toString() : string {
        let output: string = "";
        // this.nodes.forEach((node: Node) => {output += node.toString();});
        let entries = this.nodes.entries();
        this.nodes.forEach((node) => {
            output += node.toString();
        })
        return output.substring(0, output.length - 1);
    }
}

// Node represents a variable
class Node {
    // May replace with symbol table ID and lookup methods
    private name : string;
    private children : Set<Node>

    public constructor(name: string) {
        this.name = name;
        this.children = new Set<Node>();
    }

    public getName() : string {return this.name;}

    // creates edge from this node to the input node
    public addChild(child: Node): void {
        this.children.add(child);
    }

    //public removeChild(child: Node): void {}

    public getChildren() : Set<Node> {return this.children;}

    public hasChild(child: Node) : boolean {
        return this.children.has(child);
    }

    public toString(): string {
        let output: string = "";
        this.children.forEach((node) => {
            output += this.name + " -> " + node.name + "\n";
        });
        return output;
    }

}

/**
 * Takes a for loop xml element and builds a ddg from it. Throws an error
 * if the passed element is not a for loop
 * @param root the for xml element
 * @returns DDG representingb the for loop
 */
function buildLoopDDG(root: libxmljs.Element) : DirectedDependencyGraph {
    assert(root.name() === 'for');

    const ddg = new DirectedDependencyGraph();
    // go through all the declarations
    const decl_statements = root.find(".//xmlns:decl", XmlTools.ns) as libxmljs.Element[]; 
    decl_statements.forEach((decl) => {
        // TODO: add clause for INIT
        const type = (decl.get('./xmlns:type/xmlns:name', XmlTools.ns) as libxmljs.Element).text();
        const name = (decl.get('./xmlns:name', XmlTools.ns) as libxmljs.Element).text();
        ddg.addVertex(new Node(name));
    });

    // get all expressions
        // filter for init and assignment operator
    const expr_statements = root.find(".//xmlns:expr", XmlTools.ns) as libxmljs.Element[];
    expr_statements.forEach((expr) => {

        // check if expr has init parent
        const parent = expr.parent() as libxmljs.Element;
        if (parent.name() === 'init') {
            // parent of <init> is <decl> 
                // the first <name> child of <decl> is the 
                // variable being declared
            const target = parent.get('../xmlns:name', XmlTools.ns) as libxmljs.Element;

            // all <name> children of the expr are variables to be added to ddg
                // NOTE: unsure how constants appear as srcml
                // NOTE: object dependencies won't work yet;
            const vars = expr.find('./xmlns:name', XmlTools.ns) as libxmljs.Element[];
            vars.forEach((variable) => {
                ddg.addEdge( new Node(target.text()), new Node(variable.text()));
            });
        }

        // check if expr has <operator>=</operator>
        const operators = expr.find('./xmlns:operator', XmlTools.ns) as libxmljs.Element[];

        // using for over .reduce to avoid looping through entire array
            // unnecessarily
        let isAssignment = false;
        for (let i = 0; i < operators.length && !isAssignment; i++) {
            isAssignment = (operators[i].text() === '=');
        }
        if (isAssignment) {
            const variables = expr.find('./xmlns:name', XmlTools.ns) as libxmljs.Element[];
            const target = variables[0];
            variables.slice(1).forEach((source) => {
                ddg.addEdge(new Node(target.text()), new Node(source.text()))
            });
        }

    });

    console.log(ddg.toString());

    // TODO: treat array indices as variables
    // TODO: object members

    // variable usage through the name
        // exception: method names have call parent
        // exception: data types have type parent
        // TODO: figure out how to isolate the name of an object without
            // the method/attribute (vector.push_back())
    return ddg;
}

function nameToNode(name: libxmljs.Element) : Node {
    return new Node('');
}

export {DirectedDependencyGraph}