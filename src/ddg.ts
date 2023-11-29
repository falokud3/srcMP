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
        return output;
    }
}

// Node represents a variable
class Node {
    // May replace with symbol table ID and lookup methods
    private name : string;
    private data_type : string; // may replace with enum
    private children : Set<Node>

    public constructor(name: string, data_type: string) {
        this.name = name;
        this.data_type = data_type;
        this.children = new Set<Node>();
    }

    public Node(name: string) {
        this.name = name;
        this.data_type = "NULL";
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
        let output: string = this.name + " ->";
        this.children.forEach((node) => {output += " " + node.name;});
        output += "\n";
        return output;
    }

}

export {Node as DDG_Node, DirectedDependencyGraph as DDG}