// Directed Dependency Graph


class DirectedDependencyGraph {
    private nodes : Map<string, Node>;

    constructor() {
        this.nodes = new Map<string, Node>();
    }

    addVertex(node: Node) : void {
        if (!this.nodes.has(node.getName())) {
            this.nodes.set(node.getName(), node);
        }
    }

    findNode(nodeName: string) : Node {
        return this.nodes[nodeName];
    }

    addEdge(from: Node, to: Node) : void {
        this.addVertex(from);
        this.addVertex(to);
        this.nodes.get(from.getName()).addChild(to);
    }

    isCyclic() : boolean {
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

    private isCyclicUtil(vertex: Node, visitedNodes: string[], recursiveStack: string[]) : boolean {
        const nodeName : string = vertex.getName();


        if (recursiveStack.includes(nodeName)) return true;

        if (visitedNodes.includes(nodeName)) return false;

        visitedNodes.push(nodeName);
        recursiveStack.push(nodeName);

        let children = vertex.getChildren();

        for (const node of children) {
            if (this.isCyclicUtil(node, visitedNodes, recursiveStack)) {
                return true;
            }
        }

        return false;

    }

    toString() : string {
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
    protected children : Set<Node>

    public constructor(name: string, data_type: string) {
        this.name = name;
        this.data_type = data_type;
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