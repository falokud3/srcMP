
import * as Xml from './Xml.js'

import * as xpath from 'xpath';



/**
 * Represents XML Element, like `<person/>`, not XML Comments or Attributes
 */
export class XmlElement {
    /**
     * Represents Node from the DOM API that contorls the behavior of an element
     */
    private domNode: Element; 
    

    /**
     * Offset needed to get the true line number of an xml element. This is caused by
     * the additional unit element added to srcML output
     * * This is specific to this program's use of srcML, where only file is converted at a time
     */
    static LINE_NUMBER_OFFSET = 1;

    public constructor(libxml: Element) {
        this.domNode = libxml;
    }

    /**
     * Returns the internal xml object used by the library
     * * Use of this is discouraged, due to the increase in coupling
     */
    public get domElement(): Element {
        return this.domNode;
    }

    public get line(): number {
        throw new Error("NOT IMPLEMENTED YET");
    }

    /**
     * Returns the text within an xml tag including the text of all child nodes
     */
    public get text() : string {
        return this.domNode.textContent ?? "";
    }

    /**
     * Returns the text within the tag, like parent from `<parent/>`
     */
    public get name() : string {
        return this.domNode.tagName;
    }

    /**
     * Returns an array of all the child Elements
     */
    public get elementChildren() : XmlElement[] {
        const children = this.domNode.childNodes;
        const ret: XmlElement[] = [];
        for (const childNode of Array.from(children)) {
            
            if (!isDomElement(childNode)) {
                continue;
            }
            ret.push(new XmlElement(childNode));
        }
        return ret;
    }

    /**
     * Returns the parent Element of the caller or null if the Element has no
     * parent Element
     */
    public get parentElement() : XmlElement | null {
        return this.domNode.parentElement ? new XmlElement(this.domNode.parentElement) : null;
    }

    /**
     * Returns the previous sibling element or null
     */
    public get prevElement() : XmlElement | null {
        return this.domNode.previousElementSibling 
            ? new XmlElement(this.domNode.previousElementSibling) : null;
    }

    /**
     * Returns the next sibling element or null
     */
    public get nextElement() : XmlElement | null {
        return this.domNode.nextElementSibling 
            ? new XmlElement(this.domNode.nextElementSibling) : null;

    }

    public child(index: number) : XmlElement | null {
        const children = this.elementChildren;
        return children.length > 0 && index < children.length ? children[index] : null;
    }

    /**
     * Gets an XML Element based on xpath
     * @param xpathString 
     * @param namespace 
     * @returns the first Element found or null if none are found
     */
    public get(xpathString: string, namespace: Record<string, string> = Xml.ns) : XmlElement | null {
        const queryResult = this.find(xpathString, namespace);
        return queryResult.length > 0 ? queryResult[0] : null;
    }

    /**
     * Gets XML Elements based on xpath
     * @param xpathString 
     * @param namespace 
     * @returns an array of Element objects (empty if none are found)
     */
    public find(xpathString: string, namespace: Record<string, string> = Xml.ns) : XmlElement[] {
        // TODO: Experment with NS resolver to avoid xmlns: for everything

        const namespaceSelect = xpath.useNamespaces(namespace)
        const queryResult = namespaceSelect(xpathString, this.domNode, false);

        if (queryResult instanceof Node) {
            if (isDomElement(queryResult)) {
                if (queryResult.tagName === "for") {
                    return [new Xml.ForLoop(queryResult)];
                } else {
                    return [new XmlElement(queryResult)];
                }
            }
        } else if (Array.isArray(queryResult)) {
            const ret: XmlElement[] = [];
            for (const node of queryResult) {
                if (isDomElement(node)
                    && node.tagName === "for") {
                    return [new Xml.ForLoop(node)];
                } else if (isDomElement(node)) {
                    return [new XmlElement(node)];
                }
            }
            return ret;
        }

        return [];
    }

    // TODO : Allow use of xpath boolean, number, string values

    /**
     * Returns whether an element contains the xpath passed in 
     * @param xpathString 
     * @param namespace 
     * @returns true if contains, false otherwise
     */
    public contains(xpathString: string, namespace: Record<string, string> = Xml.ns) : boolean {
        return this.get(xpathString, namespace) != null
    }

    /**
     * Checks if the XML element contains a name node with specified text
     * @param xpathString
     * @param namespace 
     * @returns 
     */
    public containsName(xpathString: string, namespace: Record<string, string> = Xml.ns) : boolean {
        return this.find(`.//xmlns:name[text()='${xpathString}']`, namespace).length != 0;
    }

    public getAttribute(name: string) : string | null {
        return this.domNode.getAttribute(name);
    }

    /**
     * Setting an attribute on the Xml Element, like count from `<line count=2 />`
     * @param name the name of the attribute to set
     * @param value the value to set the attribute
     */
    public setAttribute(name: string, value: string) : void {
        this.domNode.setAttribute(name, value);
    }

    /**
     * Returns the function that directly enclosing the calling element. If the
     * object is not nested within a function, it returns the root element of 
     * the document, which should be the `<unit>` tag
     * @returns 
     */
    public get enclosingFunction() : XmlElement | null {
        const func = this.find("./ancestor::xmlns:function");
        if (func.length > 0) {
            return func[-1];
        } else {
            const root = this.domElement.getRootNode()
            return isDomElement(root) ? new XmlElement(root) : null
        }
    }

    public toString() : string {
        return this.domNode.toString();
    }

    /**
     * Decided to not use lodash to reduce number of dependecies
     * @param otherElement 
     */
    public equals(otherElement: XmlElement) : boolean {
        return this.toString() === otherElement.toString();
    }


    /**
     * Returns all the symobs
     */
    public get defSymbols() : Set<Xml.XmlElement> {
        const ret = new Set<Xml.XmlElement>();
        for (const name of this.defList) {
            const innerName = name.get("./xmlns:name")
            ret.add(innerName ? innerName : name);
        }
        return ret;
    }

    /**
     * Returns a list of all the instances of identifiers be assigned within the
     * element. They are returned as the outermost `<name>` tag, so arrays will 
     * have their index and objects will contain their . operators
     */
    public get defList() : Xml.XmlElement[] {
        const defList: Xml.XmlElement[] = [];
        const names = this.find(".//xmlns:name[count(ancestor::xmlns:name)=0]");
        for (const name of names) {
            const nextOp = name.nextElement;
            const prevOp = name.prevElement;

            if (prevOp?.name === "operator" 
            && (prevOp?.text === "--" || prevOp?.text === "++")) {
                defList.push(name);
            }

            if (nextOp?.name === "operator" 
            && ([...nextOp?.text].filter((char) => char === '=' ).length === 1
                || nextOp?.text === "--" || nextOp?.text === "++")) {
                defList.push(name);
            }
            
        }
        return defList;

    }

}

/**
 * Returns true if the DOM Node passed in is an element
 * 
 * Needed due to fact that the DOM API does not acutally exist on Node
 */
function isDomElement(node: Node): node is Element {
    return node.nodeType === node.ELEMENT_NODE;
}