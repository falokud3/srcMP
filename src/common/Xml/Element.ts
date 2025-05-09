
import * as Xml from './Xml.js';

import xpath from 'xpath';


/**
 * Represents XML Element, like `<person/>`, not XML Comments or Attributes
 * ! Aliased as Xml.Element
 */
export default class XmlElement {
// default export used to be able to reference class as Xml.Element without
// name clashing with DOM Element
// TODO: replace with named export

    /**
     * Represents Node from the DOM API that contorls the behavior of an element.
     * * Node cannot access the DOM API without a 3rd party library 
     * * All the components of DOM API available are provided by the xmldom library 
     * (intellisense may show more properties than are actually accessable)
     */
    // TODO: add link
    readonly domNode: Element; 
    

    /**
     * Offset needed to get the true line number of an xml element. This is caused by
     * the additional unit element added to srcML output
     * * This is specific to this program's use of srcML, where only file is converted at a time
     */
    static LINE_NUMBER_OFFSET = 1;

    public constructor(domNode: Element) {
        this.domNode = domNode;
    }

    public get line(): number {
        //@ts-expect-error lineNumber is unique to @xmldom/xmldom's DOM implementation and not in 
            // general DOM type definition
        return this.domNode["lineNumber"] - 1; // -1, because srcml adds the unit tag
    }

    public get col(): number {
        //@ts-expect-error lineNumber is unique to @xmldom/xmldom's DOM implementation and not in 
            // general DOM type definition
        return Number(this.domNode['columnNumber']);
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
    public get childElements() : XmlElement[] {
        const children = this.domNode.childNodes;
        const ret: XmlElement[] = [];
        for (const childNode of Array.from(children)) {
            
            if (!xpath.isElement(childNode)) {
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
        return xpath.isElement(this.domNode.parentNode) 
            ? new XmlElement(this.domNode.parentNode) : null;
    }

    /**
     * Returns the previous sibling element or null
     */
    public get prevElement() : XmlElement | null {
        let curr: Node | null = this.domNode.previousSibling;
        while (curr) {
            if (xpath.isElement(curr)) {
                return new XmlElement(curr);
            }
            curr = curr.previousSibling;
        }
        return null;
    }

    /**
     * Returns the next sibling element or null
     */
    public get nextElement() : XmlElement | null {
        let curr: Node | null = this.domNode.nextSibling;
        while (curr) {
            if (xpath.isElement(curr)) {
                return new XmlElement(curr);
            }
            curr = curr.nextSibling;
        }
        return null;
    }

    public child(index: number) : XmlElement | null {
        const children = this.childElements;
        return children.length > 0 && index < children.length ? children[index] : null;
    }

    /**
     * Gets an XML Element based on xpath
     * @param xpathString 
     * @param namespace 
     * @returns the first Element found or null if none are found
     */
    public get(xpathString: string, namespace: Record<string, string> = Xml.ns()) : XmlElement | null {
        const queryResult = this.find(xpathString, namespace);
        return queryResult.length > 0 ? queryResult[0] : null;
    }

    /**
     * Gets XML Elements based on xpath
     * @param xpathString 
     * @param namespace 
     * @returns an array of Element objects (empty if none are found)
     */
    public find(xpathString: string, namespace: Record<string, string> = Xml.ns()) : XmlElement[] {
        // TODO: Experment with NS resolver to avoid xmlns: for everything

        const namespaceSelect = xpath.useNamespaces(namespace);
        const queryResult = namespaceSelect(xpathString, this.domNode, false);


        if (xpath.isArrayOfNodes(queryResult)) {
            const ret: XmlElement[] = [];
            for (const node of queryResult) {
                if (xpath.isElement(node) 
                    && node.tagName === "for") {
                    ret.push(new Xml.ForLoop(node));
                } else if (xpath.isElement(node)) {
                    ret.push(new XmlElement(node));
                }
            }
            return ret;
        } else if (xpath.isElement(queryResult)) {
            if (queryResult.tagName === "for") {
                return [new Xml.ForLoop(queryResult)];
            } else {
                return [new XmlElement(queryResult)];
            }
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
    public contains(xpathString: string, namespace: Record<string, string> = Xml.ns()) : boolean {
        return this.get(xpathString, namespace) !== null;
    }

    /**
     * Checks if the XML element contains a name node with specified text
     * @param xpathString
     * @param namespace 
     * @returns 
     */
    public containsName(xpathString: string, namespace: Record<string, string> = Xml.ns()) : boolean {
        return this.find(`.//xmlns:name[text()='${xpathString}']`, namespace).length !== 0;
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

    remove() : void {
        const domParent = this.parentElement?.domNode;

        if (!domParent) throw new Error("Cannot remove a node that doesn't have a parent element");

        domParent.removeChild(this.domNode);
    }

    insertBefore(node: XmlElement | Node) : void {
        const domParent = this.parentElement?.domNode;

        if (!domParent) throw new Error("Cannot replace a node that doesn't have a parent element");

        domParent.insertBefore(node instanceof XmlElement ? node.domNode : node, this.domNode);
        if (node instanceof XmlElement) {
            //@ts-expect-error - fixes bug where replaceChild doesn't change readonly property
            // ownerDocument for the replacing node
            node.domNode.ownerDocument = this.domNode.ownerDocument;
        } else {
            //@ts-expect-error - fixes bug where replaceChild doesn't change readonly property
            // ownerDocument for the replacing node
            node.ownerDocument = this.domNode.ownerDocument;
        }
    }

    /**
     * Returns the function that directly enclosing the calling element. If the
     * object is not nested within a function, it returns the root element of 
     * the document, which should be the `<unit>` tag
     * @returns 
     */
    public get enclosingFunction() : XmlElement {
        const func = this.find("./ancestor::xmlns:function");
        if (func.length > 0) {
            return func.at(-1)!;
        } else {
            const root = this.get("/xmlns:unit")!;
            return root;
        }
    }

    public toString() : string {
        /* eslint-disable-next-line @typescript-eslint/no-base-to-string*/ // while the DOM API does not have a toString, the @xmldom/xmldom implementation does
        return String(this.domNode);
        // return JSON.stringify(this.domNode);
    }

    /**
     * Decided to not use lodash to reduce number of dependecies
     * @param otherElement 
     */
    public equals(otherElement: XmlElement) : boolean {
        return this.toString() === otherElement.toString();
    }

    public copy() : XmlElement {
        const cloneDoc = this.domNode.ownerDocument.cloneNode(true);
        const attributes: string[] = [];
        for (const attr of Array.from(this.domNode.attributes)) {
            attributes.push(`@${attr.name} = '${attr.value}'`);
        }
        //@ts-expect-error limitiation of xmldoc
        const matches = (new XmlElement(cloneDoc['documentElement'] as Element)).find(`//${this.name.includes(':') ? this.name : 'xmlns:' + this.name}${attributes.length > 0 ? `[${attributes.join(' and ')}]` : ''}`);

        const copy = matches.find((element) => this.equals(element));

        if (!copy) throw new Error('Copy Failed');

        return copy;
    }

    public replace(newNode: XmlElement) : XmlElement {

        const domParent = this.parentElement?.domNode;

        if (!domParent) throw new Error("Cannot replace a node that doesn't have a parent element");

        domParent.replaceChild(newNode.domNode, this.domNode);
        //@ts-expect-error - fixes bug where replaceChild doesn't change readonly property
        // ownerDocument for the replacing node
        newNode.domNode.ownerDocument = this.domNode.ownerDocument;

        return newNode;
        // // no return
    }

    public get useSymbols() : Set<Xml.Element> {
        const useList: Xml.Element[] = [];
        const names = this.find(".//xmlns:name[count(ancestor::xmlns:name)=0]");
        for (const name of names) {
            if (!variableIsDefined(name)) useList.push(name);   
        }

        const ret = new Set<Xml.Element>();
        for (const name of useList) {
            const innerName = name.get("./xmlns:name");
            ret.add(innerName ? innerName : name);
        }
        return ret; 
    }

    /**
     * Returns all the symbolls 
     */
    public get defSymbols() : Set<Xml.Element> {
        const ret = new Set<Xml.Element>();
        for (const name of this.defList) {
            const innerName = name.get("./xmlns:name");
            ret.add(innerName ? innerName : name);
        }
        return ret;
    }

    /**
     * Returns a list of all the instances of identifiers be assigned within the
     * element. They are returned as the outermost `<name>` tag, so arrays will 
     * have their index and objects will contain their . operators
     */
    public get defList() : Xml.Element[] {
        const defList: Xml.Element[] = [];
        const names = this.find(".//xmlns:name[count(ancestor::xmlns:name)=0]");
        for (const name of names) {
            if (variableIsDefined(name)) defList.push(name);   
        }
        return defList;

    }

    get emptyLines() : number {
        let lines = 0;
        let prevIndex = 0;
        let index = this.text.indexOf('\n');
        while(index !== -1) {
            if (this.text.substring(prevIndex, index).trim().length === 0) lines++;
            prevIndex = index;
            index = this.text.indexOf('\n', index + 1);
        }
        return lines;
    }

}

function variableIsDefined(name: Xml.Element) : boolean {
    const nextOp = name.nextElement;
    const prevOp = name.prevElement;
    if (prevOp?.name === "operator" 
        && (prevOp?.text === "--" || prevOp?.text === "++")) {
        return true;
    }

    if (nextOp?.name === "operator" 
        && (Xml.isAssignmentOperator(nextOp) || Xml.isAugAssignmentOperator(nextOp)
            || nextOp?.text === "--" || nextOp?.text === "++")) {
        return true;
    }

    return false;
}