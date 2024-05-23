import * as libxmljs2 from 'libxmljs2';
import * as Xml from './Xml.js'

/**
 * Represents XML Element, like `<person/>`, not XML Comments or Attributes
 */
export class Element {
    private libxml: libxmljs2.Element; 

    /**
     * Offset needed to get the true line number of an xml element. This is caused by
     * the additional unit element added to srcML output
     * * This is specific to this program's use of srcML, where only file is converted at a time
     */
    static LINE_NUMBER_OFFSET = 1;

    public constructor(libxml: libxmljs2.Element) {
        this.libxml = libxml;
    }

    /**
     * Returns the internal xml object used by the library
     * * Use of this is discouraged, due to the increase in coupling
     */
    public get libraryXmlObject(): libxmljs2.Element {
        return this.libxml;
    }

    public get line(): number {
        return this.libxml.line() - Element.LINE_NUMBER_OFFSET;
    }

    /**
     * Returns the text within an xml tag including the text of all child nodes
     */
    public get text() : string {
        return this.libxml.text();
    }

    /**
     * Returns the text within the tag, like parent from `<parent/>`
     */
    public get name() : string {
        return this.libxml.name();
    }

    /**
     * Returns an array of all the child Elements
     */
    public get children() : Element[] {
        const children = this.libxml.childNodes();
        const ret: Element[] = [];
        for (const childNode of children) {
            if (!(childNode instanceof libxmljs2.Element)) {
                continue;
            }
            ret.push(new Element(childNode));
        }
        return ret;
    }

    /**
     * Returns the parent Element of the caller or null if the Element has no
     * parent Element
     */
    public get parent() : Element {
        const ret = this.libxml.parent()
        return (ret instanceof libxmljs2.Document) ? null : new Element(ret);
    }

    /**
     * Returns the next sibling element or null
     */
    public get nextElement() : Element {
        return new Element(this.libxml.nextElement());
    }

    public child(index: number) : Element {
        const children = this.children;
        return children.length > 0 ? children[index] : null;
    }

    /**
     * Gets an XML Element based on xpath
     * @param xpath 
     * @param namespace 
     * @returns the first Element found or null if none are found
     */
    public get(xpath: string, namespace: libxmljs2.StringMap = Xml.ns) : Element {
        const queryResult = this.find(xpath, namespace);
        return queryResult.length > 0 ? queryResult[0] : null;
    }

    /**
     * Gets XML Elements based on xpath
     * @param xpath 
     * @param namespace 
     * @returns an array of Element objects (empty if none are found)
     */
    public find(xpath: string, namespace: libxmljs2.StringMap = Xml.ns) : Element[] {
        const queryResult = this.libxml.find(xpath, namespace);
        const ret: Element[] = [];
        for (const node of queryResult) {
            if (!(node instanceof libxmljs2.Element)) {
                continue;
            }

            if (node.name() === "for") {
                ret.push(new Xml.Loop(node));
            } else {
                ret.push(new Element(node));
            } 
        }
        return ret;
    }

    /**
     * Returns whether an element contains the xpath passed in 
     * @param xpath 
     * @param namespace 
     * @returns true if contains, false otherwise
     */
    public contains(xpath: string, namespace: libxmljs2.StringMap = Xml.ns) : boolean {
        return this.get(xpath, namespace) != null
    }

    /**
     * Checks if the XML element contains a name node with specified text
     * @param xpath
     * @param namespace 
     * @returns 
     */
    public containsName(xpath: string, namespace: libxmljs2.StringMap = Xml.ns) : boolean {
        return this.libxml.find(`.//xmlns:name[text()='${xpath}']`, namespace).length != 0;
    }

    public getAttribute(name: string) : string {
        return this.libxml.attr(name).value()
    }

    /**
     * Setting an attribute on the Xml Element, like count from `<line count=2 />`
     * @param name the name of the attribute to set
     * @param value the value to set the attribute
     */
    public setAttribute(name: string, value: string) : void {
        this.libxml.attr(name, value);
    }

    /**
     * Replace the Xml of an element with the Xml of another
     * @param newElement 
     */
    public replace(newElement: Element) : void {
        this.libxml.replace(newElement.libxml);
    }

}