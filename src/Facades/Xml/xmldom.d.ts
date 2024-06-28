
// declare module xmldom {


// export class DocumentType {
//     nodeType: number;
// }
// /**
//  * DOM Level 2
//  * Object DOMException
//  * @see http://www.w3.org/TR/2000/REC-DOM-Level-2-Core-20001113/ecma-script-binding.html
//  * @see http://www.w3.org/TR/REC-DOM-Level-1/ecma-script-language-binding.html
//  */
// export function DOMException(code: any, message: any): Error;
// export class DOMException {
//     /**
//      * DOM Level 2
//      * Object DOMException
//      * @see http://www.w3.org/TR/2000/REC-DOM-Level-2-Core-20001113/ecma-script-binding.html
//      * @see http://www.w3.org/TR/REC-DOM-Level-1/ecma-script-language-binding.html
//      */
//     constructor(code: any, message: any);
//     message: any;
// }
// /**
//  * The DOMImplementation interface represents an object providing methods
//  * which are not dependent on any particular document.
//  * Such an object is returned by the `Document.implementation` property.
//  *
//  * __The individual methods describe the differences compared to the specs.__
//  *
//  * @constructor
//  *
//  * @see https://developer.mozilla.org/en-US/docs/Web/API/DOMImplementation MDN
//  * @see https://www.w3.org/TR/REC-DOM-Level-1/level-one-core.html#ID-102161490 DOM Level 1 Core (Initial)
//  * @see https://www.w3.org/TR/DOM-Level-2-Core/core.html#ID-102161490 DOM Level 2 Core
//  * @see https://www.w3.org/TR/DOM-Level-3-Core/core.html#ID-102161490 DOM Level 3 Core
//  * @see https://dom.spec.whatwg.org/#domimplementation DOM Living Standard
//  */
// export class DOMImplementation {
//     hasFeature: (feature: string, version?: string) => boolean;
//     createDocument: (namespaceURI: string | null, qualifiedName: string, doctype: any) => Document;
//     createDocumentType: (qualifiedName: string, publicId?: string, systemId?: string) => DocumentType;
// }


// export class Element extends Node {
//     _nsMap: {};
//     nodeType: number;
//     readonly tagName: string;
//     hasAttribute: (name: any) => boolean;
//     getAttribute: (name: any) => any;
//     getAttributeNode: (name: any) => any;
//     setAttribute: (name: any, value: any) => void;
//     removeAttribute: (name: any) => void;
//     appendChild: (newChild: any) => any;
//     setAttributeNode: (newAttr: any) => any;
//     setAttributeNodeNS: (newAttr: any) => any;
//     removeAttributeNode: (oldAttr: any) => any;
//     removeAttributeNS: (namespaceURI: any, localName: any) => void;
//     hasAttributeNS: (namespaceURI: any, localName: any) => boolean;
//     getAttributeNS: (namespaceURI: any, localName: any) => any;
//     setAttributeNS: (namespaceURI: any, qualifiedName: any, value: any) => void;
//     getAttributeNodeNS: (namespaceURI: any, localName: any) => any;
//     getElementsByTagName: (tagName: any) => LiveNodeList;
//     getElementsByTagNameNS: (namespaceURI: any, localName: any) => LiveNodeList;
// }
// /**
//  * @see http://www.w3.org/TR/2000/REC-DOM-Level-2-Core-20001113/core.html#ID-1950641247
//  */
// export class Node {
//     toString: typeof nodeSerializeToString;
//     set textContent(value: string);
//     get textContent(): string;
//     readonly firstChild: Node | null;
//     readonly lastChild: Node | null;
//     readonly previousSibling: Node | null;
//     readonly nextSibling: Node | null;
//     readonly attributes: NamedNodeMap | null;
//     readonly parentNode: Node | null;
//     readonly childNodes: NodeList;
//     readonly ownerDocument: null;
//     nodeValue: string | null;
//     namespaceURI: string;
//     prefix: string | null;
//     readonly localName: string;
//     insertBefore: (newChild: Node, refChild: Node) => Node;
//     replaceChild: (newChild: Node, oldChild: Node) => Node;
//     removeChild: (oldChild: Node) => Node;
//     appendChild: (newChild: Node) => Node;
//     hasChildNodes: () => boolean;
//     cloneNode: (deep: boolean) => Node;
//     normalize: () => void;
//     isSupported: (feature: string, version: string) => boolean;
//     hasAttributes: () => boolean;
//     lookupPrefix: (namespaceURI: string | null) => string | null;
//     lookupNamespaceURI: (prefix: string | null) => string | null;
//     isDefaultNamespace: (namespaceURI: any) => boolean;
// }
// /**
//  * @see http://www.w3.org/TR/2000/REC-DOM-Level-2-Core-20001113/core.html#ID-536297177
//  * The NodeList interface provides the abstraction of an ordered collection of nodes, without defining or constraining how this collection is implemented. NodeList objects in the DOM are live.
//  * The items in the NodeList are accessible via an integral index, starting from 0.
//  */
// export class NodeList {
//     readonly length: number;
//     item: (index: any) => Node;
//     toString: (isHTML: any, nodeFilter: any) => string;
//     filter: (predicate: (arg0: Node) => boolean) => Node[];
//     indexOf: (item: Node) => number;
// }

// export class XMLSerializer {
//     serializeToString(node: any, isHtml: any, nodeFilter: any): any;
// }
// declare class Document {
//     ownerDocument: this;
//     getElementsByTagName: (tagName: any) => LiveNodeList;
//     getElementsByTagNameNS: (namespaceURI: any, localName: any) => LiveNodeList;
//     nodeName: string;
//     nodeType: number;
//     readonly doctype: DocumentType;
//     documentElement: null;
//     _inc: number;
//     insertBefore: (newChild: Node, refChild: Node) => Node;
//     removeChild: (oldChild: Node) => Node;
//     replaceChild: (newChild: Node, refChild: Node) => Node;
//     importNode: (importedNode: any, deep: any) => any;
//     getElementById: (id: any) => any;
//     getElementsByClassName: (classNames: string) => LiveNodeList;
//     createElement: (tagName: any) => Element;
//     createDocumentFragment: () => DocumentFragment;
//     createTextNode: (data: any) => Text;
//     createComment: (data: any) => Comment;
//     createCDATASection: (data: any) => CDATASection;
//     createProcessingInstruction: (target: any, data: any) => ProcessingInstruction;
//     createAttribute: (name: any) => Attr;
//     createEntityReference: (name: any) => EntityReference;
//     createElementNS: (namespaceURI: any, qualifiedName: any) => Element;
//     createAttributeNS: (namespaceURI: any, qualifiedName: any) => Attr;
// }
// declare class LiveNodeList {
//     constructor(node: any, refresh: any);
//     _node: any;
//     _refresh: any;
//     item(i: any): any;
//     get length(): any;
// }
// declare function nodeSerializeToString(isHtml: any, nodeFilter: any): string;

// declare class DocumentFragment {
//     nodeName: string;
//     nodeType: number;
// }

// declare class Text {
//     nodeName: string;
//     nodeType: number;
//     splitText: Function;
// }

// declare class Comment {
//     nodeName: string;
//     nodeType: number;
// }

// declare class CDATASection {
//     nodeName: string;
//     nodeType: number;
// }

// declare class ProcessingInstruction {
//     nodeType: number;
// }

// declare class Attr {
//     nodeType: number;
// }

// declare class EntityReference {
//     nodeType: number;
// }
