import * as xml from 'libxmljs2';

// namespace
const ns = {'xmlns': 'http://www.srcML.org/srcML/src'}

function contains(source: xml.Element, key: string, namespace = ns) : boolean {
    return source.find(key, namespace).length != 0;
}

function containsName(source: xml.Element, name: string, namespace = ns) : boolean {
    return source.find(`.//xmlns:name[text()='${name}']`, namespace).length != 0;
}

export {contains, containsName, ns}