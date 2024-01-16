import * as xml from 'libxmljs2';

// namespace
const ns = {'xmlns': 'http://www.srcML.org/srcML/src'}

function contains(source: xml.Element, key: string, namespace) : boolean {
    return source.find(key, namespace).length != 0;
}

export {contains, ns}