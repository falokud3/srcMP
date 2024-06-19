import * as Xml from '../Xml/Xml.js'

export function collectScalarDependencies(loop: Xml.ForLoop) : Set<Xml.Element> {
    const ret = new Set<Xml.Element>();
    const def = loop.defSymbols;
    const use = loop.useSymbols;
    
    for (const symbol of def) {
        // TODO: Private or Reduction
        if (!isScalar(symbol)) continue;

        // TODO: pointer
        // TODO: Objects

        ret.add(symbol);
    }
    return ret;
}

function isScalar(symbol: Xml.Element) : boolean {
    // ! duplicate names a[][] x.a
    return !(symbol.parentElement?.name === 'name' 
        && symbol.parentElement?.contains('xmlns:index'));
}