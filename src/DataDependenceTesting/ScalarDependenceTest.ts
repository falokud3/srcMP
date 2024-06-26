import * as Xml from '../Facades/Xml/Xml.js'

export function collectScalarDependencies(loop: Xml.ForLoop) : Set<Xml.Element> {
    const ret = new Set<Xml.Element>(); // TODO: REMOVE USE OF ALL SETS
    const def = loop.body.defSymbols;
    // const use = loop.useSymbols;

    const loopIVs = loop.getInnerLoopNest().map((loop) => Array.from(loop.header.defSymbols))
        .flat();
    
    for (const symbol of def) {
        // TODO: Private or Reduction
        if (!isScalar(symbol)) continue;
        if (loopIVs.some((iv) => symbol.equals(iv) )) continue;

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