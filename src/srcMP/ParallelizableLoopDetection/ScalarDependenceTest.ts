import * as Xml from '../../common/Xml/Xml.js';

export function collectScalarDependencies(loop: Xml.ForLoop) : Set<Xml.Element> {
    const ret = new Set<Xml.Element>(); 
    const def = loop.body.defSymbols;

    const loopIVs = loop.getInnerLoopNest().map((loop) => Array.from(loop.header.defSymbols))
        .flat();
    
    for (const symbol of def) {
        if (!isScalar(symbol)) continue;
        if (loopIVs.some((iv) => symbol.equals(iv) )) continue;

        ret.add(symbol);
    }
    return ret;
}

function isScalar(symbol: Xml.Element) : boolean {
    // ! duplicate names a[][] x.a may cause issues
    return !(symbol.parentElement?.name === 'name' 
        && symbol.parentElement?.contains('xmlns:index'));
}