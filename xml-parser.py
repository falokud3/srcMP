#!/usr/bin/python3

import sys
import xml.etree.ElementTree as ET

ns = {'src': 'http://www.srcML.org/srcML/src', 'cpp': 'http://www.srcML.org/srcML/cpp'}


def gen_xml(py_src_path: str) -> str:
    """
    Generates an srcML xml file of the input source code if it is not already a xml file
    :param py_src_path: string path to input source code file
    :return:  path to the xml file
    """

    # TODO: IMPLEMENT THIS
    return py_src_path


def parallelize_for(root: ET.Element) -> None:
    """
    Parses the root element for loops that can potentially be parallelized
    :param root: ET.element containing code tags within
    """
    # TODO: EXTRACT NAMESPACES FROM UNIT TAG
    loop_variable = root.find("./src:control/src:init/src:decl/src:name", ns).text
    loop_body = root.find("./src:block/src:block_content", ns)

    is_parallelized: bool = True
    for expr in loop_body.iter('{http://www.srcML.org/srcML/src}expr'):
        # check if expr has name that matches loop variable
        names = expr.findall("./src:name", ns)
        has_loop_variable: bool = False
        for name in names:
            if name.text == loop_variable:
                has_loop_variable = True
        # check if expr has other elements
        if has_loop_variable:
            is_independent: bool = len(expr.findall(".//", ns)) == 1
            is_parallelized = is_independent and is_parallelized
    print(root, is_parallelized,)


def parse(xml_src_path: str) -> None:
    """
    Parses the input xml file for opportunities for parallelization
    :param xml_src_path: the path to the xml file
    """
    # Open the specified source file.
    src_file = open(xml_src_path, "r")
    if not src_file.readable():
        raise IOError("Unable to read from {}".format(xml_src_path))
    tree: ET.ElementTree = ET.parse(src_file)
    root: ET.Element = tree.getroot()

    for elem in root.iter():
        # TODO: EXTRACT NAMESPACES FROM UNIT TAG
        element = elem.find('src:for', ns)
        if element is not None:
            parallelize_for(element)


def main():
    """The main function that starts the process of XML generation
    by calling suitable helper methods.

    It uses command line arguments. Each command-line argument
    is assumed to be a python program to be converted to
    srcML.
    """
    # If we don't have command-line arguments, then report an error
    if len(sys.argv) < 2:
        print("Specify python source file as command-line argument.")
    else:
        # Process each source file specified as command-line argument
        for py_src_path in sys.argv[1:]:
            xml_src_path: str = gen_xml(py_src_path)
            parse(xml_src_path)


# The top-level script.
if __name__ == "__main__":
    main()
