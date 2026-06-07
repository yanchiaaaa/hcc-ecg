import argparse
import os
import xml.etree.ElementTree as ET

import torch


def parse_icd_tabular_xml(xml_file):
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"File not found: {xml_file}")

    tree = ET.parse(xml_file)
    root = tree.getroot()

    edges = []
    all_codes = set()

    def process_diag(parent_code, diag_element):
        name_tag = diag_element.find("name")
        if name_tag is None or name_tag.text is None:
            return

        current_code = name_tag.text.strip().replace(".", "")
        all_codes.add(current_code)
        edges.append((parent_code, current_code))

        for sub_diag in diag_element.findall("diag"):
            process_diag(current_code, sub_diag)

    for chapter in root.findall("chapter"):
        elem_name = chapter.find("name")
        if elem_name is None or elem_name.text is None:
            continue

        chapter_node = "Chapter " + elem_name.text.strip()
        all_codes.add(chapter_node)

        for section in chapter.findall("section"):
            raw_section_code = section.get("id")
            section_code = raw_section_code.replace(".", "") if raw_section_code else None
            if section_code is None:
                continue

            all_codes.add(section_code)
            edges.append((chapter_node, section_code))

            for diag in section.findall("diag"):
                process_diag(section_code, diag)

    return sorted(all_codes), edges


def build_graph(xml_file, output_path):
    nodes, raw_edges = parse_icd_tabular_xml(xml_file)
    code_to_id = {code: idx for idx, code in enumerate(nodes)}
    id_to_code = {idx: code for code, idx in code_to_id.items()}

    edge_index_list = [
        [code_to_id[src], code_to_id[dst]]
        for src, dst in raw_edges
        if src in code_to_id and dst in code_to_id
    ]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    torch.save(
        {
            "code_to_id": code_to_id,
            "id_to_code": id_to_code,
            "edge_index": edge_index,
        },
        output_path,
    )

    print(f"Saved ICD graph to {output_path}")
    print(f"Nodes: {len(code_to_id)}")
    print(f"Edges: {edge_index.size(1)}")


def main():
    parser = argparse.ArgumentParser(description="Build an ICD-10-CM hierarchy graph from the tabular XML file.")
    parser.add_argument(
        "--xml_file",
        default="icd_graph/icd10cm/icd10cm_tabular_2024.xml",
        help="Path to the ICD-10-CM tabular XML file.",
    )
    parser.add_argument(
        "--output_path",
        default="icd_graph/icd_graph_data.pt",
        help="Output .pt graph file.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    build_graph(args.xml_file, args.output_path)


if __name__ == "__main__":
    main()
