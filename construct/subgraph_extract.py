import hashlib
import os
from pathlib import Path

import onnx
from onnx.utils import Extractor


def extract_subgraph(
    input_path: Path,
    output_path: Path,
    input_names: str,
    output_names: str,
    check_model: bool = True,
    infer_shapes: bool = True,
) -> None:
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid input model path: {input_path}")
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not input_names:
        raise ValueError("Input tensor names shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    if check_model:
        onnx.checker.check_model(input_path)

    if infer_shapes and os.path.getsize(input_path) > onnx.checker.MAXIMUM_PROTOBUF:
        onnx.shape_inference.infer_shapes_path(input_path, output_path)
        model = onnx.load(output_path)
    elif infer_shapes:
        model = onnx.load(input_path, load_external_data=False)
        model = onnx.shape_inference.infer_shapes(model)
        base_dir = os.path.dirname(input_path)
        onnx.load_external_data_for_model(model, base_dir)
    else:
        model = onnx.load(input_path)

    e = Extractor(model)
    extracted = e.extract_model([input_names], [output_names])

    location = os.path.basename(output_path) + ".data"
    onnx.save(extracted, output_path, save_as_external_data=True, location=location)

    onnx.checker.check_model(output_path)


def generate_union_of_subgraphs(
    paths: list[Path],
):
    merged_path = "-".join([".".join(path.stem.split(".")[0:2]) for path in paths])
    hash = hashlib.md5(string=merged_path.encode("utf-8")).hexdigest()[:8]
    final_name = "-".join([".".join(path.stem.split(".")[0:2]) for path in [paths[0], paths[-1]]]) + "_" + hash
    base = onnx.load(paths[0])
    merged_graph = onnx.GraphProto()
    merged_graph.CopyFrom(base.graph)

    node_by_name = {n.name: n for n in merged_graph.node}
    init_by_name = {i.name: i for i in merged_graph.initializer}
    vi_by_name = {v.name: v for v in merged_graph.value_info}
    output_names = {o.name for o in merged_graph.output}

    for pair_idx, path in enumerate(paths[1:], start=1):
        g = onnx.load(path).graph

        for i in g.input:
            if i.name not in {inp.name for inp in merged_graph.input}:
                merged_graph.input.append(i)

        for n in g.node:
            existing = node_by_name.get(n.name)
            if existing is not None:
                if existing.SerializeToString() != n.SerializeToString():
                    raise ValueError(
                        f"node {n.name!r} differs between graphs; cannot union",
                    )
                continue
            merged_graph.node.append(n)
            node_by_name[n.name] = n

        for i in g.initializer:
            existing = init_by_name.get(i.name)
            if existing is not None:
                if existing.raw_data != i.raw_data or list(existing.dims) != list(
                    i.dims,
                ):
                    raise ValueError(
                        f"initializer {i.name!r} differs between graphs; cannot union",
                    )
                continue
            merged_graph.initializer.append(i)
            init_by_name[i.name] = i

        for v in g.value_info:
            if v.name not in vi_by_name:
                merged_graph.value_info.append(v)
                vi_by_name[v.name] = v

        for o in g.output:
            if o.name not in output_names:
                merged_graph.output.append(o)
                output_names.add(o.name)

    merged_model = onnx.ModelProto()
    merged_model.CopyFrom(base)
    merged_model.graph.CopyFrom(merged_graph)

    save_path = paths[0].with_name(final_name + ".union.onnx")
    location = os.path.basename(save_path) + ".data"
    onnx.save(
        merged_model,
        save_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
    )
    return save_path
