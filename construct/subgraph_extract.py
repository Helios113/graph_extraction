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
) -> Path:
   
    merged_path = "-".join([".".join(path.stem.split(".")[0:2]) for path in paths])
    hash = hashlib.md5(string=merged_path.encode("utf-8")).hexdigest()[:8]
    final_name = "-".join([".".join(path.stem.split(".")[0:2]) for path in [paths[0], paths[-1]]]) + "_" + hash

    merged_graph = onnx.GraphProto()
    node_by_name: dict[str, onnx.NodeProto] = {}
    init_by_name: dict[str, onnx.TensorProto] = {}
    vi_by_name: dict[str, onnx.ValueInfoProto] = {}
    output_names: set[str] = set()
    input_names_seen: set[str] = set()

    # Maps a claimed tensor name -> True if it is protected (a declared
    # output of the subgraph that produced it), False if it is just an
    # internal node output that can be renamed away on a later collision.
    claimed_outputs: dict[str, bool] = {}

    base = onnx.load(paths[0])

    for path_idx, path in enumerate(paths):
        g = base.graph if path_idx == 0 else onnx.load(path).graph
        g_output_names = {o.name for o in g.output}

        for i in g.input:
            if i.name not in input_names_seen:
                merged_graph.input.append(i)
                input_names_seen.add(i.name)

        # Build a rename map for this subgraph: any node output that
        # collides with an already-claimed name gets suffixed, UNLESS this
        # node's output is this subgraph's own declared output and the
        # existing claim on that name is not itself protected (in which
        # case the earlier, unprotected claimant must yield instead -- see
        # below).
        rename_map: dict[str, str] = {}
        retroactive_renames: dict[str, str] = {}
        for n in g.node:
            if n.name in node_by_name:
                continue
            for o in n.output:
                is_real_output = o in g_output_names
                prior = claimed_outputs.get(o)
                if prior is None:
                    continue
                if is_real_output:
                    if prior is True:
                        raise ValueError(
                            f"output {o!r} is produced as a real output by "
                            f"multiple subgraphs (conflict introduced by "
                            f"{path.name!r}); cannot union without manual "
                            f"disambiguation",
                        )
                    # An earlier, unprotected (internal-only) node already
                    # claimed this name. This subgraph's claim is the real
                    # one -- rename the earlier claimant instead of this
                    # output.
                    retroactive_renames[o] = f"{o}__{path.stem}__internal"
                elif o not in rename_map:
                    rename_map[o] = f"{o}__{path.stem}"

        if retroactive_renames:
            _apply_retroactive_renames(merged_graph, node_by_name, vi_by_name, retroactive_renames)
            for old, new in retroactive_renames.items():
                claimed_outputs[new] = claimed_outputs.pop(old)

        def _rename(name: str, _rename_map=rename_map) -> str:
            return _rename_map.get(name, name)

        for n in g.node:
            existing = node_by_name.get(n.name)
            if existing is not None:
                if existing.SerializeToString() != n.SerializeToString():
                    raise ValueError(
                        f"node {n.name!r} differs between graphs; cannot union",
                    )
                continue
            new_n = onnx.NodeProto()
            new_n.CopyFrom(n)
            new_n.output[:] = [_rename(o) for o in n.output]
            new_n.input[:] = [_rename(i) for i in n.input]
            merged_graph.node.append(new_n)
            node_by_name[new_n.name] = new_n
            for o in new_n.output:
                claimed_outputs[o] = o in {_rename(x) for x in g_output_names}

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
            new_name = _rename(v.name)
            if new_name not in vi_by_name:
                new_v = onnx.ValueInfoProto()
                new_v.CopyFrom(v)
                new_v.name = new_name
                merged_graph.value_info.append(new_v)
                vi_by_name[new_name] = new_v

        for o in g.output:
            new_name = _rename(o.name)
            if new_name not in output_names:
                new_o = onnx.ValueInfoProto()
                new_o.CopyFrom(o)
                new_o.name = new_name
                merged_graph.output.append(new_o)
                output_names.add(new_name)

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


def _apply_retroactive_renames(
    merged_graph: onnx.GraphProto,
    node_by_name: dict[str, onnx.NodeProto],
    vi_by_name: dict[str, onnx.ValueInfoProto],
    renames: dict[str, str],
) -> None:
    """Rename tensor names already placed into merged_graph, rewriting every
    node's inputs/outputs and any matching value_info entries in place.
    Does not touch merged_graph.output -- retroactive renames only ever
    target names that were never promoted to a graph output (that's the
    precondition for triggering a retroactive rename in the first place).
    """
    for n in merged_graph.node:
        for idx, name in enumerate(n.input):
            if name in renames:
                n.input[idx] = renames[name]
        for idx, name in enumerate(n.output):
            if name in renames:
                n.output[idx] = renames[name]
    for old, new in renames.items():
        if old in vi_by_name:
            vi = vi_by_name.pop(old)
            vi.name = new
            vi_by_name[new] = vi
