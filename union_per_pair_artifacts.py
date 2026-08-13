import os

import onnx


def _tensor_producers(graph: onnx.GraphProto) -> dict[str, str]:
    return {output: n.name for n in graph.node for output in n.output}


def _rename_tensor(graph: onnx.GraphProto, old_name: str, new_name: str) -> None:
    for n in graph.node:
        n.input[:] = [new_name if x == old_name else x for x in n.input]
        n.output[:] = [new_name if x == old_name else x for x in n.output]
    for coll in (graph.initializer, graph.value_info, graph.input, graph.output):
        for entry in coll:
            if entry.name == old_name:
                entry.name = new_name


def _resolve_pivot_clashes(base_graph: onnx.GraphProto, graph: onnx.GraphProto, tag: str) -> None:
    """Rename tensors that share a name across graphs but come from different producers.

    This happens for the pivot tensor of adjacent pairs (e.g. `add_4` is the downstream
    of pair0 and the upstream of pair1): ORT training derives `{tensor}_grad` purely from
    the tensor name, so pair0's loss-seed grad for add_4 and pair1's computed weight-grad
    for add_4 collide on the name `add_4_grad` despite being unrelated tensors.
    """
    base_producers = _tensor_producers(base_graph)
    producers = _tensor_producers(graph)
    for name, producer in producers.items():
        base_producer = base_producers.get(name)
        if base_producer is not None and base_producer != producer:
            _rename_tensor(graph, name, f"{tag}_{name}")


def _align_pivot_casts(base_graph: onnx.GraphProto, graph: onnx.GraphProto) -> None:
    """Rename `graph`'s Cast-of-pivot-tensor nodes to match `base_graph`'s equivalents.

    ORT's artifact generator names auto-inserted nodes off a build-order counter that's
    perturbed by each pair's distinct loss attachment point, so independently-generated
    per-pair graphs can end up with differently-named (but functionally identical) Cast
    nodes wrapping the same shared tensor -- e.g. `Cast(add_8)` comes out as
    `onnx::Cast::2 -> onnx::cast.output::1` in one graph and `node__to_copy_7 -> _to_copy_7`
    in another. Both consume the same shared-forward-pass tensor and are otherwise
    identical, so they're the same node wearing a different name; align them before the
    by-name merge runs, or the merge will (correctly, but unhelpfully) reject them as
    conflicting.
    """
    base_casts: dict[str, onnx.NodeProto] = {}
    for n in base_graph.node:
        if n.op_type == "Cast" and len(n.input) == 1:
            base_casts.setdefault(n.input[0], n)

    for n in graph.node:
        if n.op_type != "Cast" or len(n.input) != 1:
            continue
        base_n = base_casts.get(n.input[0])
        if base_n is None or base_n.name == n.name:
            continue
        if base_n.attribute != n.attribute:
            continue
        _rename_tensor(graph, n.output[0], base_n.output[0])
        n.name = base_n.name


def _merge_graphs(models: list[onnx.ModelProto]) -> onnx.ModelProto:
    """Union several per-pair training graphs into one graph.

    Each per-pair graph is the same GPT-2 forward pass plus its own MaskedSumLoss ->
    backward tail for a single (upstream, downstream) pair, all keyed off the same shared
    `input_ids` / `mask` / `lazy_reset_grad` inputs. Nodes, initializers and value_info
    that are identical across graphs (i.e. the shared forward pass) are deduped by name;
    each graph's unique loss + backward nodes and its loss-seed grad initializer are kept,
    so the merged graph has one output per pair, each seeded from the same `mask` input.
    """
    base = models[0]
    merged_graph = onnx.GraphProto()
    merged_graph.CopyFrom(base.graph)

    node_by_name = {n.name: n for n in merged_graph.node}
    init_by_name = {i.name: i for i in merged_graph.initializer}
    vi_by_name = {v.name: v for v in merged_graph.value_info}
    output_names = {o.name for o in merged_graph.output}

    for pair_idx, model in enumerate(models[1:], start=1):
        g = model.graph
        _align_pivot_casts(merged_graph, g)
        _resolve_pivot_clashes(merged_graph, g, tag=f"pair{pair_idx}")

        for i in g.input:
            if i.name not in {inp.name for inp in merged_graph.input}:
                merged_graph.input.append(i)

        for n in g.node:
            existing = node_by_name.get(n.name)
            if existing is not None:
                if existing.SerializeToString() != n.SerializeToString():
                    raise ValueError(f"node {n.name!r} differs between graphs; cannot union")
                continue
            merged_graph.node.append(n)
            node_by_name[n.name] = n

        for i in g.initializer:
            existing = init_by_name.get(i.name)
            if existing is not None:
                if existing.raw_data != i.raw_data or list(existing.dims) != list(i.dims):
                    raise ValueError(f"initializer {i.name!r} differs between graphs; cannot union")
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

    # Not "union_" + "_".join(graph names): every per-pair graph is itself named
    # "main_graph" (onnxblock's artifact-generator default), so that would repeat
    # "main_graph" once per pair (e.g. "union_main_graph_main_graph_main_graph...") --
    # cosmetic, but it's the garbled name ORT's own Memcpy-node warning prints.
    merged_graph.name = f"union_{len(models)}_pairs"

    merged_model = onnx.ModelProto()
    merged_model.CopyFrom(base)
    merged_model.graph.CopyFrom(merged_graph)
    return merged_model


def build_union_artifacts(training_model_paths: list[str], out_path: str) -> None:
    # os.makedirs(out_dir, exist_ok=True)

    models = [onnx.load(path) for path in training_model_paths]

    mask_inputs = set()
    for model in models:
        for i in model.graph.input:
            if i.name == "mask":
                mask_inputs.add(i.SerializeToString())
    assert len(mask_inputs) == 1, "per-pair graphs must share an identical `mask` input to union their loss seeds"

    merged_model = _merge_graphs(models)
    onnx.checker.check_model(merged_model)

    onnx.save(merged_model, out_path, save_as_external_data=True, format="onnxtxt")

    print(f"union inputs: {[i.name for i in merged_model.graph.input]}")
    print(f"union outputs: {[o.name for o in merged_model.graph.output]}")
    print(f"union node count: {len(merged_model.graph.node)}")
    print(f"union initializer count: {len(merged_model.graph.initializer)}")
    print(f"saved union training model -> {out_path}")
