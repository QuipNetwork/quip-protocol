#!/usr/bin/env python3
"""AST-based auto-instrumenter for CUDA kernel profiling.

Parses a clean .cu file with tree-sitter-cuda, identifies loops,
__syncthreads(), __ldg() sequences, and device function calls,
then generates a fully-instrumented profiling variant as a temp
file plus a JSON manifest mapping region IDs to metadata.

Production .cu files stay clean — zero profiling code. The
profiled variant is compiled only when profile=True.
"""
import json
import tempfile
from pathlib import Path

import tree_sitter_cuda as tscuda
from tree_sitter import Language, Parser

CUDA_LANGUAGE = Language(tscuda.language())

# Device functions to instrument as "call" regions
INSTRUMENTED_CALLS = frozenset({
    "compute_effective_field_shared",
    "gibbs_sample",
    "metropolis_update",
    "get_flip_energy_unpacked",
    "xorshift32",
    "xoshiro128ss",
})


def auto_instrument(
    cu_path: str,
    kernel_name: str,
    mode: str = "per_thread",
) -> tuple[str, dict]:
    """Auto-instrument a CUDA kernel for profiling.

    Also descends into device functions listed in
    INSTRUMENTED_CALLS — adds prof array as a parameter,
    instruments their bodies, and passes the array at
    each call site.

    Args:
        cu_path: Path to clean .cu file.
        kernel_name: e.g. "cuda_sa_self_feeding".
        mode: "per_thread" (SA) or "thread_zero" (Gibbs).

    Returns:
        (profiled_cu_path, manifest_dict)
        profiled_cu_path is a temp file path.
    """
    source_bytes = Path(cu_path).read_bytes()
    source_text = source_bytes.decode("utf-8")

    parser = Parser(CUDA_LANGUAGE)
    tree = parser.parse(source_bytes)

    kernel_node = _find_kernel_function(
        tree.root_node, kernel_name,
    )
    assert kernel_node is not None, (
        f"Kernel '{kernel_name}' not found in {cu_path}"
    )

    regions = _collect_regions(kernel_node, kernel_name)

    # Collect device function definitions and their
    # internal sub-regions.  Sub-regions become children
    # of the corresponding "call" region in the manifest.
    dev_funcs = _find_device_functions(
        tree.root_node, INSTRUMENTED_CALLS,
    )
    dev_sub_regions = {}  # func_name -> list[region]
    for fname, fnode in dev_funcs.items():
        subs = _collect_regions(fnode, fname)
        if subs:
            dev_sub_regions[fname] = subs

    # Merge sub-regions into the main region list.
    # Each sub-region becomes a child of the first "call"
    # region that references the function.
    for fname, subs in dev_sub_regions.items():
        # Find call regions that reference this function
        call_regions = [
            r for r in regions
            if r["label"].endswith(f":call_{fname}")
        ]
        if not call_regions:
            continue
        parent_call = call_regions[0]
        for sub in subs:
            sub["id"] = len(regions)
            sub["_parent_call_func"] = fname
            regions.append(sub)

    _assign_parents_and_depth(regions)

    # For sub-regions whose parent wasn't resolved via
    # byte ranges (they live in a different function),
    # manually set parent_id to the call region.
    for r in regions:
        fname = r.get("_parent_call_func")
        if fname is None:
            continue
        if r["parent_id"] is not None:
            # Already resolved (nested inside another
            # sub-region of the same function)
            # Check if parent is also a sub-region of
            # the same function — that's fine
            parent = next(
                (x for x in regions
                 if x["id"] == r["parent_id"]),
                None,
            )
            if (parent
                    and parent.get("_parent_call_func")
                    == fname):
                continue
        # Set parent to the first call region
        call_regions = [
            c for c in regions
            if c["label"].endswith(f":call_{fname}")
        ]
        if call_regions:
            r["parent_id"] = call_regions[0]["id"]

    # Recompute depths after parent fixup
    id_map = {r["id"]: r for r in regions}
    for region in regions:
        depth = 0
        cur = region
        while cur["parent_id"] is not None:
            depth += 1
            cur = id_map[cur["parent_id"]]
        region["depth"] = depth

    # Build manifest
    manifest_regions = []
    for r in regions:
        manifest_regions.append({
            "id": r["id"],
            "label": r["label"],
            "type": r["type"],
            "line_start": r["line_start"],
            "line_end": r["line_end"],
            "depth": r["depth"],
            "parent_id": r["parent_id"],
        })

    num_regions = len(regions)
    manifest = {
        "kernel": kernel_name,
        "source_file": cu_path,
        "num_regions": num_regions,
        "profiling_mode": mode,
        "regions": manifest_regions,
    }

    profiled_source = _generate_profiled_source(
        source_text, kernel_node, regions, mode,
        dev_funcs, dev_sub_regions,
    )

    # Write profiled source to temp file
    suffix = Path(cu_path).suffix
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False,
        prefix=f"{Path(cu_path).stem}_profiled_",
    )
    tmp.write(profiled_source)
    tmp.close()

    # Write manifest next to original .cu
    manifest_path = str(
        Path(cu_path).parent
        / f"{Path(cu_path).stem}_profile_manifest.json"
    )
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return tmp.name, manifest


# ── AST helpers ────────────────────────────────────────


def _walk(node):
    """Yield all nodes in pre-order."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _find_kernel_function(root, kernel_name):
    """Find function_definition for kernel_name."""
    for node in _walk(root):
        if node.type != "function_definition":
            continue
        for child in node.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if (sub.type == "identifier"
                            and sub.text.decode()
                            == kernel_name):
                        return node
    return None


def _find_device_functions(root, names):
    """Find function_definition nodes for named functions.

    Returns dict mapping function_name -> node.
    """
    result = {}
    for node in _walk(root):
        if node.type != "function_definition":
            continue
        for child in node.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if (sub.type == "identifier"
                            and sub.text.decode() in names):
                        result[sub.text.decode()] = node
    return result


def _get_callee_name(call_node):
    """Get function name from a call_expression."""
    for child in call_node.children:
        if child.type == "identifier":
            return child.text.decode()
    return None


def _get_loop_label(node):
    """Extract short label for a for/while loop."""
    if node.type == "for_statement":
        for child in node.children:
            if child.type == "declaration":
                for sub in _walk(child):
                    if sub.type == "identifier":
                        return sub.text.decode()
            if child.type in (
                "assignment_expression",
                "expression_statement",
            ):
                for sub in _walk(child):
                    if sub.type == "identifier":
                        return sub.text.decode()
        return "for"
    if node.type == "while_statement":
        for child in node.children:
            if child.type == "parenthesized_expression":
                cond = child.text.decode()[:20]
                return cond.strip("()")
        return "while"
    return "loop"


def _find_enclosing_statement(node):
    """Walk up to find enclosing expression_statement
    or declaration."""
    current = node.parent
    while current:
        if current.type in (
            "expression_statement", "declaration",
        ):
            return current
        current = current.parent
    return None


# ── Region collection ──────────────────────────────────


def _collect_regions(kernel_node, kernel_name):
    """Walk kernel AST and collect instrumentable regions."""
    body = None
    for child in kernel_node.children:
        if child.type == "compound_statement":
            body = child
            break
    assert body is not None, "Kernel has no body"

    regions = []
    seen_ranges = set()

    for node in _walk(body):
        line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # 1. Loops
        if node.type in ("for_statement", "while_statement"):
            label_info = _get_loop_label(node)
            regions.append(_make_region(
                node, kernel_name, line, end_line,
                f"{kernel_name}+L{line}:{label_info}",
                "loop",
            ))

        # 2. __syncthreads()
        elif node.type == "expression_statement":
            for sub in node.children:
                if (sub.type == "call_expression"
                        and _get_callee_name(sub)
                        == "__syncthreads"):
                    regions.append(_make_region(
                        node, kernel_name, line, end_line,
                        f"{kernel_name}+L{line}:sync",
                        "sync",
                    ))
                    break

        # 3. Instrumented function calls
        elif node.type == "call_expression":
            callee = _get_callee_name(node)
            if callee and callee in INSTRUMENTED_CALLS:
                stmt = _find_enclosing_statement(node)
                target = stmt if stmt else node
                t_line = target.start_point[0] + 1
                t_end = target.end_point[0] + 1
                key = (t_line, t_end, "call")
                if key not in seen_ranges:
                    seen_ranges.add(key)
                    regions.append(_make_region(
                        target, kernel_name,
                        t_line, t_end,
                        f"{kernel_name}+L{t_line}"
                        f":call_{callee}",
                        "call",
                    ))

    # Assign sequential IDs
    for i, r in enumerate(regions):
        r["id"] = i

    return regions


def _make_region(node, kernel_name, line, end_line,
                 label, rtype):
    """Create a region dict."""
    return {
        "id": 0,  # assigned later
        "label": label,
        "type": rtype,
        "line_start": line,
        "line_end": end_line,
        "depth": 0,
        "parent_id": None,
        "_start_byte": node.start_byte,
        "_end_byte": node.end_byte,
    }


def _assign_parents_and_depth(regions):
    """Set parent_id and depth from byte-range nesting."""
    for region in regions:
        best_parent = None
        best_size = float("inf")
        for candidate in regions:
            if candidate["id"] == region["id"]:
                continue
            c_s = candidate["_start_byte"]
            c_e = candidate["_end_byte"]
            r_s = region["_start_byte"]
            r_e = region["_end_byte"]
            if c_s <= r_s and c_e >= r_e and (
                c_s < r_s or c_e > r_e
            ):
                size = c_e - c_s
                if size < best_size:
                    best_size = size
                    best_parent = candidate
        if best_parent:
            region["parent_id"] = best_parent["id"]

    id_map = {r["id"]: r for r in regions}
    for region in regions:
        depth = 0
        cur = region
        while cur["parent_id"] is not None:
            depth += 1
            cur = id_map[cur["parent_id"]]
        region["depth"] = depth


# ── Code generation ────────────────────────────────────


def _generate_profiled_source(
    source_text, kernel_node, regions, mode,
    dev_funcs=None, dev_sub_regions=None,
):
    """Generate profiled .cu source via byte-offset edits.

    Applies edits in reverse byte order so offsets stay valid.
    Also instruments device function bodies and threads the
    _prof array through their call sites.
    """
    if dev_funcs is None:
        dev_funcs = {}
    if dev_sub_regions is None:
        dev_sub_regions = {}

    num_regions = len(regions)
    if num_regions == 0:
        return source_text

    max_depth = max(r["depth"] for r in regions) + 2
    is_t0 = mode == "thread_zero"

    # Separate kernel regions from device-function regions
    kernel_regions = [
        r for r in regions
        if "_parent_call_func" not in r
    ]

    # Find kernel body (compound_statement)
    body = None
    for child in kernel_node.children:
        if child.type == "compound_statement":
            body = child
            break
    assert body is not None

    # Find parameter list to add profile_output param
    param_list = None
    for child in kernel_node.children:
        if child.type == "function_declarator":
            for sub in child.children:
                if sub.type == "parameter_list":
                    param_list = sub
                    break
    assert param_list is not None

    # Collect all edits as (byte_offset, insert_text)
    # We apply in reverse order so earlier offsets stay valid
    edits = []

    # 1. Add profile_output parameter before closing ')'
    param_close = param_list.end_byte - 1  # before ')'
    edits.append((
        param_close,
        ",\n    long long* profile_output\n",
    ))

    # 2. Add declarations after opening '{' of body
    body_open = body.start_byte + 1  # after '{'
    decl = _build_declarations(num_regions, max_depth, is_t0)
    edits.append((body_open, "\n" + decl + "\n"))

    # 2b. Instrument device function bodies + signatures
    _instrument_device_functions(
        edits, dev_funcs, dev_sub_regions,
        regions, is_t0,
    )

    # 2c. At each call site in the kernel, pass _prof
    #     (and _is_prof_t0) as extra arguments
    _patch_call_sites(
        edits, body, dev_funcs, is_t0,
        dev_sub_regions,
    )

    # 3. Add PROF_T before each kernel region start
    #    (device-function regions handled separately)
    for r in kernel_regions:
        depth = r["depth"]
        tv = f"_pt{depth}"
        rid = r["id"]
        if is_t0:
            code = (
                f"    PROF_T0(_is_prof_t0, {tv}); "
                f" // r{rid}:{r['label']}\n"
            )
        else:
            code = (
                f"    PROF_T({tv}); "
                f" // r{rid}:{r['label']}\n"
            )
        edits.append((r["_start_byte"], code))

    # 4. Add PROF_ACCUM after each kernel region end
    for r in kernel_regions:
        depth = r["depth"]
        tv = f"_pt{depth}"
        rid = r["id"]
        if is_t0:
            code = (
                f"\n    PROF_ACCUM0(_is_prof_t0, _prof, "
                f"{rid}, {tv});  // end r{rid}"
            )
        else:
            code = (
                f"\n    PROF_ACCUM(_prof, {rid}, {tv}); "
                f" // end r{rid}"
            )
        edits.append((r["_end_byte"], code))

    # 5. Add PROF_ACCUM + writeback before every return
    # Persistent kernels exit via return inside loops,
    # so the PROF_ACCUM after the loop never executes.
    # We accumulate all ancestor regions before the return.
    wb = _build_writeback(num_regions, is_t0)
    for node in _walk(body):
        if node.type != "return_statement":
            continue

        # Find kernel regions that contain this return
        ret_start = node.start_byte
        ancestors = []
        for r in kernel_regions:
            if (r["_start_byte"] < ret_start
                    and r["_end_byte"] > ret_start):
                ancestors.append(r)
        # Sort by depth descending (innermost first)
        ancestors.sort(key=lambda r: -r["depth"])

        # Build PROF_ACCUM for each ancestor
        accum_lines = []
        for r in ancestors:
            tv = f"_pt{r['depth']}"
            rid = r["id"]
            if is_t0:
                accum_lines.append(
                    f"    PROF_ACCUM0(_is_prof_t0, _prof, "
                    f"{rid}, {tv});  // pre-return r{rid}"
                )
            else:
                accum_lines.append(
                    f"    PROF_ACCUM(_prof, {rid}, {tv}); "
                    f" // pre-return r{rid}"
                )
        pre_return = "\n".join(accum_lines)
        if pre_return:
            pre_return += "\n"
        pre_return += wb

        parent = node.parent
        if parent and parent.type == "compound_statement":
            edits.append((
                node.start_byte,
                pre_return + "\n",
            ))
        else:
            # Brace-less body (e.g. `if (...) return;`)
            edits.append((
                node.start_byte,
                "{\n" + pre_return + "\n",
            ))
            edits.append((
                node.end_byte,
                "\n    }",
            ))

    # 6. Add macro definitions at start of file
    macros = _build_macros(is_t0)
    edits.append((0, macros + "\n\n"))

    # Sort edits by byte offset descending, with
    # post-region (ACCUM) before pre-region (T) at same
    # offset using stable sort
    edits.sort(key=lambda e: -e[0])

    # Apply edits
    result = source_text
    for offset, text in edits:
        result = result[:offset] + text + result[offset:]

    return result


def _instrument_device_functions(
    edits, dev_funcs, dev_sub_regions, all_regions, is_t0,
):
    """Add profiling to device function bodies + signatures.

    For each device function with sub-regions:
    1. Add `long long* _prof` (and `int _is_prof_t0`)
       parameters to the function signature.
    2. Add temp variable declarations at body start.
    3. Insert PROF_T / PROF_ACCUM around each sub-region.
    """
    for fname, subs in dev_sub_regions.items():
        fnode = dev_funcs[fname]
        if not subs:
            continue

        # Find parameter list
        param_list = None
        for child in fnode.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type == "parameter_list":
                        param_list = sub
                        break
        if param_list is None:
            continue

        # Find body
        fbody = None
        for child in fnode.children:
            if child.type == "compound_statement":
                fbody = child
                break
        if fbody is None:
            continue

        # Max depth within this function's sub-regions
        max_sub_depth = max(r["depth"] for r in subs) + 2

        # Add prof params before closing ')'
        extra_params = ",\n    long long* _prof"
        if is_t0:
            extra_params += ",\n    int _is_prof_t0"
        edits.append((
            param_list.end_byte - 1,
            extra_params + "\n",
        ))

        # Add temp vars after opening '{'
        tvars = []
        for i in range(max_sub_depth):
            tvars.append(f"    long long _pt{i};")
        edits.append((
            fbody.start_byte + 1,
            "\n" + "\n".join(tvars) + "\n",
        ))

        # PROF_T before / PROF_ACCUM after each sub-region
        for r in subs:
            depth = r["depth"]
            # Use depth relative to the call region
            # (sub-regions re-use _pt0, _pt1, etc.)
            call_parent = next(
                (c for c in all_regions
                 if c["label"].endswith(f":call_{fname}")),
                None,
            )
            local_depth = (
                depth - call_parent["depth"] - 1
                if call_parent else depth
            )
            tv = f"_pt{local_depth}"
            rid = r["id"]
            if is_t0:
                edits.append((
                    r["_start_byte"],
                    f"    PROF_T0(_is_prof_t0, {tv}); "
                    f" // r{rid}:{r['label']}\n",
                ))
                edits.append((
                    r["_end_byte"],
                    f"\n    PROF_ACCUM0(_is_prof_t0, _prof,"
                    f" {rid}, {tv});  // end r{rid}",
                ))
            else:
                edits.append((
                    r["_start_byte"],
                    f"    PROF_T({tv}); "
                    f" // r{rid}:{r['label']}\n",
                ))
                edits.append((
                    r["_end_byte"],
                    f"\n    PROF_ACCUM(_prof, {rid}, {tv});"
                    f"  // end r{rid}",
                ))


def _patch_call_sites(edits, body, dev_funcs, is_t0,
                      dev_sub_regions=None):
    """At each call to an instrumented device function
    that has sub-regions, append _prof (and _is_prof_t0)
    as extra arguments."""
    if dev_sub_regions is None:
        dev_sub_regions = {}
    patched_funcs = set(dev_sub_regions.keys())
    for node in _walk(body):
        if node.type != "call_expression":
            continue
        callee = _get_callee_name(node)
        if callee not in patched_funcs:
            continue
        # Find argument_list
        arg_list = None
        for child in node.children:
            if child.type == "argument_list":
                arg_list = child
                break
        if arg_list is None:
            continue
        # Insert extra args before closing ')'
        extra = ", _prof"
        if is_t0:
            extra += ", _is_prof_t0"
        edits.append((
            arg_list.end_byte - 1,
            extra,
        ))


def _build_macros(is_t0):
    """Build profiling macro definitions.

    Always emit both sets — device functions may use T0
    even when the kernel uses per-thread, and vice versa.
    """
    lines = [
        "// === Auto-generated profiling macros ===",
        "#define PROF_T(var) var = clock64()",
        "#define PROF_ACCUM(arr, idx, start_var) "
        "arr[idx] += clock64() - start_var",
        "#define PROF_T0(cond, var) "
        "if (cond) var = clock64()",
        "#define PROF_ACCUM0(cond, arr, idx, sv) "
        "if (cond) arr[idx] += clock64() - sv",
    ]
    return "\n".join(lines)


def _build_declarations(num_regions, max_depth, is_t0):
    """Build profile variable declarations."""
    lines = [
        f"    // Auto-profiling: {num_regions} regions",
        f"    long long _prof[{num_regions}];",
        f"    for (int _pi = 0; _pi < {num_regions};"
        f" _pi++) _prof[_pi] = 0;",
    ]
    if is_t0:
        lines.append(
            "    int _is_prof_t0 = (threadIdx.x == 0);"
        )
    for i in range(max_depth):
        lines.append(f"    long long _pt{i};")
    return "\n".join(lines)


def _build_writeback(num_regions, is_t0):
    """Build profile data writeback code."""
    lines = []
    if is_t0:
        lines.extend([
            "    // Auto-profile writeback (thread 0)",
            "    if (_is_prof_t0) {",
            "        int _prof_wu = blockIdx.x;",
            f"        for (int _pr = 0; "
            f"_pr < {num_regions}; _pr++)",
            f"            profile_output["
            f"_prof_wu * {num_regions} + _pr"
            f"] = _prof[_pr];",
            "    }",
        ])
    else:
        lines.extend([
            "    // Auto-profile writeback",
            "    {",
            "        int _prof_gid = "
            "blockIdx.x * blockDim.x + threadIdx.x;",
            f"        for (int _pr = 0; "
            f"_pr < {num_regions}; _pr++)",
            f"            profile_output["
            f"_prof_gid * {num_regions} + _pr"
            f"] = _prof[_pr];",
            "    }",
        ])
    return "\n".join(lines)
