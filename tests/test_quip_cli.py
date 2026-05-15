"""Unit tests for quip_cli helpers: topology parsing, injection, and hashing."""

from __future__ import annotations

import pytest
import click

import quip_cli


def test_parse_topology_valid_zephyr():
    topo = quip_cli._parse_topology("zephyr:2,2")
    assert topo.num_nodes > 0
    assert topo.num_edges > 0


def test_parse_topology_missing_colon():
    with pytest.raises(click.BadParameter, match="family:m,t"):
        quip_cli._parse_topology("zephyr22")


def test_parse_topology_unknown_family():
    with pytest.raises(click.BadParameter, match="only 'zephyr'"):
        quip_cli._parse_topology("pegasus:2,2")


def test_parse_topology_bad_params_non_integer():
    with pytest.raises(click.BadParameter):
        quip_cli._parse_topology("zephyr:a,b")


def test_inject_topology_cpu_adds_to_args():
    topo = quip_cli._parse_topology("zephyr:2,2")
    config = {"cpu": {"num_cpus": 2}}
    result = quip_cli._inject_topology(config, "cpu", topo)
    assert result["cpu"]["args"]["topology"] is topo


def test_inject_topology_cpu_preserves_existing_args():
    topo = quip_cli._parse_topology("zephyr:2,2")
    config = {"cpu": {"num_cpus": 1, "args": {"foo": "bar"}}}
    result = quip_cli._inject_topology(config, "cpu", topo)
    assert result["cpu"]["args"]["foo"] == "bar"
    assert result["cpu"]["args"]["topology"] is topo


def test_inject_topology_does_not_mutate_input():
    topo = quip_cli._parse_topology("zephyr:2,2")
    config = {"cpu": {"num_cpus": 1}}
    original = {"cpu": {"num_cpus": 1}}
    quip_cli._inject_topology(config, "cpu", topo)
    assert config == original


def test_inject_topology_gpu_returns_unchanged():
    topo = quip_cli._parse_topology("zephyr:2,2")
    config = {"cuda": [{"device": "0"}]}
    result = quip_cli._inject_topology(config, "gpu", topo)
    assert result == config


def test_zephyr_topology_hash_is_deterministic():
    topo = quip_cli._parse_topology("zephyr:2,2")
    h1 = quip_cli._zephyr_topology_hash(topo)
    h2 = quip_cli._zephyr_topology_hash(topo)
    assert h1 == h2
    assert len(h1) == 32


def test_zephyr_topology_hash_differs_across_specs():
    topo_22 = quip_cli._parse_topology("zephyr:2,2")
    topo_32 = quip_cli._parse_topology("zephyr:3,2")
    assert quip_cli._zephyr_topology_hash(topo_22) != quip_cli._zephyr_topology_hash(
        topo_32
    )
