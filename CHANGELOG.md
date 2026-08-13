# Changelog

## Version 0.3

- `quip-coordinator seed-chain` sets the default topology and difficulty on a
  new chain. This command takes the place of v0.2
  `scripts/seed-advantage2-topology.py`. The v0.3 image does not ship the
  Python chain modules that script imported. The command takes `--sudo-key` or
  `--mnemonic-file`. `--sudo-key` accepts a `//DevUri`, a BIP39 phrase, a
  32-byte hex seed, or a keystore path. The default topology is
  `advantage2-system1`.
- The coordinator binary embeds the topology presets. `drive --topology-preset`
  used to look up a path on the build machine. The release image does not have
  that path, so the flag failed in the image.
- `--sudo-key` and the `signer_key` setting accept a BIP39 phrase and any
  substrate secret URI. They also accept a keystore path, a `//DevUri`, and a
  32-byte hex seed.
