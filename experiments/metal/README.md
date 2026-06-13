# Metal experiments for NeuroMorph acceleration

These probes are Mac-only experiments for checking whether Apple GPU compute can
help Cereblix mining. They do not change the miner binary.

Build on macOS:

```sh
clang -O3 -fobjc-arc -framework Foundation -framework Metal metal_aes128_probe.m -o /tmp/metal_aes128_probe
clang -O3 -fobjc-arc -framework Foundation -framework Metal metal_aes128_ttable_probe.m -o /tmp/metal_aes128_ttable_probe
clang -O3 -fobjc-arc -framework Foundation -framework Metal metal_scratch_probe.m -o /tmp/metal_scratch_probe
clang -O3 -fobjc-arc -framework Foundation -framework Metal metal_mix_probe.m -o /tmp/metal_mix_probe
clang -O3 -fobjc-arc -framework Foundation -framework Metal metal_random_probe.m -o /tmp/metal_random_probe
```

Run:

```sh
/tmp/metal_aes128_probe
/tmp/metal_aes128_ttable_probe
/tmp/metal_scratch_probe
/tmp/metal_mix_probe
/tmp/metal_random_probe
```

The AES probe must print the NIST AES-128 test vector ciphertext:

```text
69c4e0d86a7b0430d8cdb78070b4c55a
```
