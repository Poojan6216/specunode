# Third-party data in this directory

`traces.json`, `values.json` and their manifests are derived from
[`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
by Nebius, licensed under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
That licence, not this repository's Apache-2.0 licence, governs these files.

**What was changed.** A sample of 300 of the dataset's trajectories was taken, and each was
reduced to the ordered sequence of its tool calls:

- `traces.json` keeps, for each call, the tool's name, its argument *keys*, and whether an
  argument refers to an earlier call's result. Argument values, tool results, model messages and
  everything else in the dataset are dropped.
- `values.json` keeps each call's argument *values*, step for step alongside `traces.json`, with
  any value longer than 64 canonical bytes replaced by a digest of it.

`fetch.py` in this directory reproduces both from the dataset; the manifests record the dataset,
the sample and a hash of the result. `values_full.json`, which keeps every value whole, is
produced on demand by `fetch.py --values --full` and is not distributed.
