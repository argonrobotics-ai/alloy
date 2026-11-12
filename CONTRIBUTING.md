# Contributing to Alloy

Thanks for helping. Alloy is Apache-2.0; by sending a pull request you agree that your contribution is licensed
under the same terms (see `LICENSE`).

## Before you open a pull request

- Run the CPU suite: `scripts/ci_cpu.sh` (or `python -m pytest tests -m "not gpu and not slow"`). GPU tests
  (`-m gpu`) skip without an NVIDIA GPU; run them if your change touches GPU code.
- Every new public function has a docstring with an example; every error message says what to change.
- New physics goes behind Newton's extension points (solvers, the collision pipeline, the model builder) and comes
  with a test that checks it against a law or a reference, not only that it runs.
- Material and contact values in `library/` cite a public source in the card's `note` (handbook, standard, data
  sheet or paper); estimates say so and carry a wide `unc`.
- Keep files under 500 lines and functions under 50.
- Third-party code or data: name the source, commit and licence in the file header and add it to `NOTICE`. Only
  Apache-2.0, MIT, BSD, CC0 and CC-BY material can be accepted; no non-commercial assets.
- `alloy/_vendor/` holds Newton and MuJoCo Warp with our fixes. A change there keeps the file's upstream header and
  is listed in `alloy/_vendor/VENDORED.md`; import them as `alloy._vendor.newton` and `alloy._vendor.mujoco_warp`.

## Reporting bugs

Open an issue (https://github.com/argonrobotics-ai/alloy/issues) with the smallest script that shows the problem, the device (`cpu` or the GPU model), and the
versions printed by `python -c "import alloy, warp; print(alloy.__version__, warp.config.version)"`.

Security problems: see `SECURITY.md`, not the public issue tracker.
