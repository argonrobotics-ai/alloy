# Scan-texture test fixture: source and licence

`meshes/ycb_004_sugar_box_front/visual.glb` (and the card and material beside it) is a derivative of one object of
the YCB Object and Model Set:

- Object: 004_sugar_box, the `google_16k` scan (mesh, UV map and photo texture).
- Source: http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/004_sugar_box_google_16k.tgz
  (ycbbenchmarks.com), as imported into Alloy's object library (`alloy fetch ycb_004_sugar_box`).
- Authors: B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel, A. M. Dollar, "Benchmarking in Manipulation
  Research: The YCB Object and Model Set and Benchmarking Protocols", IEEE Robotics and Automation Magazine, 2015;
  B. Calli et al., "Yale-CMU-Berkeley dataset for robotic manipulation research", IJRR, 2017.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ (full text: `alloy/_vendor/newton/licenses/CC-BY-4.0.txt`).

Changes made by Alloy (`make_scan_fixture.py` in this folder rebuilds them from the fetched scan):
- Kept only the scan's triangles that face the box's front (5,272 of 16,384), with their UVs unchanged.
- Shrank the texture from 4,096 to 512 pixels (box filter) and re-encoded it as JPEG (quality 90).
- Replaced the physics mesh with the front panel's bounding box, renamed the card to `ycb_004_sugar_box_front`, and
  copied the imported material card (`ycb_004_sugar_box_eff`) under that name.

The fixture is used only by Alloy's tests (`tests/camera/test_scan_texture_law.py`).
