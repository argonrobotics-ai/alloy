# Known limits

[docs/known_gaps/](../known_gaps/) lists the engine limits we have measured, one Markdown file each, named by area and
topic (`contact-...`, `coupling-...`, `soft-...`, `mpm-...`). The folder is the list: a new limit appears there with
no other edit, so read the folder, not a copy of it. Each file says what is wrong, the number (how big the error is,
measured how), why, what to do instead today, and the planned fix. It is not every limit: a guide page can state one
that has no file yet. Before you trust a result in an area (contact, coupling, soft bodies, cutting, liquids and
pastes), read the files for it and that area's guide page.

```bash
grep -h -m1 "^# " docs/known_gaps/*.md        # one title per known limit
grep -il "viscosity" docs/known_gaps/*.md     # the limits that mention a word from your task
```

Limits that change how you build a scene are also stated where you would meet them: AGENTS.md section 7 for liquids in
narrow containers, section 8 for precise fits and for fast bodies passing through thin parts, and the matching guide
pages. The limit's own file has the numbers and the status of the fix.

## Doing next

Larger pieces of work that are planned and not built yet. None of them is in this release, and we promise no dates.

- Cloth that takes up liquid. Today a liquid and a cloth collide and nothing more: no wetting, no soaking in, so a
  towel does not absorb a spill ([liquid-on-cloth.md](../known_gaps/liquid-on-cloth.md)). Planned: an absorption and
  wetting model, first as a reviewed design (the model, the law that checks it, its cost), then as an opt-in
  prototype.
- The welding task score. Gym-Anything's four welding tasks, judged by a port of its evaluator, score 0.75 to 0.78
  of 1 and none succeeds: the bead's steady part is sound, but the arc start, the tack starts and the end crater are
  wrong
  ([weld-bead-starts-and-craters.md](../known_gaps/weld-bead-starts-and-craters.md)). Planned: those three misses,
  one at a time.
- Dents in solid parts by default. Today a metal sheet or wire keeps a dent or a bend (`deformables.md`, `rods.md`),
  and a solid block that yields is opt-in and reads a punch's force low
  ([plasticity-mpm-solid-punch-force.md](../known_gaps/plasticity-mpm-solid-punch-force.md)). Planned: the punch
  laws checked against the textbook indentation depth first, then yielding solids on that route by default.

## Writing a limit file

One limit per file, named `<area>-<topic>.md`, with its owner in front matter. Start from this template. The five
sections are required, each with text; `tests/oss_docs/test_known_gaps.py` fails on a missing or empty one. A section
can be a `## Number` heading, as here, or a paragraph that starts with a bold lead-in such as `**The number.**`.
Accepted names: What / What is wrong, Number / The number, Why, Workaround, Planned fix / The fix and when. Add other
sections after them if you need to.

```markdown
---
owner: <owner>
---
# <the limit in one line, e.g. "Liquid leaks out of a container narrower than 24 mm">

## What is wrong
What a user sees, in the user's terms: the call or scene, what happens, what should happen.

## Number
The measured size: the quantity with its unit, the scene, device, dt and worlds, and how it was measured. Say when
it is an estimate.

## Why
The cause, as far as it is known ("not found yet" is an answer).

## Workaround
What to do today: the call, setting or scene change, and what it costs.

## Planned fix
What will change, and when or on what it depends. "None planned" is an answer.
```

Check your file before you merge: `python -m pytest tests/oss_docs/test_known_gaps.py -q`.
