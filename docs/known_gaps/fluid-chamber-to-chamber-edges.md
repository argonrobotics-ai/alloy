# A cylinder's force pressure lags when its chamber is joined to another chamber by a wide edge

**What.** Each actuator reads its chamber's pressure from the node's own implicit response to the step's volume change
(`_force_pressure` in `alloy/plugins/fluid/actuators.py`): P + (P' - P)·κ/(κ + dt·Σg). That response holds every
neighbour node at its step-start pressure, which is right for a reservoir or a large accumulator. When the edge goes to
another small chamber and dt·Σg ≫ κ, the force reads close to the step-start pressure. The oil column's stiffness then
acts a step late, and its ringing grows.

A second effect of the same split: the network adds a step's source inflow after the force is taken. While a pump is
filling a chamber, the node's end-of-step `pressure` therefore reads up to K·Q·dt/V above the pressure the piston acts
with.

**Number.** GA's forklift lift line (SCR-WP8-12 F2): a pump node of 0.5 L joined by a 1e-7 m³/(s·Pa) hose to the
free-lift and main cylinders (0.1 L dead volume each), at the 1.25 ms task step.
- Unladen carriage: the lift pressure swung between -3 and +5 MPa gauge, and the carriage crept 10 mm while held.
- Gauge while flowing (one shared node instead, 0.1 L dead volume per cylinder): in mid free lift the node read
  8.0 MPa while the piston acted with 5.97 MPa (34 % high). With the line's 1 L in the dead volumes it reads about
  0.7 MPa high. Holding pressures read true.

**Why.** Per step, each actuator writes its chambers' volumes and takes its force from the pressure each node would
reach on its own; the network then steps every node together (implicit in the conductances) and adds the sources'
inflow. A neighbour held at its old pressure is right for a reservoir but not for a small chamber, whose pressure
moves with the first one's within the step; and the inflow arrives after the force was taken.

**Workaround.**
- For a common line (one pump teed to several cylinders), declare the node `Node(shared=True)`: one oil volume and one
  pressure, the chamber of every actuator on it, with the coupling implicit through the node.
- For pressure readouts while oil flows, use `piston_pressure` (the gauge pressure each actuator's force acted with,
  A and B sides) instead of the node's `pressure`.

**Planned fix.** Not scheduled (WP13). A network-implicit force pressure: the coupled response of the connected
nodes to the actuators' volume changes, and the step's known source inflow added to the force's prediction.
