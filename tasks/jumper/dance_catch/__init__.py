"""jumper.dance_catch -- YENA's "Catch Catch", learned by imitation.

The same task as `jumper.dance` with a different clip: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/catch.npz`, 77.0 s at 50 Hz: the official dance practice video from its
start to 1:17 (136 BPM), retargeted from it.
It was made outside this repository in two steps:

- **GVHMR** recovered one dancer's 3D body motion (SMPL-X) from the video. Its
  tracker hands over between dancers twice, during the formation changes at
  about 28-36 s and 46-48 s, so three dancers are followed in turn; the
  retarget reads each body relative to itself. The handovers are where the
  clip is fastest -- 10.4 rad/s at most, at 27.5 s, against 6.0 in the first
  24 s -- and still a third of the servo's 30.7.
- **A retarget script** mapped it onto the jumper: each wrist relative to its
  shoulder moves the matching claw. Arms down is the claws hovering 35 mm
  above where they rest at HOME, not on the floor: a first clip rested them
  there, put them within 12 mm of the floor on 13% of its frames, and its
  policy struck the floor with them -- this one keeps the tips 38 mm up at the
  lowest. The pelvis bounce and torso lean become body height, roll, pitch and yaw,
  solved for the four support legs with their feet held where they stand at
  HOME. The dancer's steps are not reproduced. The base pose was then solved
  from the feet with `tools/import_wbc_dances.py::solve_base_pose`, as the
  imported dances' were.

No music ships with it, and training does not need any. The performance video
`scripts/export.py` renders does, so export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_catch",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating YENA's Catch Catch (to 1:17), retargeted "
                "from video; the clip is in tasks/jumper/dance_catch/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
