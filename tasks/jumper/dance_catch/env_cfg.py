"""Environment config for jumper.dance_catch: YENA's Catch Catch, `media/catch.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## This clip, measured

At the 50 Hz control rate, after conversion (77.0 s): per-joint position std 0.191
rad at the median and 0.355 at the most active, RMS amplitude 0.223 rad; joint
velocity RMS 1.044 rad/s, per-joint std 0.954 at the median and 1.474 at the most
active, peaking at 10.4 rad/s; the base between 87 and 118 mm; tilt at most 11.7
degrees. Measured with the same script that reproduces jumper.dance_maze's quoted
figures exactly, as the control.

**The reward weights are starting points**, as they are for jumper.dance, and one run
has been trained on them: 10 000 iterations at 1024 environments (2026-10-16_13-21-12,
RTX 4060). Replayed through the whole 77 s with no noise, model_9999 tracks the joints
within 55 mrad on average (78 worst-step mean), the base height within 3.1 mm (11) and
the tilt within 3.3 degrees (7.4), drifts 9 mm in the plane (23), and keeps the claw
tips 24.8 mm above the floor at the lowest.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.velocity.mdp import illegal_contact

from ..common.dance.env import dance_env_cfg

#: This task's material, retargeted from video (see `__init__.py`).
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. jumper.dance's: the clip (77.0 s) is far longer, so an
#: episode is a window into it and the adaptive sampler chooses where it starts.
#: 10 s at 136 bpm is about 23 beats, enough for several complete moves.
EPISODE_S = 10.0


#: Claw-on-floor contact above this force, newtons: the self-collision threshold.
CLAW_FLOOR_FORCE = 1.0
#: Per substep a claw presses on the floor. Four substeps a step, so a claw held
#: down costs 8.0 a step, more than the whole 6.5 positive budget.
CLAW_FLOOR_WEIGHT = -2.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    cfg = dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot (10 mm is a tenth of its standing
        # height), not of the clip, so they carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking terms
        # unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude, which scores a robot standing still through the clip at
        # 0.006. Here: 0.44 x 0.223 -> 0.098, rounded to 0.1.
        joint_pos_std=0.1,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS, standing still
        # about 0.10. Here: 1.044 -> 0.7. Half the position weight, so when
        # they disagree the pose wins.
        joint_vel_std=0.7,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's ~1.0 rad/s is ~6 W, so -0.02 is
        # about -0.11 a step, 2% of the budget -- about jumper.dance's figure.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. The reference crosses the soft limits (0.9 of the range) on five
        # leg joints and both claw wrists (`*F_J3_joint`), worst `RR_J2_joint` by up to
        # 82 mrad on 3.7% of frames; 0.0029 rad a step summed over all of them on
        # average, so a faithful policy pays about -0.03 a step, 0.4% of the 6.5 budget.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through 87-118 mm, a 31 mm band, so 40 mm
        # exceeds all of it and cannot fire on a policy merely tracking badly --
        # jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over". The dance itself tilts at
        # most 11.7 degrees (the dancer's lean, scaled).
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
    # ── The claws never touch the floor ──────────────────────────────────
    # This task's alone: the other dances' clips put the claws down. Here the clip
    # keeps the tips 38 mm up at the lowest, yet a policy trained without this put
    # a claw within 12 mm of the floor on 46% of a full replay (the run of
    # 2026-10-14_15-27-07, model_9999, RTX 4060) -- leaning on them for balance,
    # which no other term priced. So contact is both priced and ends the
    # episode, forfeiting every tracking reward after it. The recipe is the one
    # `constants.TIP_LINKS` describes, applied to the whole claw from the palm down.
    cfg.scene.sensors = cfg.scene.sensors + (
        ContactSensorCfg(
            name="claw_floor",
            primary=ContactMatch(mode="subtree", pattern=("LF_palm_link", "RF_palm_link"),
                                 entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            history_length=cfg.decimation,
        ),
    )
    params = {"sensor_name": "claw_floor", "force_threshold": CLAW_FLOOR_FORCE}
    cfg.rewards["claw_floor"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=CLAW_FLOOR_WEIGHT, params=params
    )
    cfg.terminations["claw_floor"] = TerminationTermCfg(func=illegal_contact, params=params)
    return cfg
