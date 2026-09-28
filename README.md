# On-Twos Retimer

Turn constraint / physics-driven bone animation into a stepped **on-N**
(hold every N frames) keyframe animation, in two steps:

1. **Bake poses → bone keyframes**: samples the constraint/physics-evaluated
   pose per frame and writes bone keyframes (with optional locking of the
   driving setup).
2. **Apply on-N**: decimates keyframes (keeps frames where `(frame - phase) % N == 0`)
   and sets constant interpolation for the classic "held frame" look.

Works with **any armature** whose bones are driven by constraints or physics
(IK, Damped Track, rigid-body simulation, cloth/hair chains, etc.). An
optional compatibility mode restricts baking to MMD physics bones for MMD rigs.

## Features

- **Bake poses → bone keyframes** in one operator: restores the driving setup,
  samples the final evaluated pose per frame, writes bone keyframes, optionally
  locks the drivers (mutes constraints + disables the rigid body world)
- **Target selection**: bones selected in Pose Mode (default) / MMD
  rigid-track bones (optional compatibility mode for MMD rigs); the
  **Include Child Chains** option grabs whole chains from a parent bone,
  and the **Highlight Target Bones** button previews exactly which bones
  would be baked
- **Two-pass bake option**: keeps the soft damping look *and* a deterministic,
  re-jump-safe result
- **Apply on-N**: decimate + constant interpolation (on-twos = 2, on-threes = 3)
- **Stepped-modifier mode**: non-destructive stepping like doing it by hand in
  the F-Curve editor, applied only to target bone channels
- **Dense-curve detection**: only processes curves that look like baked (dense)
  animation, so hand-keyed animation is never touched
- **Restore**: exact restore from a saved backup action (survives save/reopen),
  or a generic fallback (removes stepped modifiers / smooths constant keys)
- All operators are undoable (Ctrl+Z); non-destructive where possible
- Compatible with Blender 4.x / 5.0

## Requirements

- Blender 4.2 or newer
- No other add-ons required. For MMD rigs, the "MMD Rigid-Track Bones" target
  mode needs [MMD Tools](https://github.com/powroupi/blender_mmd_tools).

## Installation

### Option A — Extensions Platform (recommended)

1. In Blender: Edit → Preferences → Get Extensions → *Install from Disk*.
2. Select the downloaded `.zip` and enable **On-Twos Retimer**.

### Option B — Manual

1. Download the `.zip` from the GitHub Releases page.
2. Edit → Preferences → Add-ons → Install from Disk, select the `.zip`.
3. Enable **On-Twos Retimer**.

## Usage

1. Select an armature whose bones are driven by constraints or physics.
2. In the 3D Viewport N panel → "On-Twos" tab:
   a. Set the frame range and target bones → click **Bake Poses → Bone Keyframes**
      (the driving setup is auto-restored first if it was locked);
   b. Click **Preview Stats** to confirm the affected curves → click **Apply On-N**.
3. Not satisfied: Ctrl+Z, or **Unlock Physics** and re-bake.

## Support

- Issues & feature requests: GitHub Issues
- This project is open source. Contributions are welcome.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
