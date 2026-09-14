# On-Twos Retimer (MMD Physics Path)

Convert MMD physics simulation (rigid-body cache driven bone motion) into a
stepped **on-N** (hold every N frames) keyframe animation.

MMD physics baking produces *point cache*, not bone keyframes — the visible
hair/cloth motion is constraint-driven at evaluation time. This add-on closes
the loop in two steps:

1. **Bake physics → bone keyframes**: samples the constraint-evaluated pose per
   frame and writes bone keyframes (with optional physics locking).
2. **Apply on-N**: decimates keyframes (keeps frames where `(frame - phase) % N == 0`)
   and sets constant interpolation for the "hold" look.

## Features

- **Physics → bone keyframes** in one operator: restores physics, samples the
  final evaluated pose per frame, writes bone keyframes, optionally locks
  physics (mutes rigid-track constraints + disables the rigid body world)
- **Two-pass bake option**: keeps the damping-track look *and* a deterministic,
  re-jump-safe result
- **Apply on-N**: decimate + constant interpolation (on-twos = 2, on-threes = 3)
- **Stepped-modifier mode**: non-destructive stepping like doing it by hand in
  the F-Curve editor, applied only to physics-driven bone channels
- **MMD-aware**: auto-detects physics bones via MMD bone/rigid-body data and
  `mmd_tools_rigid_track` constraints; dense-curve detection as a fallback for
  non-MMD rigs
- **Restore**: exact restore from a saved backup action (survives save/reopen),
  or a generic fallback (removes stepped modifiers / smooths constant keys)
- All operators are undoable (Ctrl+Z); non-destructive where possible
- Compatible with Blender 4.x / 5.0

## Requirements

- Blender 4.2 or newer
- [MMD Tools](https://github.com/powroupi/blender_mmd_tools) add-on for the
  physics baking step (optional if you already have baked physics)

## Installation

### Option A — Extensions Platform (recommended)

1. In Blender: Edit → Preferences → Get Extensions → *Install from Disk*.
2. Select the downloaded `.zip` and enable **On-Twos Retimer**.

### Option B — Manual

1. Download the `.zip` from the GitHub Releases page.
2. Edit → Preferences → Add-ons → Install from Disk, select the `.zip`.
3. Enable **On-Twos Retimer**.

## Usage

1. Bake physics with MMD Tools (produces the rigid body cache).
2. In the 3D Viewport N panel → "一拍二" tab:
   a. Set the frame range → click **Bake physics → bone keyframes** (physics is
      auto-restored first if it was locked);
   b. Click **Preview stats** to confirm the affected curves → click **Apply on-N**.
3. Not satisfied: Ctrl+Z, or **Unlock physics** and re-bake.

## Support

- Issues & feature requests: GitHub Issues
- This project is open source. Contributions are welcome.

## License

GPL-2.0-or-later — see [LICENSE](LICENSE).
