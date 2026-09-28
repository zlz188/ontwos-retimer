# -*- coding: utf-8 -*-
"""
On-Twos Retimer
==============

Bone motion that is driven by constraints or physics (IK, Damped Track,
rigid-body simulation, etc.) is evaluated at playback time and produces no
keyframes. This add-on closes the loop in two steps:

1. Bake poses -> bone keyframes: sample the constraint-evaluated pose per
   frame and write bone keyframes (with optional locking of the driving setup).
2. Apply on-N: decimate keyframes (keep keys where (frame - phase) % N == 0)
   + constant interpolation for the classic "held frame" look.

Works with any armature whose bones are driven by constraints or physics.
An optional compatibility mode restricts baking to MMD physics bones
(bones with 'mmd_tools_rigid_track' constraints) for MMD rigs.

Notes
-----
- Step 1 and 2 are both UNDO-able (Ctrl+Z); step 1 is non-destructive
  (locking can be restored: unmute constraints + re-enable the physics world).
- Compatible with Blender 4.x / 5.0.

Usage
-----
1. Select an armature (bones driven by constraints / physics).
2. 3D View N panel "On-Twos" tab:
   a. Set the range and target bones -> click [Bake Poses -> Bone Keyframes];
   b. Click [Preview Stats] to confirm the affected curves -> click [Apply On-N].
3. Not satisfied: Ctrl+Z; or Unlock Physics (panel button) and re-bake.
"""

bl_info = {
    "name": "On-Twos Retimer",
    "author": "zlz",
    "version": (0, 7, 0),
    "blender": (4, 0, 0),
    "location": "3D View > Sidebar > On-Twos",
    "description": "Bake constraint/physics-driven bone poses into keyframes, then decimate + hold every N frames (on-twos / on-threes).",
    "category": "Animation",
    "wiki_url": "https://github.com/zlz188/ontwos-retimer",
    "tracker_url": "https://github.com/zlz188/ontwos-retimer/issues",
}

import math
import re

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _bone_from_data_path(data_path):
    """从 fcurve.data_path 中提取骨骼名，非骨骼曲线返回 None。"""
    m = re.match(r'pose\.bones\["([^"]+)"\]', data_path or "")
    return m.group(1) if m else None


def _key_frames_in_range(fcurve, start, end):
    frames = set()
    for kp in fcurve.keyframe_points:
        f = int(round(kp.co.x))
        if start <= f <= end:
            frames.add(f)
    return frames


_DENSE_MIN_KEYS = 8
_DENSE_RATIO = 0.9


def _is_dense(fcurve, start, end):
    """曲线在 [start,end] 内是否呈现"物理烘焙"的密集特征（按自身跨度算密度）。"""
    if end < start:
        return True
    frames = sorted(_key_frames_in_range(fcurve, start, end))
    if len(frames) < _DENSE_MIN_KEYS:
        return False
    span = frames[-1] - frames[0] + 1
    return len(frames) / span >= _DENSE_RATIO


def _mmd_physics_bone_names(obj):
    """若 obj 是带 MMD 数据的骨架，返回物理驱动骨骼名集合；无法识别返回 None。"""
    if obj.type != 'ARMATURE':
        return None
    names = set()
    has_mmd_data = False
    if hasattr(bpy.types.Bone, "mmd_bone"):
        for bone in obj.data.bones:
            mmd = getattr(bone, "mmd_bone", None)
            if mmd is not None:
                has_mmd_data = True
                if getattr(mmd, "type", None) == 'PHYSICS':
                    names.add(bone.name)
    if hasattr(bpy.types.Object, "mmd_rigid"):
        for rb_obj in bpy.data.objects:
            rigid = getattr(rb_obj, "mmd_rigid", None)
            if rigid is None:
                continue
            has_mmd_data = True
            if getattr(rigid, "type", None) != 'DYNAMIC':
                continue
            bone_name = getattr(rigid, "bone", "")
            if bone_name and bone_name in obj.data.bones:
                names.add(bone_name)
    if not has_mmd_data:
        return None
    return names or None


def _rigid_track_bones(obj):
    """带 mmd_tools_rigid_track 约束（MMD物理驱动）的骨骼名集合。"""
    names = set()
    for pbone in obj.pose.bones:
        for constr in pbone.constraints:
            if constr.name == "mmd_tools_rigid_track":
                names.add(pbone.name)
                break
    return names


def get_bone_and_descendants(armature, bone_names_set):
    """递归获取给定骨骼集合的所有后代骨骼名称（用于『包含子链』模式）。"""
    result = set(bone_names_set)
    children_map = {}
    for bone in armature.data.bones:
        if bone.parent:
            children_map.setdefault(bone.parent.name, set()).add(bone.name)

    to_process = list(bone_names_set)
    while to_process:
        current = to_process.pop()
        if current in children_map:
            for child in children_map[current]:
                if child not in result:
                    result.add(child)
                    to_process.append(child)
    return result


def _rot_data_path(pbone):
    return {
        'QUATERNION': "rotation_quaternion",
        'AXIS_ANGLE': "rotation_axis_angle",
    }.get(pbone.rotation_mode, "rotation_euler")


def _matrix_finite(mat):
    for row in mat:
        for x in row:
            if not math.isfinite(x):
                return False
    return True


def _safe_inverted(mat):
    """安全求逆：退化/不可逆/非有限时返回 None。"""
    try:
        inv = mat.inverted()
        if _matrix_finite(inv):
            return inv
    except Exception:
        pass
    return None


def _parent_first(arm_obj, bone_names):
    """按父子关系排序（父在前），用于写入关键帧时保证链一致。"""
    order = []
    visited = set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        pb = arm_obj.pose.bones.get(name)
        if pb is None:
            return
        p = pb.parent
        if p is not None and p.name in bone_names:
            visit(p.name)
        order.append(name)

    for n in list(bone_names):
        visit(n)
    return order


def _bone_basis_for(arm_obj, pbone, target, mats):
    """由『目标骨架空间矩阵』显式反推 matrix_basis（绕开 pose_bone.matrix 赋值的陈旧父矩阵问题）。
    mats: 本帧已采样到的 {bone: 骨架空间矩阵}；父骨不在 mats 时取当前求值态父矩阵。
    退化骨骼的 rest 矩阵求逆失败时兜底为 identity rest，不产出 NaN。"""
    parent = pbone.parent
    bone = pbone.bone
    if parent is not None:
        if parent.name in mats:
            parent_pose = mats[parent.name]
        else:
            parent_pose = Matrix(parent.matrix)
        local_rest = _safe_inverted(parent.bone.matrix_local) @ bone.matrix_local
        if local_rest is None:
            local_rest = Matrix()  # 退化兜底：视为无 rest 偏移
        base = _safe_inverted(parent_pose @ local_rest)
        if base is not None:
            return base @ target
        return target
    base = _safe_inverted(bone.matrix_local)
    if base is not None:
        return base @ target
    return target


def _decimate_fcurve(fcurve, start, end, step, phase):
    """对单条曲线抽稀并恒定插值。返回 (删除数, 保留数)。只动 [start,end] 内的关键帧。"""
    removed = 0
    kept = 0
    # 先设置恒定插值（此时不删帧，引用有效）
    for kp in fcurve.keyframe_points:
        f = int(round(kp.co.x))
        if start <= f <= end and (f - phase) % step == 0:
            kp.interpolation = 'CONSTANT'
            kept += 1
    # 从后往前删除非保留帧（每次重新取引用，避免删除导致的引用失效）
    kps = fcurve.keyframe_points
    i = len(kps) - 1
    while i >= 0:
        kp = kps[i]
        f = int(round(kp.co.x))
        if start <= f <= end and (f - phase) % step != 0:
            kps.remove(kp)
            removed += 1
        i -= 1
    return removed, kept


def _count_removable(fcurve, start, end, step, phase):
    n = 0
    for kp in fcurve.keyframe_points:
        f = int(round(kp.co.x))
        if start <= f <= end and (f - phase) % step != 0:
            n += 1
    return n


def _copy_fcurve(src, dst):
    """把 src 曲线的关键帧数据（值/插值/手柄）完整复制到 dst 曲线。"""
    dst.data_path = src.data_path
    dst.array_index = src.array_index
    dst.extrapolation = src.extrapolation
    for kp in src.keyframe_points:
        nkp = dst.keyframe_points.insert(kp.co.x, kp.co.y)
        nkp.interpolation = kp.interpolation
        nkp.easing = kp.easing
        nkp.back = kp.back
        nkp.handle_left_type = kp.handle_left_type
        nkp.handle_right_type = kp.handle_right_type
        nkp.handle_left = (kp.handle_left.x, kp.handle_left.y)
        nkp.handle_right = (kp.handle_right.x, kp.handle_right.y)


def _backup_action(action):
    """为动作创建/复用『应用一拍N 前』的备份（含全部曲线与关键帧数据）。
    仅首次创建，重复应用不覆盖，保证能恢复到最后一次应用一拍N之前的原始动画。
    备份动作带假用户(use_fake_user)，会随 .blend 保存——保存并重开 Blender 后仍可精确恢复。
    返回备份动作。"""
    backup_name = f"__onetwos_backup_{action.name}"
    bak = bpy.data.actions.get(backup_name)
    if bak is not None:
        return bak
    bak = bpy.data.actions.new(name=backup_name)
    bak.use_fake_user = True
    for fc in action.fcurves:
        nfc = bak.fcurves.new(fc.data_path, index=fc.array_index)
        _copy_fcurve(fc, nfc)
    return bak


def _fallback_target_fcurves(context):
    """无备份时，用与『应用一拍N』一致的选择逻辑找出目标曲线（两种模式取并集），
    避免把躯干等手K骨骼的恒定插值也误改。"""
    targets = {}
    for _obj, fc in _stepped_fcurves(context):
        targets[(fc.data_path, fc.array_index)] = fc
    s, step, phase, start, end, sel_bones = _resolve_filters(context)
    t2, _sk_b, _sk_m, _sk_d = _collect_targets(
        context, s.only_dense, s.use_mmd_physics, sel_bones, start, end
    )
    for _obj, fc in t2:
        targets[(fc.data_path, fc.array_index)] = fc
    return list(targets.values())


def _smooth_onetwos_fallbacks(context, action, fcurves):
    """通用删除一拍N（无备份兜底）：移除 STEPPED 步进修改器；把目标曲线范围内
    的恒定插值关键帧改回平滑插值(BEZIER/AUTO，退化时 LINEAR)。返回处理的曲线数。"""
    s = context.scene.ontwos
    start, end = sorted((s.frame_start, s.frame_end))
    touched = 0
    for fc in fcurves:
        changed = False
        for mod in list(fc.modifiers):
            if mod.type == 'STEPPED':
                fc.modifiers.remove(mod)
                changed = True
        for kp in fc.keyframe_points:
            f = int(round(kp.co.x))
            if start <= f <= end and kp.interpolation == 'CONSTANT':
                try:
                    kp.interpolation = 'BEZIER'
                    kp.handle_left_type = 'AUTO'
                    kp.handle_right_type = 'AUTO'
                except Exception:
                    kp.interpolation = 'LINEAR'
                changed = True
        if changed:
            touched += 1
    return touched


def _collect_targets(context, only_dense, use_mmd, selected_bone_names, start, end):
    """收集目标曲线。返回 (targets, skipped_bone, skipped_mmd, skipped_dense)。"""
    targets = []
    skipped_bone = 0
    skipped_mmd = 0
    skipped_dense = 0
    for obj in context.selected_objects:
        adata = obj.animation_data
        if not adata or not adata.action:
            continue
        mmd_physics = _mmd_physics_bone_names(obj) if use_mmd else None
        for fcurve in adata.action.fcurves:
            if selected_bone_names is not None:
                bone = _bone_from_data_path(fcurve.data_path)
                if bone is not None and bone not in selected_bone_names:
                    skipped_bone += 1
                    continue
            if mmd_physics is not None:
                bone = _bone_from_data_path(fcurve.data_path)
                if bone is None or bone not in mmd_physics:
                    skipped_mmd += 1
                    continue
            elif only_dense and not _is_dense(fcurve, start, end):
                skipped_dense += 1
                continue
            targets.append((obj, fcurve))
    return targets, skipped_bone, skipped_mmd, skipped_dense


def _resolve_filters(context):
    s = context.scene.ontwos
    step = max(1, s.step)
    phase = s.phase % step
    start, end = sorted((s.frame_start, s.frame_end))
    selected_bone_names = None
    if s.only_selected_bones:
        bones = getattr(context, "selected_pose_bones", None)
        if bones:
            selected_bone_names = {b.name for b in bones}
    return s, step, phase, start, end, selected_bone_names


def _skip_summary(skipped_bone, skipped_mmd, skipped_dense):
    parts = []
    if skipped_mmd:
        parts.append(f"non-physics-detected bones {skipped_mmd}")
    if skipped_dense:
        parts.append(f"non-dense {skipped_dense}")
    if skipped_bone:
        parts.append(f"unselected bones {skipped_bone}")
    return "; ".join(parts) if parts else ""


def _bake_target_bones(arm_obj, context):
    """烘焙/步进/恢复的目标骨骼：按 target_mode 选择（默认选中骨骼，可选子链扩展；MMD 兼容模式）。"""
    s = context.scene.ontwos
    mode = s.target_mode
    if mode == 'RIGID_TRACK':
        return _rigid_track_bones(arm_obj)
    # 'SELECTED' - 默认：姿态模式选中的骨骼；勾选『包含子链』则自动扩展到整条子链
    bones = getattr(context, "selected_pose_bones", None)
    names = {b.name for b in bones} if bones else set()
    if names and s.follow_chain:
        names = get_bone_and_descendants(arm_obj, names)
    return names


def _remove_baked_keys(arm_obj, bone_names, start, end):
    """删除指定骨骼在 [start,end] 内的全部位置/旋转关键帧（用于恢复物理时清除烘焙残留）。"""
    adata = arm_obj.animation_data
    if not adata or not adata.action:
        return 0
    removed = 0
    for fc in adata.action.fcurves:
        bone = _bone_from_data_path(fc.data_path)
        if bone is None or bone not in bone_names:
            continue
        kps = fc.keyframe_points
        i = len(kps) - 1
        while i >= 0:
            kp = kps[i]
            f = int(round(kp.co.x))
            if start <= f <= end:
                kps.remove(kp)
                removed += 1
            i -= 1
    return removed


def _stepped_fcurves(context):
    """步进插值模式下要处理的目标：只取目标骨骼的通道，避免误伤躯干等手K骨骼。"""
    out = []
    selected_pose = None
    if context.scene.ontwos.only_selected_bones:
        bones = getattr(context, "selected_pose_bones", None)
        if bones:
            selected_pose = {b.name for b in bones}
    for obj in context.selected_objects:
        if obj.type != 'ARMATURE':
            continue
        adata = obj.animation_data
        if not adata or not adata.action:
            continue
        target = _bake_target_bones(obj, context)
        if not target:
            continue
        for fc in adata.action.fcurves:
            bone = _bone_from_data_path(fc.data_path)
            if bone is None or bone not in target:
                continue
            if selected_pose is not None and bone not in selected_pose:
                continue
            out.append((obj, fc))
    return out


def _apply_stepped(context, s, step, phase, start, end):
    """步进插值模式：给目标骨骼通道加 Stepped FModifier 并置恒定插值（非破坏，可删修改器还原）。"""
    touched = 0
    for _obj, fc in _stepped_fcurves(context):
        for mod in list(fc.modifiers):
            if mod.type == 'STEPPED':
                fc.modifiers.remove(mod)
        mod = fc.modifiers.new('STEPPED')
        mod.frame_step = max(1, step)
        mod.frame_start = phase % max(1, step)
        for kp in fc.keyframe_points:
            f = int(round(kp.co.x))
            if start <= f <= end:
                kp.interpolation = 'CONSTANT'
        touched += 1
    return touched


# ---------------------------------------------------------------------------
# 场景设置
# ---------------------------------------------------------------------------

class ONTWOS_Settings(PropertyGroup):
    step: IntProperty(
        name="On-N",
        default=2,
        min=1,
        max=12,
        description="Hold a pose every N frames (on-twos=2, on-threes=3)",
    )
    phase: IntProperty(
        name="Phase Offset",
        default=0,
        min=0,
        max=11,
        description="Stepping grid phase (usually 0; adjust to offset from the character's main on-N grid, taken modulo N)",
    )
    frame_start: IntProperty(name="Start Frame", default=0)
    frame_end: IntProperty(name="End Frame", default=250)

    target_mode: EnumProperty(
        name="Target Bones",
        items=[
            ('SELECTED', "Selected Bones", "Only bake the bones selected in Pose Mode (default)"),
            ('RIGID_TRACK', "MMD Rigid-Track Bones", "Only bake bones with 'mmd_tools_rigid_track' constraints (MMD rigs, requires MMD Tools)"),
        ],
        default='SELECTED',
    )
    follow_chain: BoolProperty(
        name="Include Child Chains",
        default=False,
        description="When enabled, selecting a parent bone in Pose Mode automatically includes its entire child chain (e.g. select the hip to grab the whole leg chain)",
    )
    bake_mute_physics: BoolProperty(
        name="Lock after Bake (drivers baked in)",
        default=True,
        description="After baking, mute all constraints on baked bones and disable the rigid body world for a deterministic, re-jump-safe pure keyframe animation; "
                    "the driven motion is already sampled into the keyframes, no active constraints needed (unlock and re-bake anytime)",
    )
    bake_keep_damping: BoolProperty(
        name="Keep Damping as Active Constraint (non-deterministic)",
        default=False,
        description="Default off: driven motion is baked into keyframes, pose is deterministic and jumping back resets. On: adds another active damping layer on top of "
                    "the keyframes (softer / bigger swing, but playback, pause and re-jump are not deterministic and may not return to the start)",
    )
    bake_mute_all: BoolProperty(
        name="Freeze All (mute all constraints)",
        default=False,
        description="Force-mute all constraints on baked bones (same lock as the default, explicit confirmation); for a clean, predictable on-N hold",
    )
    bake_two_pass: BoolProperty(
        name="Two-Pass Bake (damping look + deterministic)",
        default=False,
        description="Two-pass bake: ① write 'driven motion + damping' into keyframes and mute rigid constraints; ② with the rigid body world off, sample and fix "
                    "the 'keyframes + active damping' display pose into keyframes, then fully lock. Keeps the soft look while staying deterministic and "
                    "re-jump-safe (more stable than keeping active damping; divergent frames are clamped and reported)",
    )

    use_mmd_physics: BoolProperty(
        name="MMD Physics Detection",
        default=False,
        description="When enabled, only process curves of bones detected as MMD physics-driven (requires MMD data on the armature / MMD Tools). "
                    "Disabled by default so the tool works on any rig",
    )
    only_dense: BoolProperty(
        name="Dense Baked Curves Only",
        default=True,
        description="Only process curves that look like baked (dense) animation, judged by the curve's own covered range, to avoid touching hand-keyed animation",
    )
    only_selected_bones: BoolProperty(
        name="Selected Bones Only",
        default=False,
        description="In Pose Mode, only process curves of the currently selected bones",
    )
    use_stepped: BoolProperty(
        name="Stepped Mode (like F-Curve editor)",
        default=False,
        description="Use a stepped F-modifier (step size = N) on all target bone channels for on-N holds, matching what you would do by hand with "
                    "'select all channels -> Stepped interpolation -> step size'; not limited by dense/physics filters, all channels step together, "
                    "non-destructive (remove the modifier to restore)",
    )


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def _bake_pose_to_keys(context, arm_obj, bone_names, start, end):
    """逐帧把骨骼当前（约束求值后）的最终姿态写入关键帧。

    用『目标骨架空间矩阵 → matrix_basis』显式反推（父先子后），绕开 pose_bone.matrix
    赋值时读取陈旧父矩阵导致链错乱/拉抻的问题；遇到非有限值钳制为上一有效帧姿态。
    返回被钳制（求值发散）的骨骼帧数。
    """
    ordered = _parent_first(arm_obj, bone_names)
    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = arm_obj
    last_valid = {}
    total_clamped = 0
    try:
        for f in range(start, end + 1):
            context.scene.frame_set(f)
            context.evaluated_depsgraph_get().update()
            mats = {}
            clamped = 0
            for name in ordered:
                mat = arm_obj.pose.bones[name].matrix.copy()
                if not _matrix_finite(mat):
                    mat = last_valid.get(name, arm_obj.pose.bones[name].bone.matrix_local.copy())
                    clamped += 1
                else:
                    last_valid[name] = mat
                mats[name] = mat
            total_clamped += clamped
            for name in ordered:
                pbone = arm_obj.pose.bones[name]
                pbone.matrix_basis = _bone_basis_for(arm_obj, pbone, mats[name], mats)
                pbone.keyframe_insert(data_path="location", frame=f)
                pbone.keyframe_insert(data_path=_rot_data_path(pbone), frame=f)
    finally:
        context.view_layer.objects.active = prev_active
    return total_clamped


def _sample_display(context, arm_obj, bone_names, start, end):
    """第2遍采样：在『关键帧+活动阻尼』（物理世界已关、目标冻结）下逐帧采样最终显示姿态。
    非有限值钳制为上一有效帧姿态。返回 (frames, total_clamped)。"""
    ordered = _parent_first(arm_obj, bone_names)
    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = arm_obj
    last_valid = {}
    total_clamped = 0
    try:
        frames = []
        for f in range(start, end + 1):
            context.scene.frame_set(f)
            context.evaluated_depsgraph_get().update()
            mats = {}
            clamped = 0
            for name in ordered:
                mat = arm_obj.pose.bones[name].matrix.copy()
                if not _matrix_finite(mat):
                    mat = last_valid.get(name, arm_obj.pose.bones[name].bone.matrix_local.copy())
                    clamped += 1
                else:
                    last_valid[name] = mat
                mats[name] = mat
            total_clamped += clamped
            frames.append((f, mats))
        return frames, total_clamped
    finally:
        context.view_layer.objects.active = prev_active


def _write_display_keys(context, arm_obj, bone_names, frames):
    """第2遍写入：约束已全部静默，用显式basis反推把采样到的显示姿态写入关键帧。
    父先子后 + 显式basis → 链一致，锁定后不拉抻。"""
    ordered = _parent_first(arm_obj, bone_names)
    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = arm_obj
    try:
        for f, mats in frames:
            context.scene.frame_set(f)
            for name in ordered:
                pbone = arm_obj.pose.bones[name]
                pbone.matrix_basis = _bone_basis_for(arm_obj, pbone, mats[name], mats)
                pbone.keyframe_insert(data_path="location", frame=f)
                pbone.keyframe_insert(data_path=_rot_data_path(pbone), frame=f)
    finally:
        context.view_layer.objects.active = prev_active


def _set_constraint_mutes(arm_obj, bone_names, mute_all_except=None, mute_names=None):
    """批量设置骨骼约束静默状态。mute_all_except: 除指定名称外全部静默；
    mute_names: 仅静默指定名称的约束。"""
    for pbone in arm_obj.pose.bones:
        if pbone.name not in bone_names:
            continue
        for constr in pbone.constraints:
            if mute_names is not None:
                constr.mute = constr.name in mute_names
            elif mute_all_except is not None:
                constr.mute = constr.name not in mute_all_except


class ONTWOS_OT_bake_physics(Operator):
    bl_idname = "ontwos.bake_physics"
    bl_label = "Bake Poses → Bone Keyframes"
    bl_description = "Write the constraint/physics-evaluated bone pose per frame into bone keyframes, producing dense keyframes ready for decimation (Ctrl+Z undoable)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        s = context.scene.ontwos
        start, end = sorted((s.frame_start, s.frame_end))
        if end < start:
            start, end = end, start

        armatures = [o for o in context.selected_objects if o.type == 'ARMATURE']
        if not armatures:
            self.report({'WARNING'}, "Select an armature object first")
            return {'CANCELLED'}

        rbw = context.scene.rigidbody_world
        total_bones = 0
        total_clamped = 0

        for arm_obj in armatures:
            # 1) 确定目标骨骼（按 target_mode，默认全部骨骼）
            bone_names = _bake_target_bones(arm_obj, context)
            if not bone_names:
                self.report(
                    {'WARNING'},
                    f"{arm_obj.name}: no bones matched the target. Select bones in Pose Mode or change 'Target Bones'",
                )
                continue

            # 2) 确保有动画数据/动作
            adata = arm_obj.animation_data_create()
            if adata.action is None:
                adata.action = bpy.data.actions.new(name=f"{arm_obj.name}_pose_bake")

            # 3) 先恢复驱动（解除全部约束静默 + 启用物理世界），保证采到的是驱动后的姿态
            _set_constraint_mutes(arm_obj, bone_names, mute_names=set())
            if rbw is not None:
                rbw.enabled = True

            if s.bake_two_pass:
                # ---------- 两遍烘焙：保留阻尼观感 + 确定锁定 ----------
                # 第1遍：驱动+阻尼 → 关键帧（显式basis，防拉抻）
                total_clamped += _bake_pose_to_keys(context, arm_obj, bone_names, start, end)
                # 只静默刚体跟踪类约束，保留阻尼活动；关闭物理世界（冻结目标 → 确定、防发散）
                _set_constraint_mutes(arm_obj, bone_names, mute_names={"mmd_tools_rigid_track"})
                if rbw is not None:
                    rbw.enabled = False
                # 第2遍：采样『关键帧+活动阻尼』的显示姿态
                frames, clamped2 = _sample_display(context, arm_obj, bone_names, start, end)
                total_clamped += clamped2
                # 完全锁定：先静默全部约束，再写入（链一致，不拉抻）
                _set_constraint_mutes(arm_obj, bone_names, mute_all_except=set())
                _write_display_keys(context, arm_obj, bone_names, frames)
            else:
                # ---------- 默认：单遍烘焙 ----------
                total_clamped += _bake_pose_to_keys(context, arm_obj, bone_names, start, end)
                # 锁定（可选）：默认静默全部约束（阻尼已烘焙进关键帧，姿态确定可回跳）；
                #    仅当显式勾选『保留阻尼为活动约束』时才只静默刚体跟踪约束（非确定行为）
                if s.bake_mute_all or s.bake_mute_physics:
                    if s.bake_mute_all or not s.bake_keep_damping:
                        _set_constraint_mutes(arm_obj, bone_names, mute_all_except=set())
                    else:
                        _set_constraint_mutes(arm_obj, bone_names, mute_names={"mmd_tools_rigid_track"})
                    if rbw is not None:
                        rbw.enabled = False

            total_bones += len(bone_names)

        if total_bones == 0:
            self.report({'WARNING'}, "No bones to bake")
            return {'CANCELLED'}

        if s.bake_two_pass:
            msg = f"Two-pass bake and full lock: frames {start}~{end}, {total_bones} bones (soft look kept, deterministic and re-jump-safe)"
        else:
            msg = f"Baked poses to keyframes: frames {start}~{end}, {total_bones} bones"
            if s.bake_mute_physics:
                if s.bake_mute_all or not s.bake_keep_damping:
                    msg += " (fully locked incl. damping, deterministic and re-jump-safe)"
                else:
                    msg += " (drivers locked, active damping kept - non-deterministic)"
        if total_clamped:
            msg += f"; {total_clamped} divergent evaluations clamped to the previous pose"
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class ONTWOS_OT_unlock_physics(Operator):
    bl_idname = "ontwos.unlock_physics"
    bl_label = "Unlock Physics"
    bl_description = "Unmute all muted constraints on the selected armatures and re-enable the rigid body world, removing baked bone keyframes to fully return to driven motion (Ctrl+Z undoable)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        rbw = context.scene.rigidbody_world
        s = context.scene.ontwos
        start, end = sorted((s.frame_start, s.frame_end))
        n = 0
        keys = 0
        for obj in context.selected_objects:
            if obj.type != 'ARMATURE':
                continue
            for pbone in obj.pose.bones:
                for constr in pbone.constraints:
                    if constr.mute:
                        constr.mute = False
                        n += 1
            bone_names = _bake_target_bones(obj, context)
            if bone_names:
                keys += _remove_baked_keys(obj, bone_names, start, end)
        if rbw is not None:
            rbw.enabled = True
        msg = f"Unmuted {n} constraints, rigid body world enabled"
        if keys:
            msg += f", removed {keys} baked keyframes"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class ONTWOS_OT_preview_target(Operator):
    bl_idname = "ontwos.preview_target"
    bl_label = "Highlight Target Bones"
    bl_description = "Select (highlight) in Pose Mode the bones the current target setting would affect, without modifying anything"

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            armatures = [o for o in context.selected_objects if o.type == 'ARMATURE']
            if not armatures:
                self.report({'WARNING'}, "Select an armature object first")
                return {'CANCELLED'}
            arm_obj = armatures[0]

        bone_names = _bake_target_bones(arm_obj, context)
        if not bone_names:
            self.report(
                {'WARNING'},
                "No bones matched the target: select bones in Pose Mode (or enable 'Include Child Chains' to grab whole chains) "
                "or switch 'Target Bones' to 'MMD Rigid-Track Bones'",
            )
            return {'CANCELLED'}

        current_mode = context.mode
        try:
            if current_mode != 'POSE':
                bpy.ops.object.mode_set(mode='POSE')
            bpy.ops.pose.select_all(action='DESELECT')
            for pb in arm_obj.pose.bones:
                # Blender 5.0 起选择状态存放在 PoseBone.select；旧版本（4.x 及更早）存放在 Bone.select，需兼容处理
                if hasattr(pb, 'select'):
                    pb.select = pb.name in bone_names
                else:
                    pb.bone.select = pb.name in bone_names
        finally:
            if current_mode != 'POSE':
                bpy.ops.object.mode_set(mode=current_mode)

        self.report(
            {'INFO'},
            f"Highlighted {len(bone_names)} target bone(s) in Pose Mode (nothing modified)",
        )
        return {'FINISHED'}


class ONTWOS_OT_preview(Operator):
    bl_idname = "ontwos.preview"
    bl_label = "Preview Stats (no changes)"
    bl_description = "Count the keyframes that would be removed and the curves that would be affected, without modifying anything"

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        s = context.scene.ontwos
        if s.use_stepped:
            n = len(_stepped_fcurves(context))
            if n == 0:
                self.report({'WARNING'}, "No matching bone channel curves (select an armature with an action first)")
                return {'CANCELLED'}
            self.report(
                {'INFO'},
                f"Stepped mode: will add an on-{s.step} step to {n} bone channel curves (non-destructive, remove modifiers to restore; nothing modified)",
            )
            return {'FINISHED'}
        s, step, phase, start, end, sel_bones = _resolve_filters(context)
        targets, sk_b, sk_m, sk_d = _collect_targets(
            context, s.only_dense, s.use_mmd_physics, sel_bones, start, end
        )
        curves_hit = 0
        keys_removable = 0
        for _obj, fcurve in targets:
            n = _count_removable(fcurve, start, end, step, phase)
            if n:
                curves_hit += 1
                keys_removable += n
        msg = f"Matched {curves_hit} curves, would remove {keys_removable} keyframes"
        skipped = _skip_summary(sk_b, sk_m, sk_d)
        if skipped:
            msg += f" (skipped: {skipped})"
        self.report({'INFO'}, msg + " (nothing modified)")
        return {'FINISHED'}


class ONTWOS_OT_apply(Operator):
    bl_idname = "ontwos.apply"
    bl_label = "Apply On-N (decimate + hold)"
    bl_description = "Decimate the baked keyframes of the selected objects/bones and set constant interpolation for the on-N hold (Ctrl+Z undoable)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        # 先为所有选中物体备份当前动作曲线（『删除一拍N』时据此恢复原始动画）
        for obj in context.selected_objects:
            adata = obj.animation_data
            if adata and adata.action:
                _backup_action(adata.action)
        s = context.scene.ontwos
        if s.use_stepped:
            # ---------- 步进插值模式（同曲线编辑器手动）：非破坏，全部骨骼通道统一步进 ----------
            step = max(1, s.step)
            phase = s.phase % step
            start, end = sorted((s.frame_start, s.frame_end))
            touched = _apply_stepped(context, s, step, phase, start, end)
            if touched == 0:
                self.report({'WARNING'}, "No matching bone channel curves (select an armature with an action first)")
                return {'CANCELLED'}
            self.report(
                {'INFO'},
                f"Stepped interpolation applied: on-{step} hold on {touched} bone channel curves (non-destructive, remove modifiers to restore)",
            )
            return {'FINISHED'}
        s, step, phase, start, end, sel_bones = _resolve_filters(context)
        targets, sk_b, sk_m, sk_d = _collect_targets(
            context, s.only_dense, s.use_mmd_physics, sel_bones, start, end
        )
        curves_touched = 0
        keys_removed = 0
        for _obj, fcurve in targets:
            removed, _kept = _decimate_fcurve(fcurve, start, end, step, phase)
            if removed:
                curves_touched += 1
                keys_removed += removed

        if curves_touched == 0:
            skipped = _skip_summary(sk_b, sk_m, sk_d)
            self.report(
                {'WARNING'},
                "No processable curves matched (check: armature selected, range correct, "
                + ("MMD physics detection enabled" if s.use_mmd_physics else "'Dense Baked Curves Only' switch")
                + (f"; skipped: {skipped}" if skipped else "") + ")",
            )
            return {'CANCELLED'}

        msg = f"Processed {curves_touched} curves, removed {keys_removed} keyframes (on-{step})"
        skipped = _skip_summary(sk_b, sk_m, sk_d)
        if skipped:
            msg += f" (skipped: {skipped})"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class ONTWOS_OT_remove_onetwos(Operator):
    bl_idname = "ontwos.remove_onetwos"
    bl_label = "Remove On-N (restore)"
    bl_description = (
        "Remove the applied on-N effect: first tries exact restore from the backup taken before 'Apply On-N' "
        "(backup is saved with the file and survives restarts); without a backup, falls back to a generic "
        "method - remove stepped modifiers and smooth constant keys in range (target bones only, never hand-keyed bones)."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        restored = 0
        smoothed = 0
        for obj in context.selected_objects:
            adata = obj.animation_data
            if not adata or not adata.action:
                continue
            backup_name = f"__onetwos_backup_{adata.action.name}"
            bak = bpy.data.actions.get(backup_name)
            if bak is not None:
                # ---------- 有备份：精确恢复原始曲线 ----------
                for fc in adata.action.fcurves:
                    for mod in list(fc.modifiers):
                        if mod.type == 'STEPPED':
                            fc.modifiers.remove(mod)
                for fc in list(adata.action.fcurves):
                    adata.action.fcurves.remove(fc)
                for bfc in bak.fcurves:
                    nfc = adata.action.fcurves.new(bfc.data_path, index=bfc.array_index)
                    _copy_fcurve(bfc, nfc)
                bpy.data.actions.remove(bak)
                restored += 1
            else:
                # ---------- 无备份：通用兜底（重开Blender后的旧文件等场景） ----------
                smoothed += _smooth_onetwos_fallbacks(
                    context, adata.action, _fallback_target_fcurves(context)
                )

        if restored == 0 and smoothed == 0:
            self.report(
                {'WARNING'},
                "No on-N effect found to remove (confirm an armature with an action is selected and the range is correct; "
                "without a backup there should at least be stepped modifiers or constant keys in range)",
            )
            return {'CANCELLED'}

        parts = []
        if restored:
            parts.append(f"exact restore of {restored} action(s)")
        if smoothed:
            parts.append(f"generic removal: smoothed {smoothed} target curve(s) (no backup)")
        self.report({'INFO'}, "On-N effect removed: " + "; ".join(parts))
        return {'FINISHED'}


class ONTWOS_OT_set_range(Operator):
    bl_idname = "ontwos.set_range"
    bl_label = "Use Scene Frame Range"
    bl_description = "Set the processing range to the current scene frame range"

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        context.scene.ontwos.frame_start = context.scene.frame_start
        context.scene.ontwos.frame_end = context.scene.frame_end
        self.report(
            {'INFO'},
            f"Range set to [{context.scene.frame_start}, {context.scene.frame_end}]",
        )
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# 面板
# ---------------------------------------------------------------------------

class ONTWOS_PT_panel(Panel):
    bl_label = "On-Twos Retimer"
    bl_idname = "ONTWOS_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "On-Twos"

    def draw(self, context):
        s = context.scene.ontwos
        layout = self.layout

        # 公共：范围
        box = layout.box()
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(s, "frame_start")
        row.prop(s, "frame_end")
        col.operator("ontwos.set_range", icon="VIEWZOOM")

        # 第一步：驱动姿态 → 骨骼关键帧
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Step 1 · Bake Poses → Bone Keyframes", icon="KEYFRAME_HLT")
        col.prop(s, "target_mode")
        col.prop(s, "follow_chain")
        col.operator("ontwos.preview_target", icon="RESTRICT_SELECT_OFF")
        col.prop(s, "bake_two_pass")
        col.prop(s, "bake_mute_physics")
        col.prop(s, "bake_keep_damping")
        col.prop(s, "bake_mute_all")
        col.operator("ontwos.bake_physics", icon="KEYFRAME_HLT")
        col.operator("ontwos.unlock_physics", icon="LOOP_BACK")
        tip = box.column(align=True)
        tip.scale_y = 0.85
        tip.label(text="Tip: 'Two-Pass Bake' keeps the soft damping look and stays", icon="INFO")
        tip.label(text="deterministic (recommended, no jump-back reset issue); the")
        tip.label(text="default bake writes driven motion + damping into keyframes")
        tip.label(text="and fully locks; tick 'Keep Damping' only for an extra active")
        tip.label(text="damping layer (non-deterministic). 'Include Child Chains'")
        tip.label(text="grabs whole chains from a parent bone; the highlight button")
        tip.label(text="shows exactly which bones would be baked. 'MMD Rigid-Track")
        tip.label(text="Bones' is for MMD rigs (needs MMD Tools).")

        # 第二步：一拍N抽稀
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Step 2 · On-N Hold", icon="SNAP_ON")
        col.prop(s, "step")
        col.prop(s, "phase")
        col.prop(s, "use_stepped")
        col.prop(s, "use_mmd_physics")
        col.prop(s, "only_dense")
        col.prop(s, "only_selected_bones")
        col.operator("ontwos.preview", icon="INFO")
        col.operator("ontwos.apply", icon="SNAP_ON")
        col.operator("ontwos.remove_onetwos", icon="LOOP_BACK")
        tip = box.column(align=True)
        tip.scale_y = 0.85
        tip.label(text="Tip: 'Remove On-N' removes the applied effect: exact restore from", icon="INFO")
        tip.label(text="backup when available (backup saves with the file and survives")
        tip.label(text="restarts); otherwise generic removal - remove stepped modifiers,")
        tip.label(text="smooth constant keys, target bones only. 'Stepped Mode'")
        tip.label(text="matches doing it by hand (select all channels -> stepped -> size N)")
        tip.label(text="and only touches target bones (never hand-keyed ones).")

        layout.separator()
        info = layout.box()
        info_col = info.column(align=True)
        info_col.scale_y = 0.85
        info_col.label(text="Flow: select armature -> 1. bake poses -> keyframes", icon="INFO")
        info_col.label(text="-> 2. apply on-N. Mistakes: Ctrl+Z; to redo step 1,")
        info_col.label(text="'Unlock Physics' first, then re-bake.")


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

classes = (
    ONTWOS_Settings,
    ONTWOS_OT_set_range,
    ONTWOS_OT_bake_physics,
    ONTWOS_OT_unlock_physics,
    ONTWOS_OT_preview_target,
    ONTWOS_OT_preview,
    ONTWOS_OT_apply,
    ONTWOS_OT_remove_onetwos,
    ONTWOS_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ontwos = PointerProperty(type=ONTWOS_Settings)


def unregister():
    del bpy.types.Scene.ontwos
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
