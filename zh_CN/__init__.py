# -*- coding: utf-8 -*-
"""
On-Twos 一拍二重定时插件（通用版）
=================================

被【约束/物理】驱动的骨骼运动（IK、阻尼追踪、刚体模拟、布料/头发链条等）在播放时
实时求值，并不产生关键帧。本插件两步闭环：

    ① 烘焙姿态→骨骼关键帧：逐帧采样『约束求值后』的最终姿态并写入骨骼关键帧
      （可选锁定驱动设置）
    ② 应用一拍N：对关键帧抽稀(保留满足 (帧-相位)%N==0 的 key)+恒定插值，实现定格效果

适用于任何『骨骼被约束/物理驱动』的骨架。可选兼容模式可把烘焙限制到 MMD 物理骨
（带 mmd_tools_rigid_track 约束的骨骼），用于 MMD 模型。

其他说明
--------
- ①②均带 UNDO，可 Ctrl+Z 撤销；①非破坏（锁定可恢复：解除约束静默+启用物理世界）。
- 兼容 Blender 4.x / 5.0。

用法
----
1. 选中骨架（骨骼被约束/物理驱动）；
2. 3D 视图右侧 N 面板「一拍二」标签页：
   a. 设好范围与目标骨骼 → 点【烘焙姿态→骨骼关键帧】；
   b. 点【预览统计】确认影响范围 → 点【应用一拍N】。
3. 效果不满意：Ctrl+Z；或解锁（面板按钮）重新烘焙。
"""

bl_info = {
    "name": "On-Twos 一拍二重定时（通用版）",
    "author": "zlz",
    "version": (0, 7, 0),
    "blender": (4, 0, 0),
    "location": "3D 视图 > 侧边栏 > 一拍二",
    "description": "把约束/物理驱动的骨骼姿态烘焙成关键帧，再抽稀+恒定插值实现一拍N定格效果（一拍二/一拍三）。",
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
        parts.append(f"非物理识别骨 {skipped_mmd}")
    if skipped_dense:
        parts.append(f"非密集 {skipped_dense}")
    if skipped_bone:
        parts.append(f"非选中骨 {skipped_bone}")
    return "；".join(parts) if parts else ""


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
        name="一拍 N",
        default=2,
        min=1,
        max=12,
        description="每 N 帧保持一个姿势（一拍二=2，一拍三=3）",
    )
    phase: IntProperty(
        name="相位偏移",
        default=0,
        min=0,
        max=11,
        description="步进网格相位（通常 0；与角色主体一拍网格错开时调整，实际按 N 取模）",
    )
    frame_start: IntProperty(name="起始帧", default=0)
    frame_end: IntProperty(name="结束帧", default=250)

    target_mode: EnumProperty(
        name="目标骨骼",
        items=[
            ('SELECTED', "选中骨骼", "只烘焙姿态模式下选中的骨骼（默认）"),
            ('RIGID_TRACK', "MMD 刚体跟踪骨", "只烘焙带 mmd_tools_rigid_track 约束的骨骼（MMD 模型，需 MMD Tools）"),
        ],
        default='SELECTED',
    )
    follow_chain: BoolProperty(
        name="包含子链",
        default=False,
        description="勾选后，姿态模式下选中父级骨骼会自动包含整条子链（如选中骨盆即可覆盖整条腿链）",
    )
    bake_mute_physics: BoolProperty(
        name="烘焙后锁定（驱动已含在关键帧）",
        default=True,
        description="烘焙完成后静默烘焙骨上的全部约束并关闭物理世界，得到确定、可回跳的纯关键帧动画；"
                    "驱动效果已在采样时写入关键帧，无需保留活动约束（可随时恢复重新烘焙）",
    )
    bake_keep_damping: BoolProperty(
        name="同时保留阻尼为活动约束（非确定）",
        default=False,
        description="默认关闭：驱动已烘焙进关键帧，姿态确定、跳回开头会重置。打开则以活动约束形式在关键帧上再叠加一层"
                    "阻尼（更柔顺/摆幅更大，但播放、暂停、回跳不具确定性，可能出现不回到开头的情况）",
    )
    bake_mute_all: BoolProperty(
        name="完全冻结（静默所有约束）",
        default=False,
        description="强制静默烘焙骨上的全部约束（与默认锁定位相同，仅作显式确认）；适合需要纯净、可预测的一拍N定格",
    )
    bake_two_pass: BoolProperty(
        name="两遍烘焙（保留阻尼观感+确定）",
        default=False,
        description="两遍烘焙：①把『驱动+阻尼』写入关键帧并静默刚体跟踪约束；②关闭物理世界后把『关键帧+活动阻尼』的"
                    "显示姿态采样固化进关键帧，随后完全锁定。既保留阻尼的柔顺/摆幅观感，又是确定、可回跳的"
                    "纯关键帧（比『保留活动阻尼』更稳，无跳回不重置问题；个别帧求值发散会自动钳制并提示）",
    )

    use_mmd_physics: BoolProperty(
        name="MMD 物理骨识别",
        default=False,
        description="启用后只处理被识别为 MMD 物理驱动的骨骼曲线（骨架需带 MMD 数据 / MMD Tools）。"
                    "默认关闭，保证插件在任意骨架上都能用",
    )
    only_dense: BoolProperty(
        name="只处理密集烘焙曲线",
        default=True,
        description="按曲线自身覆盖范围密度判断是否为烘焙（密集）动画，避免误删手 K 动画",
    )
    only_selected_bones: BoolProperty(
        name="仅处理选中骨骼",
        default=False,
        description="姿态模式下只处理当前选中的骨骼曲线",
    )
    use_stepped: BoolProperty(
        name="步进插值模式（同曲线编辑器手动）",
        default=False,
        description="用步进F曲线修改器(step_size=N)对目标骨骼通道做一拍N定格，与你手动"
                    "『全选所有通道→步进插值→步长尺寸』一致；不受密集/物理骨筛选限制，所有目标通道统一步进，"
                    "非破坏性（可删修改器还原）",
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
    bl_label = "烘焙姿态→骨骼关键帧"
    bl_description = "把约束/物理求值后的骨骼姿态逐帧写入骨骼关键帧，得到可抽稀的密集关键帧（可 Ctrl+Z 撤销）"
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
            self.report({'WARNING'}, "请先选中骨架对象")
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
                    f"{arm_obj.name} 没有匹配到目标骨骼：请在姿态模式下选中骨骼，或修改『目标骨骼』选项",
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
            self.report({'WARNING'}, "没有可烘焙的骨骼")
            return {'CANCELLED'}

        if s.bake_two_pass:
            msg = f"两遍烘焙并完全锁定：{start}~{end} 帧、{total_bones} 个骨骼（保留阻尼观感，姿态确定可回跳）"
        else:
            msg = f"已把驱动姿态烘焙为关键帧：{start}~{end} 帧、{total_bones} 个骨骼"
            if s.bake_mute_physics:
                if s.bake_mute_all or not s.bake_keep_damping:
                    msg += "，已完全锁定（含阻尼，姿态确定可回跳）"
                else:
                    msg += "，驱动已锁定但保留活动阻尼（非确定）"
        if total_clamped:
            msg += f"；{total_clamped} 处求值发散已按上一姿态钳制"
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class ONTWOS_OT_unlock_physics(Operator):
    bl_idname = "ontwos.unlock_physics"
    bl_label = "恢复驱动（解锁）"
    bl_description = "解除选中骨架骨骼上被静默的全部约束并重新启用物理世界，同时清除烘焙生成的骨骼关键帧，完全回到驱动状态（可 Ctrl+Z 撤销）"
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
        msg = f"已解除 {n} 个被静默的约束，物理世界已启用"
        if keys:
            msg += f"，清除烘焙关键帧 {keys} 个"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class ONTWOS_OT_preview_target(Operator):
    bl_idname = "ontwos.preview_target"
    bl_label = "高亮目标骨骼"
    bl_description = "在姿态模式下选中（高亮）当前目标设置将命中的骨骼，不做任何修改"

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            armatures = [o for o in context.selected_objects if o.type == 'ARMATURE']
            if not armatures:
                self.report({'WARNING'}, "请先选中骨架对象")
                return {'CANCELLED'}
            arm_obj = armatures[0]

        bone_names = _bake_target_bones(arm_obj, context)
        if not bone_names:
            self.report(
                {'WARNING'},
                "没有匹配到目标骨骼：请在姿态模式下选中骨骼（或勾选『包含子链』以覆盖整条链），"
                "或将『目标骨骼』切换为『MMD 刚体跟踪骨』",
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
            f"已在姿态模式高亮 {len(bone_names)} 个目标骨骼（未做任何修改）",
        )
        return {'FINISHED'}


class ONTWOS_OT_preview(Operator):
    bl_idname = "ontwos.preview"
    bl_label = "预览统计（不修改）"
    bl_description = "无损统计将被抽稀的关键帧数量与命中曲线数"

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        s = context.scene.ontwos
        if s.use_stepped:
            n = len(_stepped_fcurves(context))
            if n == 0:
                self.report({'WARNING'}, "没有命中的骨骼通道曲线（请先选中含动作的骨架）")
                return {'CANCELLED'}
            self.report(
                {'INFO'},
                f"步进插值模式：将给 {n} 条骨骼通道曲线加一拍{s.step}步进（非破坏，可删修改器还原；未做任何修改）",
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
        msg = f"命中 {curves_hit} 条曲线，将删除 {keys_removable} 个关键帧"
        skipped = _skip_summary(sk_b, sk_m, sk_d)
        if skipped:
            msg += f"（跳过 {skipped}）"
        self.report({'INFO'}, msg + "（未做任何修改）")
        return {'FINISHED'}


class ONTWOS_OT_apply(Operator):
    bl_idname = "ontwos.apply"
    bl_label = "应用一拍N（抽稀+恒定）"
    bl_description = "对选中物体/骨骼的烘焙关键帧抽稀并改为恒定插值，实现一拍N定格效果（可 Ctrl+Z 撤销）"
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
            # ---------- 步进插值模式（同曲线编辑器手动）：非破坏，目标骨骼通道统一步进 ----------
            step = max(1, s.step)
            phase = s.phase % step
            start, end = sorted((s.frame_start, s.frame_end))
            touched = _apply_stepped(context, s, step, phase, start, end)
            if touched == 0:
                self.report({'WARNING'}, "没有命中的骨骼通道曲线（请先选中含动作的骨架）")
                return {'CANCELLED'}
            self.report(
                {'INFO'},
                f"步进插值已应用：{touched} 条骨骼通道曲线 一拍{step} 定格（非破坏，可删修改器还原）",
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
                "没有命中任何可处理的曲线（请检查：选中的是骨架对象、范围正确、"
                + ("MMD物理骨识别是否识别到" if s.use_mmd_physics else "『只处理密集』开关")
                + (f"；跳过 {skipped}" if skipped else "") + "）",
            )
            return {'CANCELLED'}

        msg = f"已处理 {curves_touched} 条曲线，删除 {keys_removed} 个关键帧（一拍{step}）"
        skipped = _skip_summary(sk_b, sk_m, sk_d)
        if skipped:
            msg += f"（跳过 {skipped}）"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class ONTWOS_OT_remove_onetwos(Operator):
    bl_idname = "ontwos.remove_onetwos"
    bl_label = "删除一拍N效果（恢复）"
    bl_description = (
        "删除已添加的一拍N效果：优先用『应用一拍N』前备份的原始曲线精确恢复"
        "（备份随文件保存，重开 Blender 也有效）；若无备份则用通用方式——移除步进插值修改器、"
        "把范围内恒定插值改回平滑插值（只作用于目标骨骼，不碰手K骨骼）。"
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
                "未找到可恢复的一拍N效果（请确认已选中含动作的骨架、范围正确；"
                "若无备份则至少应有步进修改器或范围内的恒定插值曲线）",
            )
            return {'CANCELLED'}

        parts = []
        if restored:
            parts.append(f"精确恢复 {restored} 个动作")
        if smoothed:
            parts.append(f"通用去除：平滑 {smoothed} 条目标曲线（无备份）")
        self.report({'INFO'}, "已删除一拍N效果，" + "；".join(parts))
        return {'FINISHED'}


class ONTWOS_OT_set_range(Operator):
    bl_idname = "ontwos.set_range"
    bl_label = "范围取自场景帧区间"
    bl_description = "把处理范围设置为当前场景的帧起止"

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        context.scene.ontwos.frame_start = context.scene.frame_start
        context.scene.ontwos.frame_end = context.scene.frame_end
        self.report(
            {'INFO'},
            f"范围已设为 [{context.scene.frame_start}, {context.scene.frame_end}]",
        )
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# 面板
# ---------------------------------------------------------------------------

class ONTWOS_PT_panel(Panel):
    bl_label = "一拍二重定时（通用版）"
    bl_idname = "ONTWOS_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "一拍二"

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
        col.label(text="第一步 · 烘焙姿态→骨骼关键帧", icon="KEYFRAME_HLT")
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
        tip.label(text="提示：勾选『两遍烘焙』= 保留阻尼观感且姿态确定", icon="INFO")
        tip.label(text="（推荐，无跳回不重置问题）；默认烘焙=驱动+阻尼写入关键帧")
        tip.label(text="并完全锁定；想再叠加一层活阻尼才勾选『保留阻尼』(非确定)。")
        tip.label(text="勾选『包含子链』= 选中父骨自动覆盖整条子链；『高亮目标")
        tip.label(text="骨骼』可预览将被烘焙的骨骼。『MMD 刚体跟踪骨』仅用于")
        tip.label(text="MMD 模型（需 MMD Tools）。")

        # 第二步：一拍N抽稀
        box = layout.box()
        col = box.column(align=True)
        col.label(text="第二步 · 一拍N定格", icon="SNAP_ON")
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
        tip.label(text="提示：『删除一拍N』= 删除已添加的一拍N效果：优先用备份精确恢复", icon="INFO")
        tip.label(text="（备份随文件保存，重开Blender也有效）；无备份时通用去除——移")
        tip.label(text="除步进修改器、把范围内恒定插值改回平滑，且只作用目标骨骼。")
        tip.label(text="勾选『步进插值模式』= 与你手动全选通道→步进插值→步长尺寸 一致，")
        tip.label(text="且只作用于目标骨骼（不碰躯干等手K骨骼），可删修改器还原。")

        layout.separator()
        info = layout.box()
        info_col = info.column(align=True)
        info_col.scale_y = 0.85
        info_col.label(text="流程：选中骨架 → ①烘焙姿态→关键帧", icon="INFO")
        info_col.label(text="→ ②应用一拍N。误操作可 Ctrl+Z；")
        info_col.label(text="想重做①：先『恢复驱动』再重新烘焙。")


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
