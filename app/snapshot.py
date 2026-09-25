"""已保存方案物理快照解析器(只读, 不改变求解与保存格式)。

用途
----
失效槽位应急迁移(app/evacuation.py)、库存配重补偿(app/counterweight.py)
与备品试管替换(app/spare_replacement.py)三个只读规划器, 都需要从已保存
方案的结果中恢复同一组物理事实: 转子半径、逐槽角度与启用状态、槽位占用
关系以及已放置试管的编号/质量/落槽。本模块把这份**物理快照**的解析与
交叉核对集中到一处, 三个规划器共用同一份校验过的快照与同一个质径积
向量计算, 避免各自重复解析、口径不一。

快照来源(全部为方案结果 result 中的只读数据)
------------------------------------------
- ``config.radius_mm``: 转子半径 R(mm); 缺省时回退 ``request.radius_mm``,
  再缺省取 100.0(与三个规划器原有口径一致);
- ``slot_state``: 逐槽 ``{slot, angle, enabled, tube_id, locked}``;
- ``placements``: 逐管 ``{tube_id, mass, locked, slot, angle}``
  (``slot`` 为空表示该管未放置, 不进入已放置试管列表)。

交叉核对(任一不满足即抛 ValueError, API 层转为 400)
--------------------------------------------------
- **槽号**: slot_state 必须非空, 槽号为不重复的整数且恰好覆盖
  0..N-1(重复/缺失/越界一律拒绝);
- **管号**: 摆放明细中的试管编号不得重复(含未放置试管);
- **有限性**: 半径、槽角、标称质量与摆放角度都必须是有限数值,
  标称质量必须为正;
- **占用关系互相一致**(矛盾即拒绝):
  * 已放置试管的槽位必须存在且为启用槽, slot_state 中该槽登记的
    占用管号必须正是该管(锁定标记也须一致);
  * slot_state 中登记有管的槽位, 必须在摆放明细中找到该管落在该槽;
  * 摆放明细中的槽角与 slot_state 中该槽角度必须一致(1e-9 相对容差);
  * 同一槽位不得被两支试管占用。

快照无效时, 三个规划器在任何搜索之前即抛出 ValueError(HTTP 400),
不会基于损坏或互相矛盾的数据生成达标建议。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# 槽角一致性核对的相对容差(与 app/evacuation.py 等精确模块一致)
_ANGLE_TOL = 1e-9


# ---------------------------------------------------------------------------
# 共用质径积向量计算
# ---------------------------------------------------------------------------
def mr_vector(mass_g: float, radius_mm: float, angle_deg: float) -> complex:
    """质径积向量 R·m·e^{iθ}(g·mm): 质量 mass_g 置于半径 radius_mm、
    角度 angle_deg(度)处的一阶不平衡量贡献。"""
    a = math.radians(angle_deg)
    m = mass_g * radius_mm
    return complex(m * math.cos(a), m * math.sin(a))


def mr_block(x: float, y: float) -> dict[str, Any]:
    """质径积向量 (x, y) 的展示块(模长与方向角), 三个规划器共用。"""
    return {
        "u1_x_g_mm": x,
        "u1_y_g_mm": y,
        "u1_mag_g_mm": math.hypot(x, y),
        "u1_angle_deg": math.degrees(math.atan2(y, x)) % 360.0,
    }


# ---------------------------------------------------------------------------
# 快照数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotSnapshot:
    """单个物理槽的快照: 槽角、启用状态与占用。"""
    slot: int
    angle_deg: float
    enabled: bool
    tube_id: Optional[int]   # 占用该槽的试管编号; 空槽为 None
    locked: bool             # 占用管是否为锁定试管(空槽为 False)


@dataclass(frozen=True)
class TubeSnapshot:
    """一支已放置试管的快照: 编号、标称质量、落槽与槽角。"""
    tube_id: int
    mass_g: float
    slot: int
    angle_deg: float
    locked: bool


@dataclass(frozen=True)
class PlanSnapshot:
    """已保存方案的物理快照(只读): 半径、槽位与已放置试管。"""
    plan_id: Any
    radius_mm: float
    slots: tuple[SlotSnapshot, ...]        # 全部物理槽, 按槽号升序
    tubes: tuple[TubeSnapshot, ...]        # 已放置试管, 摆放明细原顺序
    unplaced_tube_ids: tuple[int, ...]     # 未放置试管编号, 升序
    slot_map: dict[int, SlotSnapshot]      # 槽号 → 槽快照
    tube_map: dict[int, TubeSnapshot]      # 管号 → 已放置试管快照

    @property
    def enabled_count(self) -> int:
        return sum(1 for s in self.slots if s.enabled)

    @property
    def free_slots(self) -> tuple[SlotSnapshot, ...]:
        """空闲启用槽(启用且未被占用), 按槽号升序。"""
        return tuple(s for s in self.slots if s.enabled and s.tube_id is None)


# ---------------------------------------------------------------------------
# 解析与交叉核对
# ---------------------------------------------------------------------------
def _as_int(raw: Any, what: str) -> int:
    """快照中的整数编号: 拒绝布尔、非数值与非整数值(如 3.5)。"""
    if isinstance(raw, bool):
        raise ValueError(f"{what}必须为整数编号(当前 {raw!r})。")
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}必须为整数编号(当前 {raw!r})。") from exc
    if isinstance(raw, float) and val != raw:
        raise ValueError(f"{what}必须为整数编号(当前 {raw!r})。")
    return val


def _as_finite_float(raw: Any, what: str) -> float:
    """快照中的有限数值: 拒绝非数值与 NaN/Inf。"""
    if isinstance(raw, bool):
        raise ValueError(f"{what}必须为有限数值(当前 {raw!r})。")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}必须为有限数值(当前 {raw!r})。") from exc
    if not math.isfinite(val):
        raise ValueError(f"{what}必须为有限数值(当前 {raw!r})。")
    return val


def parse_plan_snapshot(plan: dict[str, Any]) -> PlanSnapshot:
    """解析并交叉核对已保存方案的物理快照(只读, 不修改入参)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result)。

    返回
    ----
    PlanSnapshot: 半径、逐槽角度/启用/占用与已放置试管(编号/质量/落槽)。

    异常
    ----
    ValueError: 快照数据重复、缺失、非有限或互相矛盾(槽号/管号/占用
        关系任一核对不通过); API 层转为 400, 三个规划器在搜索前拒绝。
    """
    plan_id = plan.get("id") if isinstance(plan, dict) else None
    result = plan.get("result") if isinstance(plan, dict) else None
    if not isinstance(result, dict):
        raise ValueError(f"方案 {plan_id} 缺少求解结果, 无法建立物理快照。")

    # ---- 半径(config 优先, 回退 request, 再缺省 100.0) ----
    cfg = result.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
    radius = cfg.get("radius_mm")
    if radius is None:
        request = plan.get("request")
        if not isinstance(request, dict):
            request = {}
        radius = request.get("radius_mm", 100.0)
    radius = _as_finite_float(radius, f"方案 {plan_id} 的转子半径")
    if radius <= 0.0:
        raise ValueError("方案中的转子半径必须为正数。")

    # ---- 槽位状态: 槽角、启用与占用 ----
    slot_state = result.get("slot_state")
    if not isinstance(slot_state, list) or not slot_state:
        raise ValueError(
            f"方案 {plan_id} 缺少槽位状态信息, 无法建立物理快照。"
        )
    slot_map: dict[int, SlotSnapshot] = {}
    for item in slot_state:
        if not isinstance(item, dict):
            raise ValueError(
                f"方案 {plan_id} 的槽位状态格式非法(每项必须为对象)。"
            )
        idx = _as_int(item.get("slot"), f"方案 {plan_id} 槽位状态中的槽号")
        if idx in slot_map:
            raise ValueError(f"槽位状态中槽位 {idx} 重复。")
        angle = _as_finite_float(item.get("angle"), f"槽位 {idx} 的角度")
        occupant = item.get("tube_id")
        if occupant is not None:
            occupant = _as_int(occupant, f"槽位 {idx} 的占用试管编号")
        slot_map[idx] = SlotSnapshot(
            slot=idx,
            angle_deg=angle,
            enabled=bool(item.get("enabled", True)),
            tube_id=occupant,
            locked=bool(item.get("locked", False)),
        )
    # 槽号必须恰好覆盖 0..N-1(与求解器规范化后的几何一致): 缺失/越界拒绝
    expected = set(range(len(slot_map)))
    if set(slot_map) != expected:
        missing = sorted(expected - set(slot_map))
        extra = sorted(set(slot_map) - expected)
        parts = []
        if missing:
            parts.append(f"缺失槽号: {', '.join(map(str, missing))}")
        if extra:
            parts.append(f"越界槽号: {', '.join(map(str, extra))}")
        raise ValueError(
            f"方案 {plan_id} 的槽位状态必须恰好覆盖 0~{len(slot_map) - 1}"
            f"({'; '.join(parts)})。"
        )
    slots = tuple(slot_map[i] for i in sorted(slot_map))

    # ---- 摆放明细: 已放置试管 ----
    raw_placements = result.get("placements")
    if raw_placements is None:
        raw_placements = []
    if not isinstance(raw_placements, list):
        raise ValueError(f"方案 {plan_id} 的摆放明细格式非法。")
    tubes: list[TubeSnapshot] = []
    tube_map: dict[int, TubeSnapshot] = {}
    unplaced: list[int] = []
    seen_ids: set[int] = set()
    seen_slots: set[int] = set()
    for p in raw_placements:
        if not isinstance(p, dict):
            raise ValueError(
                f"方案 {plan_id} 的摆放明细格式非法(每项必须为对象)。"
            )
        tid = _as_int(p.get("tube_id"), "摆放明细中的试管编号")
        if tid in seen_ids:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        seen_ids.add(tid)
        if p.get("slot") is None:
            unplaced.append(tid)
            continue
        slot = _as_int(p.get("slot"), f"试管 {tid} 的槽位")
        if slot in seen_slots:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        seen_slots.add(slot)
        info = slot_map.get(slot)
        if info is None:
            raise ValueError(
                f"试管 {tid} 的槽位 {slot} 在方案槽位状态中不存在。"
            )
        mass = _as_finite_float(p.get("mass"), f"试管 {tid} 的标称质量")
        if mass <= 0.0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        angle = _as_finite_float(p.get("angle"),
                                 f"试管 {tid} 的摆放槽位角度")
        locked = bool(p.get("locked", False))
        # 占用关系交叉核对: 槽位状态登记必须与摆放明细一致
        if not info.enabled:
            raise ValueError(
                f"试管 {tid} 落在禁用槽 {slot}: 快照数据互相矛盾"
                "(已放置试管不得占用禁用槽位)。"
            )
        if info.tube_id != tid:
            raise ValueError(
                f"槽位 {slot} 的占用登记(试管 {info.tube_id})与摆放明细"
                f"(试管 {tid})互相矛盾。"
            )
        if info.locked != locked:
            raise ValueError(
                f"试管 {tid} 的锁定标记在槽位状态与摆放明细中不一致, "
                "快照数据互相矛盾。"
            )
        if abs(angle - info.angle_deg) > _ANGLE_TOL * max(
                1.0, abs(angle), abs(info.angle_deg)):
            raise ValueError(
                f"试管 {tid} 的摆放角度({angle:g}°)与槽位 {slot} 的槽角"
                f"({info.angle_deg:g}°)不一致, 快照数据互相矛盾。"
            )
        tube = TubeSnapshot(
            tube_id=tid, mass_g=mass, slot=slot,
            angle_deg=angle, locked=locked,
        )
        tubes.append(tube)
        tube_map[tid] = tube

    # 反向核对: slot_state 登记有管的槽位, 必须是某支已放置试管的落槽
    for s in slots:
        if s.tube_id is None:
            continue
        occupant = tube_map.get(s.tube_id)
        if occupant is None or occupant.slot != s.slot:
            raise ValueError(
                f"槽位状态登记槽 {s.slot} 被试管 {s.tube_id} 占用, "
                "但摆放明细中该管并未落在该槽, 快照数据互相矛盾。"
            )

    return PlanSnapshot(
        plan_id=plan_id,
        radius_mm=radius,
        slots=slots,
        tubes=tuple(tubes),
        unplaced_tube_ids=tuple(sorted(unplaced)),
        slot_map=slot_map,
        tube_map=tube_map,
    )
