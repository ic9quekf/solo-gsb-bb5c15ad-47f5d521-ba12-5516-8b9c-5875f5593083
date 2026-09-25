"""已保存方案的物理快照解析器(失效槽迁移/库存配重/备品替换共用, 只读)。

三个只读规划器都要从 GET /api/plan/{id} 的结果中还原转子的物理事实:
半径 R、逐槽角度与启用状态(``slot_state``)、逐管落槽与质量(``placements``),
并据此计算质径积向量 ``v = R·m·e^{iθ}``(g·mm)。历史记录是落库的外部数据,
可能被旧版本、手工导出/导入或损坏写坏; 若各规划器各自宽容解析, 就可能在
槽角、占用关系互相矛盾的快照上生成看似"达标"的错误建议。

本模块在**任何搜索开始之前**对方案结果做一次统一、严格、只读的物理快照
解析与交叉核对:

- **半径**: 取 ``result.config.radius_mm``(缺省回退 ``request.radius_mm``),
  必须存在且为有限正数;
- **槽位状态**: 必须存在且非空; 槽号为整数、不重复且恰好连续覆盖
  0..N-1(与 ``config.slot_count`` 一致), 角度为有限数, 启用状态为布尔;
- **摆放明细**: 必须为列表; 管号/槽号为整数(不接受布尔)、管号全表不重复,
  质量存在且为有限正数; 已放置行(``slot`` 非空)的槽号必须存在;
- **交叉核对(槽号 ⇄ 管号 ⇄ 占用)**:
    1. 已放置明细指向的槽位必须启用, 不得落在禁用槽;
    2. 明细角度必须与该槽在 ``slot_state`` 中的角度一致(1e-9 相对容差)
       —— 槽位状态是槽角的唯一事实来源;
    3. 明细占用与 ``slot_state.tube_id`` 必须一一对应: 不得缺失、不得多余、
       管号必须相同;
    4. 行内 ``locked`` 与槽位状态 ``locked`` 不得矛盾;
    5. 一个槽位至多被一支试管占用(槽号不重复)。

任何不满足都抛 :class:`ValueError`(消息以方案 ID 开头), 由三个接口统一
转为 HTTP 400, 在搜索前拒绝, 不生成任何达标建议。解析不修改入参方案。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# 与 evacuation/counterweight/spare_replacement 一致的数值容差
_NUM_TOL = 1e-9


@dataclass(frozen=True)
class TubeSnapshot:
    """一支已放置试管的物理事实(管号升序排列)。"""

    tube_id: int
    slot: int
    angle_deg: float
    mass_g: float
    locked: bool
    # 质径积向量 R·m·e^{iθ} (g·mm), 解析期一次性算好供三个规划器共用
    vector: complex


@dataclass(frozen=True)
class PlanSnapshot:
    """已保存方案结果的物理快照。

    Attributes
    ----------
    plan_id, name, created_at: 方案元信息(只用于错误消息与响应回显)。
    radius_mm: 转子半径(mm), 有限正数。
    rpm: 转速(rpm); 缺省 0(仅备品替换的离心力换算用到)。
    slot_count: 槽位总数 N, 槽号恰好覆盖 0..N-1。
    slots: 槽号 -> {"angle_deg", "enabled", "tube_id", "locked"}。
    tubes: 已放置试管, 按管号升序; 至少一支由各规划器自行判定。
    occupied: 槽号 -> 已放置试管管号(与 slots[s]["tube_id"] 一致)。
    unplaced_tube_ids: 明细中未放置(``slot`` 为 null)的试管编号。
    base_vector: 全部已放置试管质径积向量之和 R·Σ m·e^{iθ} (g·mm)。
    """

    plan_id: Any
    name: Any
    created_at: Any
    radius_mm: float
    rpm: float
    slot_count: int
    slots: dict[int, dict[str, Any]]
    tubes: list[TubeSnapshot]
    occupied: dict[int, int]
    unplaced_tube_ids: list[int]
    base_vector: complex

    # ---- 便捷访问 ----
    @property
    def enabled_slots(self) -> list[int]:
        """启用槽号(升序)。"""
        return sorted(s for s, info in self.slots.items() if info["enabled"])

    @property
    def free_enabled_slots(self) -> list[int]:
        """空闲启用槽号(升序): 启用且未被任何已放置试管占用。"""
        return sorted(
            s for s, info in self.slots.items()
            if info["enabled"] and s not in self.occupied
        )

    def angle_of(self, slot: int) -> float:
        """槽位角度(度), 槽号不存在时抛 KeyError。"""
        return self.slots[slot]["angle_deg"]

    def tube_at(self, slot: int) -> Optional[TubeSnapshot]:
        """占用某槽位的试管; 空槽返回 None。"""
        tid = self.occupied.get(slot)
        if tid is None:
            return None
        return next(t for t in self.tubes if t.tube_id == tid)

    def mass_radius_vector(self, mass_g: float, slot: int) -> complex:
        """质量 mass_g 的物体放在 slot 时的质径积向量 R·m·e^{iθ} (g·mm)。

        三个规划器(配重、备品、迁移注入)共用同一向量计算, 角度一律以
        ``slot_state`` 为唯一事实来源。
        """
        a = math.radians(self.slots[slot]["angle_deg"])
        v = self.radius_mm * mass_g * complex(math.cos(a), math.sin(a))
        if not math.isfinite(v.real) or not math.isfinite(v.imag):
            raise ValueError(
                f"方案 {self.plan_id} 的质径积计算溢出"
                f"(质量 {mass_g:g} g、半径 {self.radius_mm:g} mm、"
                f"槽位 {slot}), 请检查方案数据。"
            )
        return v


def _as_int(raw: Any, what: str, plan_id: Any) -> int:
    """严格整数转换: 拒绝布尔(``True`` 不等于 1)与不可转换值。"""
    if isinstance(raw, bool):
        raise ValueError(f"方案 {plan_id} 的{what}必须为整数编号(当前 {raw!r})。")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"方案 {plan_id} 的{what}必须为整数编号(当前 {raw!r})。"
        ) from exc


def _angles_equal(a: float, b: float) -> bool:
    """两个角度是否在 1e-9 相对容差内一致。"""
    return abs(a - b) <= _NUM_TOL * max(1.0, abs(a), abs(b))


def parse_plan_snapshot(plan: Any) -> PlanSnapshot:
    """把已保存方案(GET /api/plan/{id} 结构)解析为 :class:`PlanSnapshot`。

    只读: 不修改 ``plan``。任何结构缺失、重复、非有限或槽号/管号/占用
    互相矛盾的数据都抛 :class:`ValueError`, 供接口层在搜索前转为 400。
    """
    plan_id = plan.get("id") if isinstance(plan, dict) else None

    def err(msg: str) -> ValueError:
        return ValueError(f"方案 {plan_id} 的物理快照无效: {msg}")

    if not isinstance(plan, dict):
        raise ValueError("已保存方案数据无效: 方案必须为对象。")
    result = plan.get("result")
    if not isinstance(result, dict):
        raise err("缺少方案结果(result), 无法解析转子物理快照。")

    # ---- 半径: result.config 优先, 回退 request.radius_mm; 必须有限为正 ----
    cfg = result.get("config")
    if not isinstance(cfg, dict):
        raise err("缺少方案配置(result.config), 无法解析转子半径等物理参数。")
    radius_raw = cfg.get("radius_mm")
    if radius_raw is None:
        request = plan.get("request")
        radius_raw = request.get("radius_mm") if isinstance(request, dict) else None
    try:
        radius = float(radius_raw)
    except (TypeError, ValueError) as exc:
        raise err("转子半径必须为数值。") from exc
    if not math.isfinite(radius):
        raise err("转子半径必须为有限数值。")
    if radius <= 0.0:
        raise err("转子半径必须为正数。")

    # ---- 转速(仅备品替换的力学量换算需要): 缺省 0, 显式给出时必须有限 ----
    rpm_raw = cfg.get("rpm")
    if rpm_raw is None:
        request = plan.get("request")
        rpm_raw = request.get("rpm") if isinstance(request, dict) else None
    if rpm_raw is None:
        rpm = 0.0
    else:
        try:
            rpm = float(rpm_raw)
        except (TypeError, ValueError) as exc:
            raise err("转速必须为数值。") from exc
        if not math.isfinite(rpm):
            raise err("转速必须为有限数值。")
        if rpm < 0.0:
            raise err("转速不能为负。")

    # ---- 槽位状态 slot_state ----
    slot_state = result.get("slot_state")
    if not isinstance(slot_state, list) or not slot_state:
        raise err("缺少槽位状态信息(slot_state), 无法解析槽角与占用关系。")
    declared_n = cfg.get("slot_count")
    n_decl: Optional[int] = None
    if declared_n is not None:
        n_decl = _as_int(declared_n, "配置槽位数(slot_count)", plan_id)
        if n_decl <= 0:
            raise err(f"配置槽位数必须为正数(当前 {n_decl})。")

    slots: dict[int, dict[str, Any]] = {}
    for item in slot_state:
        if not isinstance(item, dict):
            raise err("槽位状态中的每项必须为对象。")
        idx = _as_int(item.get("slot"), "槽位编号", plan_id)
        if idx < 0:
            raise err(f"槽位编号不能为负(当前 {idx})。")
        if idx in slots:
            raise err(f"槽位状态中槽位 {idx} 重复。")
        try:
            ang = float(item["angle"])
        except (KeyError, TypeError, ValueError) as exc:
            raise err(f"槽位 {idx} 的角度缺失或不是数值。") from exc
        if not math.isfinite(ang):
            raise err(f"槽位 {idx} 的角度必须为有限数值。")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise err(f"槽位 {idx} 的启用状态必须为布尔值。")
        state_tid = item.get("tube_id")
        if state_tid is not None:
            state_tid = _as_int(state_tid, f"槽位 {idx} 的占用管号", plan_id)
        state_locked = item.get("locked", False)
        if not isinstance(state_locked, bool):
            raise err(f"槽位 {idx} 的锁定标记必须为布尔值。")
        slots[idx] = {
            "angle_deg": ang,
            "enabled": enabled,
            "tube_id": state_tid,
            "locked": state_locked,
        }

    # 槽号必须恰好连续覆盖 0..N-1(与 config.slot_count 一致):
    # 规划器按物理槽号索引, 编号缺失/空洞或越界都会让"槽角/占用"结论失真。
    n = len(slots)
    if n_decl is not None and n_decl != n:
        raise err(
            f"配置槽位数 {n_decl} 与槽位状态条数 {n} 不一致。"
        )
    missing = sorted(set(range(n)) - set(slots))
    if missing:
        raise err(
            "槽位编号必须恰好覆盖 0~%d; 缺失编号: %s。"
            % (n - 1, ", ".join(map(str, missing)))
        )
    extra = sorted(s for s in slots if s >= n)
    if extra:
        raise err(
            "槽位编号必须恰好覆盖 0~%d; 越界编号: %s。"
            % (n - 1, ", ".join(map(str, extra)))
        )

    # ---- 摆放明细 placements ----
    placements = result.get("placements")
    if not isinstance(placements, list):
        raise err("缺少摆放明细(placements), 无法解析试管与落槽。")

    tubes: list[TubeSnapshot] = []
    occupied: dict[int, int] = {}
    unplaced: list[int] = []
    seen_ids: set[int] = set()
    for p in placements:
        if not isinstance(p, dict):
            raise err("摆放明细中的每项必须为对象。")
        tid = _as_int(p.get("tube_id"), "摆放明细中的试管编号", plan_id)
        if tid in seen_ids:
            raise err(f"摆放明细中试管 {tid} 重复。")
        seen_ids.add(tid)
        try:
            mass = float(p.get("mass"))
        except (TypeError, ValueError) as exc:
            raise err(f"试管 {tid} 的标称质量必须为数值。") from exc
        if not math.isfinite(mass):
            raise err(f"试管 {tid} 的标称质量必须为有限数值。")
        if mass <= 0.0:
            raise err(f"试管 {tid} 的标称质量必须为正数。")
        locked = p.get("locked", False)
        if not isinstance(locked, bool):
            raise err(f"试管 {tid} 的锁定标记必须为布尔值。")
        raw_slot = p.get("slot")
        if raw_slot is None:
            # 未放置行: 不参与占用与质径积; 编号已计入 seen_ids 防止与已放置
            # 管重复。质量为正的校验对全表统一进行。
            unplaced.append(tid)
            continue
        slot = _as_int(raw_slot, f"试管 {tid} 的槽位", plan_id)
        if slot not in slots:
            raise err(
                f"摆放明细中试管 {tid} 落在不存在的槽位 {slot}。"
            )
        if slot in occupied:
            raise err(f"摆放明细中槽位 {slot} 被重复占用。")
        info = slots[slot]
        if not info["enabled"]:
            raise err(
                f"摆放明细中试管 {tid} 落在禁用槽位 {slot}, 占用关系矛盾。"
            )
        angle_raw = p.get("angle")
        if angle_raw is None:
            raise err(f"试管 {tid} 的摆放结果缺少槽位角度。")
        try:
            angle = float(angle_raw)
        except (TypeError, ValueError) as exc:
            raise err(f"试管 {tid} 的槽位角度不是数值。") from exc
        if not math.isfinite(angle):
            raise err(f"试管 {tid} 的槽位角度必须为有限数值。")
        if not _angles_equal(angle, info["angle_deg"]):
            raise err(
                f"试管 {tid} 的摆放角度 {angle:g}° 与槽位 {slot} 的"
                f"实际槽角 {info['angle_deg']:g}° 不一致。"
            )
        # 交叉核对: 明细占用必须与 slot_state.tube_id 一一对应
        state_tid = info["tube_id"]
        if state_tid is None:
            raise err(
                f"摆放明细中试管 {tid} 占用槽位 {slot}, 但槽位状态中该槽"
                "未标注占用管号, 占用关系缺失。"
            )
        if state_tid != tid:
            raise err(
                f"槽位 {slot} 在摆放明细中由试管 {tid} 占用, 槽位状态却"
                f"标注为试管 {state_tid}, 占用关系互相矛盾。"
            )
        if bool(info["locked"]) != locked:
            raise err(
                f"试管 {tid}(槽位 {slot})的锁定状态在摆放明细"
                f"({'锁定' if locked else '未锁定'})与槽位状态"
                f"({'锁定' if info['locked'] else '未锁定'})中不一致。"
            )
        a = math.radians(info["angle_deg"])
        vec = radius * mass * complex(math.cos(a), math.sin(a))
        if not math.isfinite(vec.real) or not math.isfinite(vec.imag):
            raise err(
                f"试管 {tid} 的质径积计算溢出(质量 {mass:g} g、半径 "
                f"{radius:g} mm), 请检查方案数据。"
            )
        occupied[slot] = tid
        tubes.append(TubeSnapshot(
            tube_id=tid, slot=slot, angle_deg=info["angle_deg"],
            mass_g=mass, locked=locked, vector=vec,
        ))

    # 反向交叉核对: slot_state 标注有管的槽必须能在明细中找到同一支管
    for s, info in slots.items():
        state_tid = info["tube_id"]
        if state_tid is None:
            if s in occupied:
                # 理论上不会进入(正向写入时已要求 state_tid 非空), 兜底防御
                raise err(  # pragma: no cover - 与正向检查互斥
                    f"槽位状态中槽位 {s} 未标注占用, 却在摆放明细中被试管 "
                    f"{occupied[s]} 占用。"
                )
            continue
        if s not in occupied:
            raise err(
                f"槽位状态标注槽位 {s} 由试管 {state_tid} 占用, "
                "但摆放明细中缺少该占用记录。"
            )
        if occupied[s] != state_tid:
            raise err(
                f"槽位 {s} 的占用管号在摆放明细({occupied[s]})与槽位状态"
                f"({state_tid})中不一致。"
            )

    tubes.sort(key=lambda t: t.tube_id)
    unplaced.sort()
    total_vec = sum((t.vector for t in tubes), 0j)

    return PlanSnapshot(
        plan_id=plan.get("id"),
        name=plan.get("name"),
        created_at=plan.get("created_at"),
        radius_mm=radius,
        rpm=rpm,
        slot_count=n,
        slots=slots,
        tubes=tubes,
        occupied=occupied,
        unplaced_tube_ids=unplaced,
        base_vector=total_vec,
    )
