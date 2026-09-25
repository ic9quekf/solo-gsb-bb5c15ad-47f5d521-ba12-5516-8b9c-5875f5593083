"""共用已保存方案物理快照解析器测试: python -m unittest tests.test_plan_snapshot

核心断言:
- 正常方案快照被完整解析(半径、转速、槽角、启用/占用、已放置试管),
  质径积向量与基线 U₁ 独立复算一致, 解析不修改入参(只读);
- 半径缺失/非正/非有限, 槽位状态缺失/空/重复/缺号/非有限角度,
  摆放明细缺失/管号或槽位重复/质量非正/缺角度等**在搜索前**即被拒绝;
- 交叉核对: 明细指向不存在/禁用槽、明细角度与槽角矛盾、占用关系缺失或
  多余(明细 ⇄ slot_state.tube_id)、管号错配、locked 矛盾一律拒绝;
- 布尔值不被当作整数编号; 未放置行的编号同样不得与已放置管重复,
  且纳入"原方案管号"集合。
"""
import copy
import math
import unittest

from app.plan_snapshot import parse_plan_snapshot


def make_plan(radius=100.0, rpm=4000.0, slots_n=12, enabled=None):
    if enabled is None:
        enabled = set(range(slots_n))
    placements = {0: (1, 50.0, False), 6: (2, 50.0, False)}
    slot_state = [
        {
            "slot": i,
            "angle": 360.0 * i / slots_n,
            "enabled": i in enabled,
            "tube_id": next(
                (t for s, (t, _, _) in placements.items() if s == i), None
            ),
            "locked": next(
                (lk for s, (_, _, lk) in placements.items() if s == i), False
            ),
        }
        for i in range(slots_n)
    ]
    pls = [
        {
            "tube_id": t,
            "mass": m,
            "slot": s,
            "angle": 360.0 * s / slots_n,
            "locked": lk,
        }
        for s, (t, m, lk) in placements.items()
    ]
    return {
        "id": 7,
        "name": "快照方案",
        "created_at": "2026-09-25T00:00:00+00:00",
        "request": {"radius_mm": radius, "rpm": rpm,
                    "tubes": [{"id": 1}, {"id": 2}]},
        "result": {
            "config": {"radius_mm": radius, "rpm": rpm,
                       "slot_count": slots_n},
            "slot_state": slot_state,
            "placements": pls,
        },
    }


def mutate(plan, path, value):
    """按路径(键序列)深拷贝并改写方案, 返回新对象。"""
    out = copy.deepcopy(plan)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out


class TestPlanSnapshotOK(unittest.TestCase):
    def test_parse_valid_plan(self):
        plan = make_plan()
        snap = parse_plan_snapshot(plan)
        self.assertEqual(snap.plan_id, 7)
        self.assertEqual(snap.radius_mm, 100.0)
        self.assertEqual(snap.rpm, 4000.0)
        self.assertEqual(snap.slot_count, 12)
        self.assertEqual(len(snap.slots), 12)
        self.assertEqual(snap.occupied, {0: 1, 6: 2})
        self.assertEqual([t.tube_id for t in snap.tubes], [1, 2])
        self.assertEqual(snap.tubes[0].slot, 0)
        self.assertEqual(snap.tubes[1].slot, 6)
        self.assertEqual(snap.free_enabled_slots,
                         [1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
        # 对径等质量两管: |U₁| = 0
        self.assertAlmostEqual(abs(snap.base_vector), 0.0, places=9)
        # 单管向量独立复算
        t1 = snap.tubes[0]
        self.assertAlmostEqual(t1.vector.real, 5000.0, places=6)
        self.assertAlmostEqual(t1.vector.imag, 0.0, places=6)
        # mass_radius_vector 与 TubeSnapshot.vector 同一计算
        self.assertAlmostEqual(
            snap.mass_radius_vector(50.0, 0).real, t1.vector.real, places=9)
        self.assertEqual(snap.angle_of(6), 180.0)
        self.assertEqual(snap.tube_at(0).tube_id, 1)
        self.assertIsNone(snap.tube_at(3))

    def test_disabled_slot_not_free(self):
        plan = make_plan(enabled=set(range(12)) - {3})
        snap = parse_plan_snapshot(plan)
        self.assertNotIn(3, snap.enabled_slots)
        self.assertNotIn(3, snap.free_enabled_slots)
        self.assertFalse(snap.slots[3]["enabled"])

    def test_radius_fallback_to_request(self):
        plan = make_plan()
        del plan["result"]["config"]["radius_mm"]
        plan["request"]["radius_mm"] = 80.0
        snap = parse_plan_snapshot(plan)
        self.assertEqual(snap.radius_mm, 80.0)
        # rpm 缺省为 0
        plan2 = make_plan()
        del plan2["result"]["config"]["rpm"]
        del plan2["request"]["rpm"]
        self.assertEqual(parse_plan_snapshot(plan2).rpm, 0.0)

    def test_unplaced_tubes_recorded(self):
        plan = make_plan()
        plan["result"]["placements"].append({
            "tube_id": 9, "mass": 12.0, "slot": None, "angle": None,
            "locked": False,
        })
        snap = parse_plan_snapshot(plan)
        self.assertEqual(snap.unplaced_tube_ids, [9])
        self.assertEqual(len(snap.tubes), 2)
        self.assertNotIn(9, snap.occupied.values())

    def test_read_only(self):
        plan = make_plan()
        before = copy.deepcopy(plan)
        parse_plan_snapshot(plan)
        self.assertEqual(plan, before)


class TestPlanSnapshotReject(unittest.TestCase):
    def _assert(self, plan):
        with self.assertRaises(ValueError):
            parse_plan_snapshot(plan)

    def test_result_and_config(self):
        p = make_plan()
        bad = copy.deepcopy(p)
        bad["result"] = []
        self._assert(bad)
        bad = copy.deepcopy(p)
        del bad["result"]
        self._assert(bad)
        bad = copy.deepcopy(p)
        bad["result"] = {"slot_state": [], "placements": []}
        self._assert(bad)
        self._assert("not-a-plan")

    def test_radius(self):
        for rad in (0.0, -1.0, float("nan"), float("inf"), "x"):
            plan = make_plan()
            plan["result"]["config"]["radius_mm"] = rad
            self._assert(plan)
        # None 在 config 中会回退 request; 只有两处都缺失才拒绝
        plan = make_plan()
        plan["result"]["config"]["radius_mm"] = None
        snap = parse_plan_snapshot(plan)
        self.assertEqual(snap.radius_mm, 100.0)
        plan = make_plan()
        del plan["result"]["config"]["radius_mm"]
        del plan["request"]["radius_mm"]
        self._assert(plan)

    def test_rpm(self):
        for rpmv in (-1.0, float("nan"), float("inf"), "x"):
            plan = make_plan()
            plan["result"]["config"]["rpm"] = rpmv
            self._assert(plan)

    def test_slot_state(self):
        p = make_plan()
        # 缺失 / 为空 / 非列表
        for ss in (None, [], {}):
            bad = copy.deepcopy(p)
            bad["result"]["slot_state"] = ss
            self._assert(bad)
        # 槽号重复
        bad = mutate(p, ["result", "slot_state", 3, "slot"], 2)
        self._assert(bad)
        # 缺号(没有 0..N-1 全覆盖): 把 11 改成 12
        bad = mutate(p, ["result", "slot_state", 11, "slot"], 12)
        self._assert(bad)
        # 非有限角度
        bad = mutate(p, ["result", "slot_state", 5, "angle"], float("nan"))
        self._assert(bad)
        # 缺角度
        bad = copy.deepcopy(p)
        del bad["result"]["slot_state"][5]["angle"]
        self._assert(bad)
        # enabled 非布尔
        bad = mutate(p, ["result", "slot_state", 5, "enabled"], 1)
        self._assert(bad)
        # config.slot_count 与条数不一致
        bad = mutate(p, ["result", "config", "slot_count"], 8)
        self._assert(bad)
        # 布尔槽号
        bad = mutate(p, ["result", "slot_state", 3, "slot"], True)
        self._assert(bad)

    def test_placements_basic(self):
        p = make_plan()
        for pls in (None, {}):
            bad = copy.deepcopy(p)
            bad["result"]["placements"] = pls
            self._assert(bad)
        # 管号重复
        bad = mutate(p, ["result", "placements", 1, "tube_id"], 1)
        self._assert(bad)
        # 槽位重复
        bad = mutate(p, ["result", "placements", 1, "slot"], 0)
        # 槽位重复后 slot_state 仍指向管 2 → 同时构成占用矛盾, 均应拒绝
        self._assert(bad)
        # 质量非正 / 非有限 / 缺失
        for m in (0.0, -5.0, float("inf"), float("nan")):
            bad = mutate(p, ["result", "placements", 0, "mass"], m)
            self._assert(bad)
        bad = copy.deepcopy(p)
        del bad["result"]["placements"][0]["mass"]
        self._assert(bad)
        # 缺角度
        bad = mutate(p, ["result", "placements", 0, "angle"], None)
        self._assert(bad)
        # 非有限角度
        bad = mutate(p, ["result", "placements", 0, "angle"], float("inf"))
        self._assert(bad)
        # 布尔管号
        bad = mutate(p, ["result", "placements", 0, "tube_id"], True)
        self._assert(bad)

    def test_cross_checks(self):
        p = make_plan()
        # 明细指向不存在的槽
        bad = mutate(p, ["result", "placements", 1, "slot"], 42)
        bad["result"]["placements"][1]["angle"] = 360.0 * 42 / 12
        self._assert(bad)
        # 明细落在禁用槽(槽 5 禁用, 但明细管 2 改到槽 5)
        bad = copy.deepcopy(p)
        bad["result"]["slot_state"][5]["enabled"] = False
        bad["result"]["slot_state"][5]["tube_id"] = 2
        bad["result"]["slot_state"][6]["tube_id"] = None
        bad["result"]["slot_state"][6]["locked"] = False
        row = bad["result"]["placements"][1]
        row["slot"] = 5
        row["angle"] = 360.0 * 5 / 12
        self._assert(bad)
        # 明细角度与槽角矛盾
        bad = mutate(p, ["result", "placements", 0, "angle"], 1.0)
        self._assert(bad)
        # 槽位状态标注的占用管号与明细不一致
        bad = mutate(p, ["result", "slot_state", 6, "tube_id"], 1)
        self._assert(bad)
        # 明细有占用, slot_state 却标空(占用缺失)
        bad = mutate(p, ["result", "slot_state", 0, "tube_id"], None)
        self._assert(bad)
        # slot_state 标注占用, 明细缺少该记录(占用多余):
        # 删除管 1 明细行(槽 0 在状态中仍标注管 1)
        bad = copy.deepcopy(p)
        bad["result"]["placements"] = [
            row for row in bad["result"]["placements"]
            if row["tube_id"] != 1
        ]
        self._assert(bad)
        # locked 矛盾
        bad = mutate(p, ["result", "placements", 0, "locked"], True)
        self._assert(bad)
        bad = mutate(p, ["result", "slot_state", 0, "locked"], True)
        self._assert(bad)

    def test_unplaced_row_conflicts(self):
        p = make_plan()
        # 未放置行编号与已放置管重复
        p["result"]["placements"].append({
            "tube_id": 1, "mass": 9.0, "slot": None, "angle": None,
            "locked": False,
        })
        self._assert(p)
        # 未放置行质量非法仍需拒绝
        p2 = make_plan()
        p2["result"]["placements"].append({
            "tube_id": 9, "mass": 0.0, "slot": None, "angle": None,
            "locked": False,
        })
        self._assert(p2)

    def test_slot_state_occupying_disabled(self):
        """slot_state 自相矛盾: 禁用槽却标注有管(明细也一致)。"""
        p = copy.deepcopy(make_plan())
        st = p["result"]["slot_state"]
        st[0]["enabled"] = False
        self._assert(p)


if __name__ == "__main__":
    unittest.main()
