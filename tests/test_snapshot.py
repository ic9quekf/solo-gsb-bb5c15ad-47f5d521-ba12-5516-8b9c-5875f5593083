"""已保存方案物理快照解析器测试: python -m unittest discover -s tests

核心断言: 解析器从结果中的半径、槽位状态和摆放明细提取槽角、占用与试管;
交叉核对槽号、管号及占用关系, 对重复、缺失、非有限或互相矛盾的数据一律
抛 ValueError(API 层转为 400); 失效槽迁移、库存配重、备品替换三个只读
规划器在快照无效时必须于任何搜索之前拒绝, 不得基于损坏数据生成达标建议;
解析与规划均不修改入参(只读)。
"""
import copy
import math
import unittest

from app.counterweight import plan_counterweight_compensation
from app.evacuation import plan_slot_evacuation
from app.snapshot import mr_block, mr_vector, parse_plan_snapshot
from app.spare_replacement import plan_spare_replacement


def _plan(pid=1, placements=None, radius=100.0, slots_n=12, enabled=None,
          rpm=4000.0):
    """placements: {槽号: (试管编号, 质量, 是否锁定)}; 角度按槽号均布。"""
    if placements is None:
        placements = {0: (1, 50.0, True), 6: (2, 50.0, False)}
    if enabled is None:
        enabled = set(range(slots_n))
    slot_state = [
        {
            "slot": i,
            "angle": 360.0 * i / slots_n,
            "enabled": i in enabled,
            "tube_id": next(
                (t for s, (t, m, lk) in placements.items() if s == i), None
            ),
            "locked": next(
                (lk for s, (t, m, lk) in placements.items() if s == i), False
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
        "id": pid,
        "name": f"方案{pid}",
        "created_at": "2026-09-25T00:00:00+00:00",
        "request": {
            "radius_mm": radius, "rpm": rpm,
            "tubes": [{"id": t} for t, _, _ in placements.values()],
        },
        "result": {
            "config": {"radius_mm": radius, "rpm": rpm},
            "slot_state": slot_state,
            "placements": pls,
            "metrics": {"u1_mag_g_mm": 0.0},
        },
    }


class TestMrVector(unittest.TestCase):
    """共用质径积向量计算: R·m·e^{iθ} 与展示块。"""

    def test_vector_values(self):
        v = mr_vector(50.0, 100.0, 0.0)
        self.assertAlmostEqual(v.real, 5000.0, places=12)
        self.assertAlmostEqual(v.imag, 0.0, places=12)
        v = mr_vector(50.0, 100.0, 90.0)
        self.assertAlmostEqual(v.real, 0.0, places=9)
        self.assertAlmostEqual(v.imag, 5000.0, places=9)
        v = mr_vector(2.0, 150.0, 60.0)
        self.assertAlmostEqual(v.real, 150.0, places=9)
        self.assertAlmostEqual(v.imag, 150.0 * math.sqrt(3.0), places=9)

    def test_block_fields(self):
        b = mr_block(3.0, 4.0)
        self.assertEqual(
            set(b), {"u1_x_g_mm", "u1_y_g_mm", "u1_mag_g_mm", "u1_angle_deg"})
        self.assertAlmostEqual(b["u1_mag_g_mm"], 5.0, places=12)
        self.assertAlmostEqual(b["u1_angle_deg"],
                               math.degrees(math.atan2(4.0, 3.0)), places=9)
        # 负象限方向角归一化到 [0, 360)
        self.assertAlmostEqual(mr_block(0.0, -2.0)["u1_angle_deg"], 270.0)


class TestParseValidSnapshot(unittest.TestCase):
    """合法方案: 快照完整提取半径、槽角、启用、占用与试管。"""

    def test_extracts_slots_tubes_occupancy(self):
        p = _plan(7, {0: (1, 50.0, True), 6: (2, 49.5, False)},
                  radius=120.0, enabled=set(range(12)) - {11})
        snap = parse_plan_snapshot(p)
        self.assertEqual(snap.plan_id, 7)
        self.assertAlmostEqual(snap.radius_mm, 120.0)
        self.assertEqual(len(snap.slots), 12)
        self.assertEqual([s.slot for s in snap.slots], list(range(12)))
        self.assertEqual(snap.enabled_count, 11)
        # 槽角按槽号均布
        for s in snap.slots:
            self.assertAlmostEqual(s.angle_deg, 30.0 * s.slot)
        # 占用关系
        self.assertEqual(snap.slot_map[0].tube_id, 1)
        self.assertTrue(snap.slot_map[0].locked)
        self.assertEqual(snap.slot_map[6].tube_id, 2)
        self.assertFalse(snap.slot_map[6].locked)
        self.assertIsNone(snap.slot_map[3].tube_id)
        self.assertFalse(snap.slot_map[11].enabled)
        # 已放置试管(摆放明细原顺序)
        self.assertEqual([t.tube_id for t in snap.tubes], [1, 2])
        t1 = snap.tube_map[1]
        self.assertEqual((t1.slot, t1.mass_g, t1.angle_deg, t1.locked),
                         (0, 50.0, 0.0, True))
        t2 = snap.tube_map[2]
        self.assertEqual((t2.slot, t2.mass_g, t2.angle_deg, t2.locked),
                         (6, 49.5, 180.0, False))
        self.assertEqual(snap.unplaced_tube_ids, ())
        # 空闲启用槽: 除 0/6(占用)与 11(禁用)外
        self.assertEqual([s.slot for s in snap.free_slots],
                         [1, 2, 3, 4, 5, 7, 8, 9, 10])

    def test_radius_fallback_chain(self):
        # config 缺 radius_mm 时回退 request, 再缺省取 100.0
        p = _plan(1, radius=80.0)
        del p["result"]["config"]["radius_mm"]
        self.assertAlmostEqual(parse_plan_snapshot(p).radius_mm, 80.0)
        del p["request"]["radius_mm"]
        self.assertAlmostEqual(parse_plan_snapshot(p).radius_mm, 100.0)

    def test_unplaced_tube_ids_collected(self):
        p = _plan(1)
        p["result"]["placements"].append(
            {"tube_id": 9, "mass": 40.0, "slot": None, "angle": None,
             "locked": False})
        snap = parse_plan_snapshot(p)
        self.assertEqual(snap.unplaced_tube_ids, (9,))
        self.assertEqual([t.tube_id for t in snap.tubes], [1, 2])

    def test_read_only(self):
        p = _plan(3)
        before = copy.deepcopy(p)
        parse_plan_snapshot(p)
        self.assertEqual(p, before)


class TestSnapshotRejections(unittest.TestCase):
    """重复、缺失、非有限或互相矛盾的快照数据一律 ValueError。"""

    def test_missing_result(self):
        with self.assertRaises(ValueError):
            parse_plan_snapshot({"id": 1, "result": None})
        with self.assertRaises(ValueError):
            parse_plan_snapshot({"id": 1})

    def test_bad_radius(self):
        for bad in (0.0, -5.0, float("nan"), float("inf"), None, "abc"):
            p = _plan(1)
            p["result"]["config"]["radius_mm"] = bad
            p["request"]["radius_mm"] = bad
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_plan_snapshot(p)

    def test_slot_state_missing_or_empty(self):
        p = _plan(1)
        del p["result"]["slot_state"]
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)
        p = _plan(1)
        p["result"]["slot_state"] = []
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_duplicate_slot_number(self):
        p = _plan(1)
        p["result"]["slot_state"].append(dict(p["result"]["slot_state"][0]))
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_missing_and_out_of_range_slot_numbers(self):
        # 缺失槽号 5(非连续)
        p = _plan(1)
        del p["result"]["slot_state"][5]
        for item in p["result"]["slot_state"]:
            if item["slot"] > 5:
                item["slot"] -= 1
        # 现在槽号 0..10 共 11 项, 但摆放槽 6 已变 5 → 占用也矛盾;
        # 仅验证抛错(缺失/矛盾任一)
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)
        # 越界槽号: 12 个槽但编号到 12
        p = _plan(1)
        p["result"]["slot_state"][11]["slot"] = 12
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_non_finite_slot_angle(self):
        for bad in (float("nan"), float("inf"), None):
            p = _plan(1)
            p["result"]["slot_state"][3]["angle"] = bad
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_plan_snapshot(p)

    def test_duplicate_tube_id(self):
        p = _plan(1)
        p["result"]["placements"][1]["tube_id"] = 1
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)
        # 未放置试管与已放置试管同号同样拒绝
        p = _plan(1)
        p["result"]["placements"].append(
            {"tube_id": 1, "mass": 40.0, "slot": None, "angle": None})
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_duplicate_slot_occupancy(self):
        p = _plan(1)
        p["result"]["placements"][1]["slot"] = 0
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_placement_slot_not_in_slot_state(self):
        p = _plan(1)
        p["result"]["placements"][0]["slot"] = 99
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_bad_mass(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), None):
            p = _plan(1)
            p["result"]["placements"][0]["mass"] = bad
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_plan_snapshot(p)

    def test_bad_placement_angle(self):
        for bad in (None, float("nan"), float("inf")):
            p = _plan(1)
            p["result"]["placements"][0]["angle"] = bad
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_plan_snapshot(p)

    def test_angle_mismatch_between_placement_and_slot(self):
        p = _plan(1)
        p["result"]["placements"][0]["angle"] = 30.0  # 槽 0 的槽角为 0°
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_occupancy_contradiction(self):
        # slot_state 登记槽 0 被管 2 占用, 摆放明细中管 1 落在槽 0
        p = _plan(1)
        p["result"]["slot_state"][0]["tube_id"] = 2
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)
        # slot_state 登记空槽, 摆放明细却有管落在该槽
        p = _plan(1)
        p["result"]["slot_state"][0]["tube_id"] = None
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)
        # slot_state 登记了摆放明细中不存在的占用管
        p = _plan(1)
        p["result"]["slot_state"][3]["tube_id"] = 99
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_tube_on_disabled_slot(self):
        p = _plan(1, enabled=set(range(12)) - {0})
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_locked_flag_mismatch(self):
        p = _plan(1)
        p["result"]["slot_state"][0]["locked"] = False
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)

    def test_non_integer_ids(self):
        p = _plan(1)
        p["result"]["placements"][0]["tube_id"] = 1.5
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)
        p = _plan(1)
        p["result"]["placements"][0]["tube_id"] = True
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)
        p = _plan(1)
        p["result"]["slot_state"][2]["slot"] = "x"
        with self.assertRaises(ValueError):
            parse_plan_snapshot(p)


class TestPlannersRejectInvalidSnapshot(unittest.TestCase):
    """三个只读规划器: 快照无效时在搜索前拒绝(ValueError → HTTP 400)。"""

    def _corrupt_plans(self):
        """各类无效快照(占用矛盾 / 槽号重复 / 非有限角度 / 质量非正)。"""
        plans = []
        p = _plan(1)
        p["result"]["slot_state"][0]["tube_id"] = 999      # 占用矛盾
        plans.append(p)
        p = _plan(1)
        p["result"]["slot_state"].append(dict(p["result"]["slot_state"][4]))
        plans.append(p)                                     # 槽号重复
        p = _plan(1)
        p["result"]["slot_state"][2]["angle"] = float("nan")
        plans.append(p)                                     # 非有限槽角
        p = _plan(1)
        p["result"]["placements"][0]["mass"] = -3.0
        plans.append(p)                                     # 质量非正
        return plans

    def test_evacuation_rejects_before_search(self):
        for p in self._corrupt_plans():
            before = copy.deepcopy(p)
            with self.assertRaises(ValueError):
                plan_slot_evacuation(p, failed_slots=[3],
                                     u1_limit_g_mm=1e9, max_moves=3)
            self.assertEqual(p, before)  # 只读: 拒绝也不修改入参

    def test_counterweight_rejects_before_search(self):
        for p in self._corrupt_plans():
            with self.assertRaises(ValueError):
                plan_counterweight_compensation(
                    p, [{"mass_g": 5.0, "stock": 3}],
                    u1_limit_g_mm=1e9, max_weights=3)

    def test_spare_replacement_rejects_before_search(self):
        for p in self._corrupt_plans():
            with self.assertRaises(ValueError):
                plan_spare_replacement(
                    p, [1], [{"tube_id": 10, "mass_g": 50.0}],
                    u1_limit_g_mm=1e9)

    def test_snapshot_checked_before_search_bounds(self):
        # 快照损坏 + 超出搜索边界(11 管 > 10): 必须先暴露快照问题
        p = _plan(1, {i: (i + 1, 10.0, False) for i in range(11)}, slots_n=16)
        p["result"]["slot_state"][0]["tube_id"] = 999
        with self.assertRaises(ValueError) as ctx:
            plan_slot_evacuation(p, failed_slots=[15],
                                 u1_limit_g_mm=1.0, max_moves=1)
        self.assertIn("矛盾", str(ctx.exception))

    def test_valid_snapshot_results_unchanged(self):
        # 正常方案: 三个规划器结果与快照语义一致(基线 |U₁| 独立复算)
        p = _plan(5, {0: (1, 50.0, True), 6: (2, 51.0, False)})
        expect_base = 100.0 * abs(50.0 - 51.0)
        ev = plan_slot_evacuation(p, failed_slots=[3],
                                  u1_limit_g_mm=1e9, max_moves=0)
        self.assertTrue(ev["feasible"])
        self.assertAlmostEqual(ev["original_u1"]["u1_mag_g_mm"],
                               expect_base, places=6)
        cw = plan_counterweight_compensation(
            p, [{"mass_g": 1.0, "stock": 3}],
            u1_limit_g_mm=1e9, max_weights=0)
        self.assertAlmostEqual(cw["baseline"]["u1_mag_g_mm"],
                               expect_base, places=6)
        sp = plan_spare_replacement(
            p, [2], [{"tube_id": 10, "mass_g": 50.0}], u1_limit_g_mm=1e9)
        self.assertTrue(sp["feasible"])
        self.assertAlmostEqual(sp["baseline"]["u1_mag_g_mm"],
                               expect_base, places=6)
        self.assertAlmostEqual(sp["final"]["u1_mag_g_mm"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
