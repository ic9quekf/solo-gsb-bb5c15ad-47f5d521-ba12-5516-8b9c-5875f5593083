"""配平引擎回归测试(仅依赖标准库): python -m unittest discover -s tests"""
import json
import math
import unittest

from app.balancer import run_plan


def base_payload(**over):
    p = {
        "slot_count": 12,
        "slots": [{"index": i, "angle": i * 30, "enabled": True} for i in range(12)],
        "radius_mm": 100.0,
        "rpm": 4000,
        "restarts": 6,
        "chain_steps": 900,
    }
    p.update(over)
    return p


class TestBalancer(unittest.TestCase):
    def six_tubes(self, **over):
        p = base_payload(**over)
        p["tubes"] = [
            {"id": 1, "mass": 50}, {"id": 2, "mass": 50},
            {"id": 3, "mass": 51}, {"id": 4, "mass": 49},
            {"id": 5, "mass": 30}, {"id": 6, "mass": 30},
        ]
        return p

    def test_uniform_12_slot_balance(self):
        r = run_plan(self.six_tubes())
        m = r["metrics"]
        # 该数据集的一阶最优约 3.85 g·mm; 启发式给出良好近似
        self.assertLess(m["u1_mag_g_mm"], 600.0)
        self.assertEqual(len(r["placements"]), 6)
        slots_used = [p["slot"] for p in r["placements"]]
        self.assertEqual(len(slots_used), len(set(slots_used)))

    def test_equal_masses_perfect_balance(self):
        p = base_payload()
        p["tubes"] = [{"id": i, "mass": 50} for i in range(1, 5)]
        r = run_plan(p)
        self.assertAlmostEqual(r["metrics"]["u1_mag_g_mm"], 0.0, places=6)
        self.assertAlmostEqual(r["metrics"]["u2_mag_g_mm"], 0.0, places=6)
        self.assertAlmostEqual(r["metrics"]["centrifugal_force_n"], 0.0, places=6)

    def test_three_equal_tubes_theoretical_min(self):
        p = base_payload()
        p["tubes"] = [{"id": i, "mass": 40} for i in range(1, 4)]
        r = run_plan(p)
        # 三支等质量在 30° 栅格 12 槽上最小一阶残余 = 40g*100mm = 4000 g·mm
        self.assertLessEqual(r["metrics"]["u1_mag_g_mm"], 4000.001)

    def test_disabled_slots_never_used(self):
        p = self.six_tubes()
        p["slots"][6]["enabled"] = False
        p["slots"][0]["enabled"] = False
        r = run_plan(p)
        for pl in r["placements"]:
            self.assertNotIn(pl["slot"], (0, 6))
        for ss in r["slot_state"]:
            if not ss["enabled"]:
                self.assertIsNone(ss["tube_id"])

    def test_locked_tube_respected(self):
        p = self.six_tubes()
        p["tubes"][0].update(locked=True, slot=0)
        r = run_plan(p)
        self.assertEqual(next(x for x in r["placements"] if x["tube_id"] == 1)["slot"], 0)

    def test_locked_on_disabled_raises(self):
        p = self.six_tubes()
        p["tubes"][0].update(locked=True, slot=6)
        p["slots"][6]["enabled"] = False
        with self.assertRaises(ValueError):
            run_plan(p)

    def test_too_many_tubes_raises(self):
        p = self.six_tubes()
        p["tubes"] = p["tubes"] * 3
        with self.assertRaises(ValueError):
            run_plan(p)

    def test_incremental_preserves_unmoved(self):
        p0 = self.six_tubes()
        p0["tubes"][0].update(locked=True, slot=0)
        r0 = run_plan(p0)
        p1 = json.loads(json.dumps(p0))
        p1["tubes"].append({"id": 7, "mass": 20})
        p1["previous"] = [
            {"tube_id": x["tube_id"], "slot": x["slot"]} for x in r0["placements"]
        ]
        r1 = run_plan(p1)
        before = {x["tube_id"]: x["slot"] for x in r0["placements"]}
        after = {x["tube_id"]: x["slot"] for x in r1["placements"]}
        moved = set(r1["moved_tube_ids"])
        for tid, slot in before.items():
            if tid != 1 and tid not in moved:
                self.assertEqual(after[tid], slot)
        # 新增试管已摆放
        self.assertIsNotNone(after.get(7))

    def test_incremental_remove(self):
        p0 = self.six_tubes()
        r0 = run_plan(p0)
        p1 = json.loads(json.dumps(p0))
        p1["tubes"] = [t for t in p1["tubes"] if t["id"] != 2]
        p1["previous"] = [
            {"tube_id": x["tube_id"], "slot": x["slot"]} for x in r0["placements"]
        ]
        r1 = run_plan(p1)
        self.assertEqual(r1["total_tubes"], 5)

    def test_deterministic(self):
        p = self.six_tubes()
        a = run_plan(json.loads(json.dumps(p)))
        b = run_plan(json.loads(json.dumps(p)))
        la = [(x["tube_id"], x["slot"]) for x in a["placements"]]
        lb = [(x["tube_id"], x["slot"]) for x in b["placements"]]
        self.assertEqual(la, lb)

    def test_replay_steps_consistent(self):
        r = run_plan(self.six_tubes())
        self.assertTrue(r["steps"])
        last = r["steps"][-1]
        self.assertAlmostEqual(last["vectors"]["u1_mag"], r["metrics"]["u1_mag_g_mm"], places=6)
        self.assertAlmostEqual(last["vectors"]["u2_mag"], r["metrics"]["u2_mag_g_mm"], places=6)
        for st in r["steps"]:
            self.assertEqual(len(st["placements"]), 6)
            self.assertIn("reason", st)

    def test_metric_relationships(self):
        p = base_payload(rpm=6000, radius_mm=120)
        p["tubes"] = [{"id": 1, "mass": 100}, {"id": 2, "mass": 100}]
        r = run_plan(p)
        m = r["metrics"]
        omega = 2 * math.pi * 6000 / 60
        # F = omega^2 * |U1| * 1e-6
        self.assertAlmostEqual(m["centrifugal_force_n"],
                               omega ** 2 * m["u1_mag_g_mm"] * 1e-6, places=6)
        self.assertAlmostEqual(m["resultant_moment_n_m"],
                               m["centrifugal_force_n"] * 0.120, places=6)

    def test_irregular_angles(self):
        p = {
            "slot_count": 4,
            "slots": [
                {"index": 0, "angle": 0}, {"index": 1, "angle": 95},
                {"index": 2, "angle": 180}, {"index": 3, "angle": 272},
            ],
            "radius_mm": 100, "rpm": 3000,
            "restarts": 4, "chain_steps": 600,
        }
        p["tubes"] = [{"id": 1, "mass": 100}, {"id": 2, "mass": 100}]
        r = run_plan(p)
        # 最接近对径的是 0/180
        slots = sorted(x["slot"] for x in r["placements"])
        self.assertEqual(slots, [0, 2])

    def test_shuffled_slots_solved_by_index(self):
        # 槽位按任意顺序提交, 求解器必须按编号(index)而不是列表下标取角度。
        p = {
            "slot_count": 4,
            "slots": [
                {"index": 3, "angle": 270}, {"index": 0, "angle": 0},
                {"index": 2, "angle": 180}, {"index": 1, "angle": 90},
            ],
            "radius_mm": 100, "rpm": 4000, "restarts": 4, "chain_steps": 300,
        }
        # 锁定管在槽 0: 必须按槽 0 的 0° 计算 U1=(+mR, 0), 而非列表首位槽 3 的 270°
        p["tubes"] = [{"id": 1, "mass": 50, "locked": True, "slot": 0}]
        r = run_plan(p)
        pl = next(x for x in r["placements"] if x["tube_id"] == 1)
        self.assertEqual(pl["slot"], 0)
        self.assertAlmostEqual(pl["angle"], 0.0)
        self.assertAlmostEqual(pl["u1"]["x"], 5000.0, places=6)
        self.assertAlmostEqual(pl["u1"]["y"], 0.0, places=6)
        # 落库几何与 slot_state 均按槽号 0..n-1 排列
        self.assertEqual([s["index"] for s in p["slots"]], [0, 1, 2, 3])
        self.assertEqual(p["slot_count"], 4)
        self.assertEqual([s["slot"] for s in r["slot_state"]], [0, 1, 2, 3])

    def test_shuffled_slots_optimal_matches_ordered(self):
        # 同一几何乱序与顺序提交的最优指标必须一致; 等质量对应落对径槽 0/2。
        def payload(slots):
            return {
                "slot_count": 4,
                "slots": slots,
                "tubes": [{"id": 1, "mass": 100, "locked": True, "slot": 2},
                          {"id": 2, "mass": 100}],
                "radius_mm": 100, "rpm": 3000,
                "restarts": 6, "chain_steps": 600, "seed": 7,
            }
        ordered = [{"index": i, "angle": a}
                   for i, a in enumerate((0, 95, 180, 272))]
        shuffled = [ordered[i] for i in (3, 1, 0, 2)]
        r1 = run_plan(payload(ordered))
        r2 = run_plan(payload(json.loads(json.dumps(shuffled))))
        self.assertAlmostEqual(r1["metrics"]["u1_mag_g_mm"],
                               r2["metrics"]["u1_mag_g_mm"], places=6)
        loc2 = {x["tube_id"]: x["slot"] for x in r2["placements"]}
        self.assertEqual(loc2, {1: 2, 2: 0})

    def test_shuffled_angles_drive_ring_adjacency(self):
        # 物理相邻关系按实际角度排序, 与提交顺序、槽号差无关。
        p = {
            "slot_count": 4,
            "slots": [
                {"index": 3, "angle": 250}, {"index": 2, "angle": 120},
                {"index": 1, "angle": 10}, {"index": 0, "angle": 350},
            ],
            "tubes": [
                {"id": 1, "mass": 50, "locked": True, "slot": 0},
                {"id": 2, "mass": 50}, {"id": 3, "mass": 50},
            ],
            "radius_mm": 100, "rpm": 4000, "restarts": 4, "chain_steps": 300,
            "mutex_pairs": [{"tube_a": 1, "tube_b": 2}],
        }
        r = run_plan(p)
        # 角序 槽1(10)-槽2(120)-槽3(250)-槽0(350): 槽0 的物理邻槽是 {1,3}
        edges = {tuple(sorted(e)) for e in r["exact_search"]["adjacent_slot_pairs"]}
        self.assertEqual(edges, {(0, 1), (0, 3), (1, 2), (2, 3)})
        loc = {x["tube_id"]: x["slot"] for x in r["placements"]}
        self.assertNotIn(loc[2], (1, 3))
        self.assertEqual(loc[2], 2)

    def test_default_angles_by_index_when_omitted(self):
        p = {
            "slot_count": 6,
            # 乱序、省略或显式置空 angle: 默认角度按槽号 360*index/n 生成
            "slots": [{"index": 5}, {"index": 2, "enabled": True},
                      {"index": 0, "angle": None}, {"index": 4},
                      {"index": 1}, {"index": 3}],
            "radius_mm": 100, "rpm": 4000, "restarts": 4, "chain_steps": 300,
        }
        p["tubes"] = [{"id": 1, "mass": 10}]
        run_plan(p)
        angles = {s["index"]: s["angle"] for s in p["slots"]}
        for i in range(6):
            self.assertAlmostEqual(angles[i], 60.0 * i)

    def test_duplicate_missing_and_nonfinite_slots_rejected(self):
        base = {
            "slot_count": 4,
            "slots": [{"index": i, "angle": 90.0 * i} for i in range(4)],
            "radius_mm": 100, "rpm": 4000, "restarts": 2, "chain_steps": 100,
            "tubes": [{"id": 1, "mass": 10}],
        }
        dup = json.loads(json.dumps(base))
        dup["slots"][1]["index"] = 0
        with self.assertRaises(ValueError):
            run_plan(dup)
        missing = json.loads(json.dumps(base))
        missing["slots"] = [s for s in missing["slots"] if s["index"] != 2]
        with self.assertRaises(ValueError):
            run_plan(missing)
        oob = json.loads(json.dumps(base))
        oob["slots"][3]["index"] = 4
        with self.assertRaises(ValueError):
            run_plan(oob)
        for bad in (float("nan"), float("inf"), float("-inf")):
            nf = json.loads(json.dumps(base))
            # NaN/Inf 经 JSON 往返会变成 null, 这里直接构造载荷
            nf2 = dict(nf)
            nf2["slots"] = [
                {"index": i, "angle": (bad if i == 1 else 90.0 * i)}
                for i in range(4)
            ]
            with self.assertRaises(ValueError):
                run_plan(nf2)

    def test_pair_and_singleton_explanations(self):
        r = run_plan(self.six_tubes())
        for pr in r["pairs"]:
            self.assertIn("reason", pr)
            self.assertGreaterEqual(pr["u1_cancel_pct"], 0.0)
        for sg in r["singletons"]:
            self.assertIn("reason", sg)


class TestNumericBoundary(unittest.TestCase):
    """数值边界: 非有限输入与计算溢出必须抛 ValueError(API 层转为 400 且不保存)。

    正常量级(含很大但不溢出的数值)的求解行为与指标语义不变。
    """

    def two_tubes(self, **over):
        p = base_payload(tubes=[{"id": 1, "mass": 50}, {"id": 2, "mass": 50}])
        p.update(over)
        return p

    def test_nonfinite_tube_mass_rejected(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            p = self.two_tubes()
            p["tubes"][0]["mass"] = bad
            with self.assertRaises(ValueError, msg=f"mass={bad}"):
                run_plan(p)

    def test_nonfinite_params_rejected(self):
        for key in ("radius_mm", "rpm", "w1", "w2", "move_weight"):
            for bad in (float("inf"), float("-inf"), float("nan")):
                p = self.two_tubes(**{key: bad})
                with self.assertRaises(ValueError, msg=f"{key}={bad}"):
                    run_plan(p)

    def test_sign_rules_unchanged(self):
        # 有限但符号非法的输入沿用原有拒绝语义
        for key, bad in (("radius_mm", 0.0), ("radius_mm", -1.0), ("rpm", -1.0)):
            p = self.two_tubes(**{key: bad})
            with self.assertRaises(ValueError, msg=f"{key}={bad}"):
                run_plan(p)
        p = self.two_tubes()
        p["tubes"][0]["mass"] = 0.0
        with self.assertRaises(ValueError):
            run_plan(p)

    def test_huge_mass_radius_product_overflow(self):
        # 质量×半径(质径积上界)溢出
        with self.assertRaises(ValueError):
            run_plan(self.two_tubes(radius_mm=1e308))
        p = self.two_tubes(tubes=[{"id": 1, "mass": 1e200}, {"id": 2, "mass": 1e200}],
                           radius_mm=1e200)
        with self.assertRaises(ValueError):
            run_plan(p)

    def test_huge_mass_objective_overflow(self):
        # 总质量 / 质量平方(目标函数上界)溢出: 即使权重为 0 也会在评分中溢出
        p = self.two_tubes(tubes=[{"id": 1, "mass": 1e308}, {"id": 2, "mass": 1e308}])
        with self.assertRaises(ValueError):
            run_plan(p)
        p = self.two_tubes(tubes=[{"id": 1, "mass": 1e200}, {"id": 2, "mass": 1e200}],
                           w1=0.0, w2=0.0)
        with self.assertRaises(ValueError):
            run_plan(p)
        # 权重极大: 评分静默溢出为 inf
        p = self.two_tubes(w1=1e308,
                           tubes=[{"id": 1, "mass": 1e100}, {"id": 2, "mass": 1e100}])
        with self.assertRaises(ValueError):
            run_plan(p)

    def test_huge_rpm_metric_overflow(self):
        # 转速极大: 角速度/离心力指标链溢出
        for bad_rpm in (1e308, 1e200):
            with self.assertRaises(ValueError, msg=f"rpm={bad_rpm}"):
                run_plan(self.two_tubes(rpm=bad_rpm))

    def test_overflow_rejected_on_exact_mutex_path(self):
        p = self.two_tubes(
            tubes=[{"id": i, "mass": 1e308} for i in range(1, 4)],
            mutex_pairs=[{"tube_a": 1, "tube_b": 2}],
        )
        with self.assertRaises(ValueError):
            run_plan(p)

    def test_large_but_safe_values_still_solve(self):
        # 很大但各环节均不溢出的数值: 正常求解且全部派生指标有限
        p = base_payload(
            radius_mm=1e4, rpm=1e6,
            tubes=[{"id": 1, "mass": 1e6}, {"id": 2, "mass": 1e6},
                   {"id": 3, "mass": 9.99e5}, {"id": 4, "mass": 1.001e6}],
        )
        r = run_plan(p)
        m = r["metrics"]
        for key in ("u1_mag_g_mm", "u2_mag_g_mm", "centrifugal_force_n",
                    "resultant_moment_n_m", "equivalent_eccentricity_um",
                    "angular_velocity_rad_s", "total_mass_g"):
            self.assertTrue(math.isfinite(m[key]), key)
        self.assertEqual(len(r["placements"]), 4)
        # 力学关系保持原有语义: F = ω²·|U₁|·1e-6, M = F·R/1000
        omega = 2 * math.pi * 1e6 / 60
        self.assertAlmostEqual(m["centrifugal_force_n"],
                               omega ** 2 * m["u1_mag_g_mm"] * 1e-6,
                               delta=max(1.0, m["centrifugal_force_n"] * 1e-9))
        self.assertAlmostEqual(m["resultant_moment_n_m"],
                               m["centrifugal_force_n"] * 1e4 / 1000.0,
                               delta=max(1.0, m["resultant_moment_n_m"] * 1e-9))


if __name__ == "__main__":
    unittest.main()
