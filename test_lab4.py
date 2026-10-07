import math
import unittest
from dataclasses import replace
from lab4_model import Parameters, VARIANTS, calculate, costs, inventory_points, schedule


class WilsonTests(unittest.TestCase):
    def test_variant1_values(self):
        r = calculate(Parameters())
        self.assertAlmostEqual(r.Q, 894.4271909999159)
        self.assertAlmostEqual(r.N, 8.94427190999916)
        self.assertAlmostEqual(r.T, 40.80824058937116)
        self.assertAlmostEqual(r.ROP, 87.67123287671232)
        self.assertAlmostEqual(r.total, 964472.1359549996)
        self.assertEqual(r.integer_Q, 894)

    def test_all_variants_are_minima(self):
        for values in VARIANTS.values():
            p = Parameters(*values)
            r = calculate(p)
            self.assertAlmostEqual(r.ordering, r.holding)
            self.assertLess(r.total, costs(p, r.Q * .9)[3])
            self.assertLess(r.total, costs(p, r.Q * 1.1)[3])
            self.assertLessEqual(r.integer_total, costs(p, max(1, r.integer_Q-1))[3])
            self.assertLessEqual(r.integer_total, costs(p, r.integer_Q+1)[3])

    def test_purchase_does_not_change_optimum(self):
        p = Parameters()
        free = calculate(replace(p, Cp=0))
        paid = calculate(p)
        self.assertEqual(free.Q, paid.Q)
        self.assertEqual(free.management, paid.management)
        self.assertAlmostEqual(paid.total-free.total, p.D*p.Cp)

    def test_stock_and_delivery(self):
        p = Parameters()
        r = calculate(p)
        points = inventory_points(p, r)
        self.assertEqual(len(points), 7)
        self.assertEqual(points[-1][0], 3*r.T)
        for i in range(1, 7, 2):
            self.assertEqual(points[i][0], points[i+1][0])
            self.assertEqual(points[i][1], 0)
            self.assertEqual(points[i+1][1], r.Q)
        for row in schedule(p, r):
            self.assertAlmostEqual(row['arrival_day']-row['order_day'], p.L)
            self.assertAlmostEqual(r.Q-r.daily_demand*(row['order_day']-(row['cycle']-1)*r.T), r.ROP)

    def test_invalid_parameters(self):
        for key in ('D', 'Co', 'Ch', 'L', 'Cp'):
            for value in [-1, float('nan'), float('inf')]:
                with self.assertRaises(ValueError):
                    calculate(replace(Parameters(), **{key: value}))
        for key in ('D', 'Co', 'Ch'):
            with self.assertRaises(ValueError):
                calculate(replace(Parameters(), **{key: 0}))
        for q in (0, -10, math.nan):
            with self.assertRaises(ValueError):
                costs(Parameters(), q)

    def test_zero_lead_and_long_lead(self):
        p = replace(Parameters(), L=0, Cp=0)
        r = calculate(p)
        self.assertEqual(r.ROP, 0)
        self.assertEqual(r.purchase, 0)
        self.assertEqual(schedule(p, r)[0]['order_day'], r.T)
        long = replace(p, L=100)
        result = calculate(long)
        self.assertGreater(result.ROP, result.Q)
        self.assertLess(schedule(long, result)[0]['order_day'], 0)


if __name__ == '__main__':
    unittest.main()
