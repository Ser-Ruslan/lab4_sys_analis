"""Проверки математического результата и граничных состояний."""
import unittest
import numpy as np
from lab5_model import DEFAULT_A, DEFAULT_B, DEFAULT_C, solve, sensitivity


class ProductionTests(unittest.TestCase):
    def test_optimality_certificate(self):
        r = solve()
        self.assertEqual(r.status, 'OPTIMAL')
        self.assertAlmostEqual(r.value, 1020)
        self.assertTrue(np.all(np.array(DEFAULT_A) @ r.x <= np.array(DEFAULT_B) + 1e-7))
        self.assertTrue(np.all(r.x >= 0))
        # Двойственная допустимость и совпадение значений подтверждают оптимальность.
        self.assertTrue(np.all(np.array(DEFAULT_A).T @ r.shadow >= np.array(DEFAULT_C) - 1e-7))
        self.assertAlmostEqual(float(np.array(DEFAULT_B) @ r.shadow), r.value)

    def test_alternative_optimum(self):
        for t in (60, 90, 120, 156):
            x = np.array([260-5*t/3, 80+t/3, t])
            self.assertTrue(np.all(np.array(DEFAULT_A) @ x <= np.array(DEFAULT_B) + 1e-7))
            self.assertAlmostEqual(float(np.array(DEFAULT_C) @ x), 1020)

    def test_sensitivity(self):
        rows = sensitivity(DEFAULT_A, DEFAULT_B, DEFAULT_C)
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertAlmostEqual(row['gain'], row['delta'] if row['resource'] < 3 else 0)

    def test_unbounded(self):
        self.assertEqual(solve([[0]], [1], [1]).status, 'UNBOUNDED')

    def test_zero_stock(self):
        self.assertAlmostEqual(solve(DEFAULT_A, [0, 0, 0], DEFAULT_C).value, 0)

    def test_invalid_data(self):
        for A, b, c in [([[1]], [-1], [1]), ([[1]], [1], [float('nan')]), ([[1, 2]], [1], [1])]:
            with self.assertRaises(ValueError):
                solve(A, b, c)


if __name__ == '__main__':
    unittest.main()
