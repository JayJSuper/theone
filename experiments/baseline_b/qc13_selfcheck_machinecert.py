"""R3 v3 machine-cert of Jack's Q-C13 self-check (causal-aware retrieval).
Certifies the ARITHMETIC only. Flags the conceptual muddiness (M2/Q are both
linear, yet Jack zeroes the structural-eq match for M2) as a design-tier caveat.
"""
import sys

Q = {("U", "X"), ("U", "Y"), ("X", "Y")}
M1 = {("U", "X"), ("U", "Y"), ("X", "Y")}
M2 = {("X", "Y")}
jac = lambda a, b: len(a & b) / len(a | b)

ok = True
def ck(name, got, exp, tol=1e-3):
    global ok
    p = abs(got - exp) <= tol
    ok = ok and p
    print(f"{'PASS' if p else 'FAIL'}  {name}: got {got:.4f} (Jack {exp})")

ck("Q13.1 Jaccard(M1,Q)", jac(M1, Q), 1.0)
ck("Q13.2 Jaccard(M2,Q)", jac(M2, Q), 0.333)
# causal sim = jaccard * struct_eq_match ; Jack: M1=1.0, M2=0 (struct mismatch)
s1 = 0.5 * 1.0 + 0.5 * 0.8           # combined, lambda=0.5, sem M1=0.8
s2 = 0.5 * 0.0 + 0.5 * 0.9           # sem M2=0.9
ck("Q13.3 combined M1", s1, 0.9)
ck("Q13.3 combined M2", s2, 0.45)
print(f"retrieval picks {'M1' if s1 > s2 else 'M2'} (Jack: M1) "
      f"{'PASS' if s1 > s2 else 'FAIL'}")
print("\nCAVEAT (design-tier): Jack sets struct_eq_match(M2)=0 calling it "
      "'linear vs none', but M2 and Q are both linear — the metric's brittle "
      "0/1 structural gate is under-specified. Arithmetic certified; metric NOT "
      "frozen-ready (matches his own self-audit 2.3).")
print("VERDICT:", "PASS (arithmetic)" if ok else "FAIL")
sys.exit(0 if ok else 1)
