from sympy import symbols, solve, Eq

x = symbols('x')
equall = x**3 - 1
print(solve(equall, x))