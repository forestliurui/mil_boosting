import numpy as np

e_p = 0.5
e_n = 0.4
e_0 = 1 - e_p - e_n

alpha_prime = 2

part1 = (e_p+e_0*(np.exp(-alpha_prime)/(2.0*np.cosh(alpha_prime))   )  )*np.sqrt(e_n/e_p) 
part2 = (e_n+e_0*(np.exp(alpha_prime)/(2.0*np.cosh(alpha_prime))   )  )*np.sqrt(e_p/e_n)

print(part1)
print(part2)
print(part1+part2)

