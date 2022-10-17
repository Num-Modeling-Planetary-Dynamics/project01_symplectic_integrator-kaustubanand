# Project 1 - Symplectic Integrator - Notes
###### *By Kaustub Anand*


I have submitted the final code and initial conditions data used for the Symplectic Integrator project. I used the Democratic Heliocentric Method. It is set up to run automatically and is set up to use $AU-Days-M_{sun}$ as the units. The arrays defined as [ [[X, Y, Z], [X, Y, Z],...], [[...], [...], ...], [[...], [...], ...] ] with a dimensions/shape as (time-step, planet, cartesian coordinate).

Below are the main issues I ran into, and why i believe the code doesn't work well:

- The Heliocentric to Barycentric origin calculation and thus conversion might be incorrect.
- The Cartesian to Orbital element conversion function (*xy_to_el()*) is incorrect. It does not calculate the semi-major axis ($a$) and eccentricity ($e$) correctly. It throws out a factor of ~2 and leads to NaNs in those two steps. As a result, the rest of the code also goes wrong.

Here is one note to improve the code:
- I could redefine the arrays as [[X1, X2, X3, ... XN], [Y1, Y2, Y3, ... YN], ....] as this helps make the array multiplications easier and avoids any need for transposing them.