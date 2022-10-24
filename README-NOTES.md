# Project 1 - Symplectic Integrator - Notes
###### *By Kaustub Anand*


I have submitted the final code and initial conditions data used for the Symplectic Integrator project. I used the Democratic Heliocentric Method. It is set up to run automatically and is set up to use $AU-Days-M_{sun}$ as the units. The arrays defined as [[X1, X2, X3, ... XN], [Y1, Y2, Y3, ... YN], ....] with a dimensions/shape as (time-step, cartesian coordinate, planet).

The plots submitted on Brightspace are of the Energy error values and Resonance Angle until the code breaks down into `NaNs` after a ~100 Earth Years.

Below are the main issues I ran into, and why i believe the code doesn't work well:

- The Heliocentric to Barycentric origin calculation and thus conversion is incorrect. It leads to abnormlly large values for the momentum.
  - I have avoided this by keeping a separate barycentric momentum array for calculations.
- The Danby $f$ and $g$ functions are calculating the position vector ($r$) with a high error. 
  - While the magnitude of other quantities stay almost the same, the position vector update statement leads to a rapid increase in the magnitude of the $r$ vector. 
  - This leads to a subsequent incorrect calculation for the eccentricity of Neptune that goes over 1 and thus gives `NaN` values. This further gives `NaN` values for other quantities.

