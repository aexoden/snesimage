# snesimage TODO

The following is a list of possible improvements or changes for the future.

## Interface

* The interface needs a complete overhaul. Two unlabeled buttons is not very
  user friendly.
* The interface should not hang in between optimization iterations.

## Optimization

The unique challenge this program attempts to solve is two-fold:

* The SNES generally divides its backgrounds into tiles, each of which can
  select from a number of smaller palettes. (Either 4, 8 or 16 colors depending
  on the mode and background layer. There are 256 color modes, but are not
  necessarily appropriate for all situations, so being able to optimize for the
  other modes is desirable.) I have been unable to locate any other optimizers
  that can handle this tiling.
* In addition, most other optimizers, while being capable of dithering, seem to
  not take that dithering into account when generating the optimized palette.
  For instance, an image with three zones of white, gray and black, when reduced
  to 2 colors, will be reduced to black and a lighter gray, instead of taking
  the dithering into account and choosing white and black. When working with
  very small palettes, this is a notable detriment.

Ideas for changes or improvements to the optimization include:

* Optionally start from completely empty palettes and tile assignments and let
  the algorithm work from there. This probably wouldn't actually improve
  anything, but it might be fun to observe.
* Find a much better optimization algorithm. The ultimate results are good, but
  the process can be very time consuming. In addition, it's not guaranteed to
  find the global optimum, and can easily get stuck in local optima.
* Currently, no attempt is made to reassign tiles dynamically if it could
  improve the overall result. The initial guess is probably not optimal.
* If the optimization cannot be made much faster, being able to resume from a
  previous run is desirable.
