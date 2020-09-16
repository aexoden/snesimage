snesimage
=========

snesimage is a utility for optimizing a given image for display on the SNES. It
currently uses a combination of a k-means clustering algorithm to determine the
initial palette, as well as a stochastic iteration algorithm to iteratively
improve the palette. The second phase is not very advanced and may or may not
actually improve the result.

Usage
-----

snesimage currently expects to be given a 256x256 source image. Other sizes will
likely do very strange things.

The following command-line options are available:

* `--subpalette-count` determines the number of subpalettes to use.
* `--subpalette-size` determines the size of each subpalette, not including the
  transparent color. As a result, values of 3, 7 and 15 are most likely to be
  useful.
* `--p-delta` determines the rate of decrease of the probability that a worse
  result will be accepted during the stochastic optimization phase. To get
  decent results out of this phase, a value of 0.0001 or less is recommended. To
  only allow improvements, use a value such as 1.0.
* `--dither` enables dithering of the result.

The results are not yet completely satisfactory in all cases. Once the program
opens, each tile is initially assigned to a palette. By clicking on a given
tile, you can change the palette it's assigned to. Clicking the green button
will begin the stochastic optimization phase.

There is currently no way to actually output the results in a useful form.
