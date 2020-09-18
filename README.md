snesimage
=========

snesimage is a utility for optimizing a given image for display on the SNES. It
currently uses a combination of a k-means clustering algorithm to determine the
initial palette, with a iterative optimization algorithm to improve the
generated palette.

Usage
-----

snesimage currently expects to be given a 256x256 source image. Other sizes will
likely do very strange things, if they work at all.

The following command-line options are available:

* `--subpalette-count` determines the number of subpalettes to use.
* `--subpalette-size` determines the size of each subpalette, not including the
  transparent color. As a result, values of 3, 7 and 15 are most likely to be
  useful.
* `--dither` enables dithering of the result.

The program takes two mandatory arguments: an image to optimize, and the
filename you wish any JSON output to be written to.

After starting the program, it will begin in tile assignment mode, assigning
each tile to a palette index. The default assignment is often fine, but if you
wish to tweak it, you may do so by clicking on the tile in question to switch it
to the next available palette. (The colors of the tiles reflect the average
color of the originally chosen tiles using that palette.)

Once the tiles are set to your satisfaction, click the green button to assign
the initial palettes. Depending on your image, this assignment may be completely
satisfactory. If so, you may press the blue button to immediately write the
current state to the output file. Otherwise, press the green button again to
begin the optimization phase.

The optimization phase will run indefinitely, though it will generally stop
improving the result within a few minutes. At any point, you can press the
blue button to output the current state.

Output Format
-------------

The output format is currently a rather simple JSON document, with three
separate arrays. The first, named `tiles`, is an array of arrays, each of which
contains the palette number (within each subpalette, therefore ranging from 0 to
15) used by that pixel of the tile. The `palette` section gives the full palette
entries, automatically expanding each subpalette to the full 16 colors
(including the transparent color at index zero). The entries are already
converted to the SNES palette format and are given as 16-bit unsigned integers.
The final section, `tile_palettes`, provides the palette index used by each
tile, numbered starting from zero.

Notes
------------

Any fully transparent pixel in the original image will remain transparent in the
final output. Partially transparent pixels will have their transparency silently
ignored.

No attempt is made to consolidate identical tiles, and each of the original
tiles is output in its entirety.
