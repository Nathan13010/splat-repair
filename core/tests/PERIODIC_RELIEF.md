# Periodic relief benchmark

The benchmark runs on the base Python dependencies. It builds each intact
reference scan in memory and evaluates the repair with and without lattice
candidates. An optional trial reads surviving splats in the same tangent chart,
checks repeated shifts on separate sectors, and uses agreeing donor heights to
adjust the generated face. The repair engine is called normally for all three
comparisons.

From the repository root, with the engine environment activated:

```powershell
python core/tests/bench_periodic_relief.py --experiment --repeats 3
python core/tests/bench_periodic_relief.py --holdout --experiment --repeats 3 --save-images data/relief-holdout
python core/tests/bench_periodic_relief.py --validation --experiment --repeats 3 --save-images data/relief-validation
```

Each line of stdout is a JSON result. Use `--scene` to select specific cases.
The image runs save the intact reference, the empty hole and each repair at
the same framing in three views.

- The development set includes the audit's straight and bent relief at two
  periods with uniform DC and the audit's textured physical holes.
- The holdout set changes the radius, period and scene seed. Its controls
  include a dome, two shells, a free edge and a material junction.
- The validation set rotates and scales the ripples and adds one surface that
  warps both periodic directions.
- Height error uses newly created face splats inside the disk. Backing and
  reseated gap copies are excluded. Coverage is the fraction of reference
  points in the disk within 0.12 XY units of a new face splat. The slope score
  compares locally fitted normals to the derivative of the reference height.
- The image score is RGB mean absolute error divided by 255, within the
  projected area of the removed reference points. Rendering includes backing
  and reseating. Times cover repair and, where enabled, trial processing.

Pass selected indices as the application does. In the audited engine, passing
a boolean mask to the unpainted-volume path can pollute `_close_gaps` because
it converts the mask to integers before checking its dtype. Seed 16 exposes
this with a large number of reseated copies. The benchmark also prints the
reseated count to make this behavior visible.

## Interpretation

The height trial evaluates geometry offline. It adjusts the generated face
along the chart normal using independent height measurements. It acts on
single-chart patches and excludes unfolded repairs, reseated copies and
backing. A production candidate would need to choose a phase-matched donor
and turn its covariance and spherical harmonics with the local surface.
Review that candidate from multiple views on a shareable real scan.


## Windows benchmark results

Python 3.13.7, NumPy 2.5.5 and SciPy 1.18.1. The table reports the 95th
percentile absolute height error in scene units.

| Scene | Seed | Current lattice | Lattice disabled | Height trial | Trial decision |
| --- | ---: | ---: | ---: | ---: | --- |
| curved-holdout | 3 | 0.0671 | 0.0566 | 0.0232 | Applied |
| curved-holdout | 4 | 0.4682 | 0.4762 | 0.4682 | Fallback |
| curved-holdout | 5 | 0.5139 | 0.5139 | 0.5139 | Fallback |
| curved-holdout | 6 | 0.1186 | 0.1022 | 0.0590 | Applied |
| curved-holdout | 7 | 0.5366 | 0.5472 | 0.0739 | Applied |
| rotated-curved | 12 | 0.0997 | 0.0900 | 0.0564 | Applied |
| rotated-curved | 13 | 0.5281 | 0.5369 | 0.0275 | Applied |
| rotated-curved | 14 | 0.1938 | 0.1808 | 0.0428 | Applied |
| rotated-curved | 15 | 0.0698 | 0.0705 | 0.0297 | Applied |
| rotated-curved | 16 | 0.4542 | 0.4542 | 0.4542 | Fallback |
| two-axis-warp | 17 | 0.4788 | 0.4871 | 0.4788 | Fallback |

The trial improved seven of ten held-out one-axis cases by 30% or more when
applied. It fell back on the three cases that entered the engine's unfold
pass and on the two-axis case, where the sector checks did not find reliable
repeats. The largest coverage loss was 0.09 percentage points. The worst
oblique error for each measured scene matched or improved on the current
repair. Across the seven adjusted held-out cases, the median of each case's
relative change in median repair time was 7.8%.

Keep this trial offline until full donor and spherical-harmonic transport
is validated. The two-axis case still needs a candidate that can follow
local changes in both directions.
