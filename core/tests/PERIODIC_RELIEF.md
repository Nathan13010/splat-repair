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
and turn its covariance and spherical harmonics with the local surface. Review that candidate from
multiple views on a shareable real scan.
