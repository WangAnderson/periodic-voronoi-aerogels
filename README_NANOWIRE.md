# Periodic Voronoi nanowire display model

This generator complements the Voronoi sheet model in the same repository.
It turns the unique edges of a three-dimensional periodic Voronoi tessellation into
finite-diameter nanowires, keeps a connected three-axis-periodic random
subnetwork, and controls porosity from the physical union volume of the rods.

## Default model

- cubic unit cell: 1000 nm
- periodic Voronoi seeds: 400
- nanowire diameter: 20 nm
- target porosity: 95%
- absolute porosity tolerance: 0.5 percentage point
- deterministic random seed: 42

Nanowires are modelled as capsules for volume estimation: a cylindrical
centre-line segment with radius `rod_diameter_nm / 2` and rounded endpoints.
Overlaps at junctions are counted once.  Complete logical Voronoi edges are
selected, so the requested porosity is reached to a tolerance rather than by
shortening the final edge.

## Run

```powershell
python -m voronoi_nanowire.cli --config nanowire_config.json
```

The output directory contains a self-contained interactive HTML model, a PNG
preview, the exact configuration and JSON metadata.  The surface triangles in
the HTML are intended for visualisation; they are not a Boolean-unioned solver
or manufacturing mesh.

The static preview renders every surface triangle as an opaque, shaded solid;
it does not subsample individual faces.  Coordinate axes and text are hidden,
leaving a light unit-cell outline on a white background.  True Voronoi junctions
receive rounded nodes, while rods cut by the periodic cell boundary receive a
flat display cap rather than an artificial spherical endpoint.

The matched 95%, 97% and 99% examples use the same tessellation, rod diameter,
random seed, camera and lighting:

```powershell
python -m voronoi_nanowire.cli --config nanowire_configs/porosity_95.json
python -m voronoi_nanowire.cli --config nanowire_configs/porosity_97.json
python -m voronoi_nanowire.cli --config nanowire_configs/porosity_99.json
```

For an ad-hoc target, `--porosity 0.97 --output output/my_model` overrides the
configured porosity and destination without changing the other dimensions.

Important parameters in `nanowire_config.json` are:

- `cube_size_nm`
- `num_seeds`
- `rod_diameter_nm`
- `target_porosity`
- `porosity_tolerance`
- `random_seed`

The combination of seed count, diameter and porosity must be feasible for the
selected backbone and connected-growth strategy. The program reports the
range between that backbone and the complete reachable Voronoi edge network.
A failure is not a proof of global infeasibility across every possible edge
selection. The requested diameter and porosity are never changed silently.

For a binary STL of the same display mesh, add `--stl` to the command.  The
output is `nanowire_display_nm.stl` with numeric coordinates in nm.  Rods and
node spheres are not Boolean-unioned, so this STL is for visualisation and is
not guaranteed watertight, strictly clipped at the finite rod radius, or ready
for manufacturing/simulation. Porosity metadata describes the sampled periodic
capsule union, not a volume computed from the display STL.

See [the detailed Chinese guide](docs/nanowire_model_zh.md) for parameter editing,
Spyder use, figure generation and STL export instructions.
For direct editing and F5 execution in Spyder, open `run_nanowire_spyder.py`.
