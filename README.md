# TWIN-TOWER

Layer duplication detector: scan a stack of same-shape weight matrices and find out how many layers are actually doing the same job.

TWIN-TOWER flattens every layer, measures absolute cosine similarity between all pairs, and clusters near-duplicates into towers. The report shows the clusters, the number of towers, how many layers are redundant, and the most similar pair.

## What it does

- Cross-layer absolute cosine similarity matrix.
- Greedy clustering into towers of near-duplicates above a similarity tolerance.
- Redundancy rate: the fraction of layers that are duplicates.
- Most similar pair across the whole stack.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from twin_tower.scan import duplication_report

layers = [np.load(f) for f in ["layer_0.npy", "layer_1.npy", "layer_2.npy"]]
report = duplication_report(layers, cos_tol=0.9)
print("towers:", report["num_towers"])
print("redundancy:", report["redundancy_rate"])
```

## How it works

Layers that are scaled copies of each other (a very common form of duplication) have cosine similarity 1 in absolute value. The greedy clustering walks layers in order: the first unseen layer becomes a tower representative, and every later layer similar to it joins that tower.

## Honest scope

- Only same-shape layers can be compared; mixed stacks raise an error.
- Weight similarity does not prove functional duplication; use it as a signal, not a verdict.
- Clustering is greedy on similarity to the tower representative, not a full graph partition.

## License

MIT