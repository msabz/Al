#!/usr/bin/env python3
import numpy as np
import torch
from cpu_v10_onion_sparse_intersection_deepmind import SparseIntersectionOnion, _mutual_matches, BRANCHES, DEPTH

all_dist=[]
route=[]
for seed in range(24):
    torch.manual_seed(12000+seed)
    m=SparseIntersectionOnion()
    route.append(m.route_stats())
    for depth in range(DEPTH):
        clouds=[m._layer(c, depth).manifold_point().reshape(-1,3) for c in range(BRANCHES)]
        for i in range(BRANCHES):
            for j in range(i+1, BRANCHES):
                ds,_=_mutual_matches(clouds[i],clouds[j])
                all_dist.extend(float(d.detach()) for d in ds)

a=np.asarray(all_dist, dtype=np.float64)
qs={str(q):float(np.quantile(a,q)) for q in (0,0.001,0.005,0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90)}
counts={str(t):int((a<t).sum()) for t in (0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.035)}
print('MUTUAL_MATCH_COUNT', len(a))
print('DISTANCE_QUANTILES', qs)
print('DISTANCE_COUNTS', counts)
print('ROUTE_MEAN_AVG', float(np.mean([s['mean_route'] for s in route])))
print('ROUTE_GT025_MAX', max(s['edges_gt_0.25'] for s in route))
print('V10_DISTANCE_DIAGNOSTIC_PASS')
