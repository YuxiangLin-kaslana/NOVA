import sys; sys.path.insert(0,'/u/ylin30/sigLA/code')
import numpy as np
import sigla_exp.ovbench as CB
import sigla_pipeline.profile as P
rng=np.random.default_rng(0); mu,sd=CB.normal_stats(rng)
print("=== 纯概念:主导识别准确率 + 平均伴随响应数(正交benchmark应主导准、伴随少)===")
for c in P.CONCEPTS:
    ok=0; nco=0
    for _ in range(60):
        x=CB.make_window_strength(c,rng,1.0)
        pr=P.disentangle(x,mu,sd); ok+=int(pr["primary"]==c); nco+=len(pr["co_responses"])
    print(f"  {c:18s} 主导准确={ok/60:.0%}  平均伴随={nco/60:.2f}")
print("\n=== 双注入纠缠:spike(主) + 同窗叠一个 variance_burst,看去纠缠能否归因 ===")
for _ in range(3):
    x=CB.base_normal(rng); CB.INJ['spike'](x,rng); CB.INJ['variance_burst'](x,rng)
    pr=P.disentangle(x,mu,sd)
    print(f"  primary={pr['primary']}  prim_z={pr['prim_z']:.1f}  co_responses={ {k:round(v,2) for k,v in pr['co_responses'].items()} }")
    print(f"     校准分 z={ {k:round(v,1) for k,v in pr['z'].items()} }")
