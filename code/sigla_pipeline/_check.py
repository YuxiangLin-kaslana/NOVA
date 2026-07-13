import sys; sys.path.insert(0,'/u/ylin30/sigLA/code')
import numpy as np, sigla_exp.ovbench as CB, sigla_pipeline.env as ENV, sigla_pipeline.profile_naive as PF
rng=np.random.default_rng(1); samp,_=PF.normal_stats(rng)
def avg(kind,n=60):
    R=[];N=[];prim={}
    for _ in range(n):
        if kind=='spike_train':
            x=CB.base_normal(rng)
            for _ in range(int(rng.integers(6,12))): CB.INJ['spike'](x,rng)
        else: x=CB.make_window_strength('variance_burst',rng,1.0)
        pr=PF.disentangle(x,samp); R.append(pr['z']['variance_burst']); N.append(pr['net']['variance_burst']); prim[pr['primary']]=prim.get(pr['primary'],0)+1
    print(f"{kind:14s} raw_var={np.mean(R):4.1f} net_var={np.mean(N):4.1f} 剥离={1-np.mean(N)/(np.mean(R)+1e-6):.0%} primary={prim}")
print("纠缠验证(spike串 raw_var 假高→去纠缠 net_var 剥离;真var-burst 不剥离):")
avg('spike_train'); avg('variance_burst')
