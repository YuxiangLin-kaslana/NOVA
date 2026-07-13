import sys; sys.path.insert(0, '/u/ylin30/sigLA/code')
import numpy as np
import sota_compare.exp_mmd_earlywarning as M
ot, dates, rep = M.load()
for q in [0.70, 0.75, 0.80]:
    for H in [3, 4, 6]:
        thr = float(np.quantile(ot, q)); n = pos = withrep = wrpos = 0
        for i in range(M.K, len(ot) - H):
            if ot[i] > thr:
                continue
            lab = int(np.any(ot[i + 1:i + 1 + H] > thr))
            n += 1; pos += lab
            if M.recent_report(rep, dates[i]) is not None:
                withrep += 1; wrpos += lab
        print(f"q={q} H={H}: 全部点 n={n} 正例率={pos/max(1,n):.0%} | 有报告点={withrep} 其中正例={wrpos}")
