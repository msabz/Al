#!/usr/bin/env python3
"""Export app-ready files from pinned official Google DeepMind mathematics_dataset linear_2d."""
import argparse, math, random, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import sympy as sp
REPO="https://github.com/google-deepmind/mathematics_dataset.git"
COMMIT="427f45075f84b8b9774950196ad63867ca20ffb3"
ROOT=Path("/tmp/deepmind-math-export")
def prepare():
    if ROOT.exists(): shutil.rmtree(ROOT)
    subprocess.run(["git","clone","-q",REPO,str(ROOT)],check=True);subprocess.run(["git","-C",str(ROOT),"checkout","-q",COMMIT],check=True)
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    if head!=COMMIT: raise RuntimeError("DeepMind source drift")
    sys.path.insert(0,str(ROOT));from mathematics_dataset.modules import algebra;return algebra
def canon(a,b,c):
    r=np.array([a,b,c],dtype=float);s=np.max(np.abs(r))
    if not math.isfinite(s) or s<=1e-12: raise ValueError
    r/=s
    for v in r[:2]:
        if abs(v)>1e-12:
            if v<0:r=-r
            break
    return r
def parse(problem):
    q=str(problem.question).strip();m=re.match(r"^Solve\s+(.+)\s+for\s+([A-Za-z]+)\.$",q)
    if not m: raise ValueError
    body,asked=m.groups();parts=[p.strip() for p in re.split(r"\s*(?:,|\band\b)\s*",body) if "=" in p]
    if len(parts)!=2: raise ValueError
    eq=[];syms=set()
    for t in parts:
        l,r=t.split("=",1);e=sp.expand(sp.sympify(l.replace("^","**"))-sp.sympify(r.replace("^","**")));eq.append(e);syms|=e.free_symbols
    syms=sorted(syms,key=str)
    if len(syms)!=2: raise ValueError
    rows=[]
    for e in eq:
        p=sp.Poly(e,*syms)
        if p.total_degree()>1: raise ValueError
        rows.append(canon(float(p.coeff_monomial(syms[0])),float(p.coeff_monomial(syms[1])),-float(p.coeff_monomial(1))))
    rows=sorted(rows,key=lambda x:tuple(np.round(x,14)));a,b,c=rows[0];d,e,f=rows[1];det=a*e-b*d
    if abs(det)<=1e-10: raise ValueError
    y=np.array([(c*e-b*f)/det,(a*f-c*d)/det],dtype=float);idx=next(i for i,s in enumerate(syms) if str(s)==asked);off=float(sp.N(sp.sympify(str(problem.answer)),18))
    if abs(off-y[idx])>1e-5: raise ValueError
    return np.array([a,b,c,d,e,f],dtype=float),y
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["train","interpolate"],default="train");ap.add_argument("--count",type=int,default=50000);ap.add_argument("--out",required=True);ap.add_argument("--seed",type=int,default=11);args=ap.parse_args()
    algebra=prepare();random.seed(args.seed);np.random.seed(args.seed);gen=algebra.train(lambda r:r)["linear_2d"] if args.split=="train" else algebra.test()["linear_2d"]
    out=Path(args.out);accepted=0
    with out.open("w",encoding="utf-8") as f:
        f.write("# DEEPMIND_MATHEMATICS_DATASET\n# commit="+COMMIT+"\n# module=algebra.linear_2d\n# split="+args.split+"\n# format=a,b,c,d,e,f|x,y\n")
        while accepted<args.count:
            try:x,y=parse(gen());f.write(",".join(f"{v:.9g}" for v in x)+"|"+",".join(f"{v:.9g}" for v in y)+"\n");accepted+=1
            except Exception:pass
    print(f"EXPORTED {accepted} {args.split} rows to {out} @ {COMMIT}")
if __name__=="__main__":main()
