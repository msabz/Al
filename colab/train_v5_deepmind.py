# Math AI Lab v5 - Google Colab trainer
# Copy this whole file into one Colab cell or run with: %run train_v5_deepmind.py
# It trains the EXACT Android v5 architecture and exports an Android-importable .mai5 checkpoint.

# ========================= USER SETTINGS =========================
TOTAL_STEPS = 30000          # increase for longer training
BATCH_SIZE = 128             # 128 is usually good on a Colab GPU
LEARNING_RATE = 6e-4
CONSISTENCY_WEIGHT = 0.05
DEEPMIND_RATIO = 0.60        # rest is v5 synthetic curriculum
CHECKPOINT_EVERY = 1000
SEED = 165
RESUME_FROM_MAI5 = False     # True -> Colab asks you to upload an existing .mai5
AUTO_DOWNLOAD_AT_END = True
OUTPUT_FILE = "/content/math_ai_v5_trained.mai5"

# ========================= INSTALL / IMPORT =========================
import os, sys, re, math, random, struct, subprocess, pathlib, shutil
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "absl-py", "sympy", "six"], check=True)
repo = pathlib.Path("/content/mathematics_dataset")
if not repo.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/google-deepmind/mathematics_dataset.git", str(repo)], check=True)
# NumPy 2 removed ndarray.itemset; patch the two old calls used by the official generator.
p = repo / "mathematics_dataset/sample/polynomials.py"
if p.exists():
    s = p.read_text()
    s = s.replace("coeffs.itemset(index, value)", "coeffs[index] = value")
    s = s.replace("expanded_coefficients.itemset(power, coeffs)", "expanded_coefficients[power] = coeffs")
    p.write_text(s)
sys.path.insert(0, str(repo))

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mathematics_dataset.modules import algebra as dm_algebra

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ========================= MAI5 CONTRACT (must match Android) =========================
MAGIC=0x4D414935; VERSION=1
MAX_NODES=80; TOKEN_VOCAB=22; EMB=16; EXTRA=3; NODE_FEATURES=19; INPUT=1520
SHARED1=160; SHARED2=128; HEAD_HIDDEN=64; HEADS=4; ROOT_SLOTS=5; STATES=4; HEAD_OUT=14
ROOT_SCALE=100.0; MAX_GRAD_NORM=5.0
PAD,NUMBER,X,Y,PI,E,ADD,SUB,MUL,DIV,POW,NEG,SIN,COS,TAN,SQRT,LOG,LN,EXP,ABS,EQ,SEP=range(22)
LINEAR,POLYNOMIAL,ANALYTIC,SYSTEM=range(4)
FINITE,NO_SOLUTION,INFINITE,UNSUPPORTED=range(4)
FUNCTIONS={"sin":SIN,"cos":COS,"tan":TAN,"sqrt":SQRT,"log":LOG,"ln":LN,"exp":EXP,"abs":ABS}

# ========================= EXACT PYTHON MIRROR OF ANDROID RPN ENCODER =========================
def normalize_text(s):
    trans=str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹سص", "01234567890123456789xy")
    return (str(s).translate(trans).lower().replace("**","^").replace("²","^2").replace("³","^3")
            .replace("π","pi").replace("√","sqrt").replace("−","-").replace("×","*").replace("÷","/").replace(" ",""))

def family_of(raw):
    s=normalize_text(raw)
    if ";" in s: return SYSTEM
    if any(f+"(" in s for f in FUNCTIONS): return ANALYTIC
    slash=s.find("/")
    if slash>=0 and ("x" in s[slash+1:] or "y" in s[slash+1:]): return ANALYTIC
    deg=[int(v) for v in re.findall(r"[xy]\^(\d+)",s)]
    if deg and max(deg)>=2: return POLYNOMIAL
    if re.search(r"\([^)]*[xy][^)]*\)\*\([^)]*[xy][^)]*\)",s): return POLYNOMIAL
    if re.search(r"[xy]\*+[xy]",s): return POLYNOMIAL
    if re.search(r"[xy]\*\([^)]*[xy][^)]*\)|\([^)]*[xy][^)]*\)\*[xy]",s): return POLYNOMIAL
    if len(re.findall(r"\([^)]*[xy][^)]*\)",s))>=2 and "*" in s: return POLYNOMIAL
    return LINEAR

def lex(expr):
    out=[]; i=0
    while i<len(expr):
        c=expr[i]
        if c.isdigit() or c=='.':
            j=i+1
            while j<len(expr) and (expr[j].isdigit() or expr[j]=='.'): j+=1
            out.append(("num",expr[i:j])); i=j; continue
        if c.isalpha():
            j=i+1
            while j<len(expr) and expr[j].isalpha(): j+=1
            out.append(("name",expr[i:j])); i=j; continue
        if c=='(': out.append(("lp",c))
        elif c==')': out.append(("rp",c))
        elif c in '+-*/^': out.append(("op",c))
        else: raise ValueError("unsupported char "+c)
        i+=1
    return out

def implicit(tokens):
    out=[]
    for i,t in enumerate(tokens):
        if i:
            p=tokens[i-1]
            left=p[0] in ("num","rp") or (p[0]=="name" and p[1] not in FUNCTIONS)
            right=t[0] in ("num","lp","name")
            func=p[0]=="name" and p[1] in FUNCTIONS and t[0]=="lp"
            if left and right and not func: out.append(("op","*"))
        out.append(t)
    return out

def prec(op): return {"+":1,"-":1,"*":2,"/":2,"neg":3,"^":4}.get(op,5)
def pop_op(op,out):
    m={"+":ADD,"-":SUB,"*":MUL,"/":DIV,"^":POW,"neg":NEG,**FUNCTIONS}
    out.append((m[op],0.0))

def rpn(expr):
    toks=implicit(lex(expr)); out=[]; ops=[]; prev=None
    for typ,text in toks:
        if typ=="num": out.append((NUMBER,float(text)))
        elif typ=="name":
            if text in FUNCTIONS: ops.append((typ,text))
            else: out.append(({"x":X,"y":Y,"pi":PI,"e":E}[text],0.0))
        elif typ=="lp": ops.append((typ,text))
        elif typ=="rp":
            while ops and ops[-1][0]!="lp": pop_op(ops.pop()[1],out)
            if not ops: raise ValueError("parentheses")
            ops.pop()
            if ops and ops[-1][0]=="name" and ops[-1][1] in FUNCTIONS: pop_op(ops.pop()[1],out)
        else:
            unary=text=='-' and (prev is None or prev[0] in ("op","lp")); cur="neg" if unary else text
            while ops:
                top_t,top=ops[-1]
                if top_t=="lp": break
                if top_t=="name": pop_op(ops.pop()[1],out); continue
                right=cur in ("^","neg")
                if prec(top)>prec(cur) or (prec(top)==prec(cur) and not right): pop_op(ops.pop()[1],out)
                else: break
            ops.append(("op",cur))
        prev=(typ,text)
    while ops:
        typ,text=ops.pop()
        if typ in ("lp","rp"): raise ValueError("parentheses")
        pop_op(text,out)
    return out

def encode(raw):
    src=normalize_text(raw); equations=[q for q in src.split(';') if q]
    if not 1<=len(equations)<=2: raise ValueError("equation count")
    nodes=[]
    for idx,q in enumerate(equations):
        # top-level equals
        d=0; at=-1
        for i,c in enumerate(q):
            if c=='(': d+=1
            elif c==')': d-=1
            elif c=='=' and d==0: at=i; break
        if at<0: raise ValueError("missing equals")
        nodes += rpn(q[:at]); nodes += rpn(q[at+1:]); nodes.append((EQ,0.0))
        if idx>0: nodes.append((SEP,0.0))
    if len(nodes)>MAX_NODES: raise ValueError("too long")
    kinds=np.zeros(MAX_NODES,np.int64); numeric=np.zeros(MAX_NODES,np.float32); depth=np.zeros(MAX_NODES,np.float32)
    stack=0
    for i,(k,v) in enumerate(nodes):
        kinds[i]=k
        if k==NUMBER and abs(v)>=1e-15: numeric[i]=np.clip(np.sign(v)*np.log1p(abs(v))/8.0,-2.5,2.5)
        if k in (NUMBER,X,Y,PI,E): stack+=1
        elif k in (NEG,SIN,COS,TAN,SQRT,LOG,LN,EXP,ABS): stack=max(1,stack)
        elif k in (ADD,SUB,MUL,DIV,POW,EQ,SEP): stack=max(1,stack-1)
        depth[i]=min(max(stack,0),12)/12.0
    return kinds,numeric,depth,family_of(src),src

# ========================= MODEL =========================
class MAI5(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(TOKEN_VOCAB,EMB,padding_idx=PAD)
        nn.init.uniform_(self.embedding.weight,-0.05,0.05)
        with torch.no_grad(): self.embedding.weight[PAD].zero_()
        self.shared1=nn.Linear(INPUT,SHARED1); self.shared2=nn.Linear(SHARED1,SHARED2)
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(SHARED2,HEAD_HIDDEN),nn.ReLU(),nn.Linear(HEAD_HIDDEN,HEAD_OUT)) for _ in range(HEADS)])
        nn.init.kaiming_uniform_(self.shared1.weight, a=0, nonlinearity='relu'); nn.init.zeros_(self.shared1.bias)
        nn.init.kaiming_uniform_(self.shared2.weight, a=0, nonlinearity='relu'); nn.init.zeros_(self.shared2.bias)
        for h in self.heads:
            nn.init.kaiming_uniform_(h[0].weight,a=0,nonlinearity='relu'); nn.init.zeros_(h[0].bias)
            nn.init.kaiming_uniform_(h[2].weight,a=math.sqrt(5)); nn.init.zeros_(h[2].bias)
    def forward(self,kinds,numeric,depth,family):
        b=kinds.shape[0]; emb=self.embedding(kinds)
        pos=torch.linspace(0,1,MAX_NODES,device=kinds.device).view(1,MAX_NODES,1).expand(b,-1,-1)
        feats=torch.cat([emb,numeric.unsqueeze(-1),pos,depth.unsqueeze(-1)],-1).reshape(b,-1)
        z=F.relu(self.shared1(feats)); z=F.relu(self.shared2(z))
        all_heads=torch.stack([h(z) for h in self.heads],1)
        return all_heads[torch.arange(b,device=z.device),family]

model=MAI5().to(device)
print("Parameters:",sum(p.numel() for p in model.parameters()))
assert sum(p.numel() for p in model.parameters())==300984

# Android-compatible Adam: one global step; even routed heads receive zero-gradient moment decay.
params=list(model.parameters()); moments=[torch.zeros_like(p) for p in params]; velocities=[torch.zeros_like(p) for p in params]; adam_step=0
def android_adam_step(lr):
    global adam_step
    grads=[p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    norm=torch.sqrt(sum((g.float()**2).sum() for g in grads)); scale=min(1.0,MAX_GRAD_NORM/(float(norm)+1e-30))
    adam_step+=1; b1=.9; b2=.999; c1=1-b1**adam_step; c2=1-b2**adam_step
    with torch.no_grad():
        for p,g,m,v in zip(params,grads,moments,velocities):
            g=g*scale; m.mul_(b1).add_(g,alpha=1-b1); v.mul_(b2).addcmul_(g,g,value=1-b2)
            p.addcdiv_(m/c1,(v/c2).sqrt().add_(1e-8),value=-lr)
        model.embedding.weight[PAD].zero_(); moments[0][PAD].zero_(); velocities[0][PAD].zero_()
    model.zero_grad(set_to_none=True)
    return float(norm)

PERMS=torch.tensor(list(__import__('itertools').permutations(range(ROOT_SLOTS))),device=device,dtype=torch.long)

def loss_fn(out,roots,root_count,systems,states,families,other_out=None):
    # state loss
    state_loss=F.cross_entropy(out[:,10:14],states)
    assigned_vals=torch.zeros((len(out),ROOT_SLOTS),device=device); assigned_pres=torch.zeros_like(assigned_vals)
    finite=states==FINITE; sysmask=finite & (families==SYSTEM); nonsys=finite & (families!=SYSTEM)
    if sysmask.any():
        assigned_vals[sysmask,:2]=systems[sysmask,:2]/ROOT_SCALE; assigned_pres[sysmask,:2]=1
    ids=torch.where(nonsys)[0]
    if len(ids):
        tv=roots[ids]/ROOT_SCALE; tc=root_count[ids]
        basepres=(torch.arange(ROOT_SLOTS,device=device)[None,:] < tc[:,None]).float()
        # target permutation: [N,120,5]
        pv=tv[:,PERMS]; pp=basepres[:,PERMS]
        predv=out[ids,:5][:,None,:]; predlog=out[ids,5:10][:,None,:].expand(-1,len(PERMS),-1)
        active=pp.sum(-1).clamp_min(1)
        cost=(((predv-pv)**2)*pp).sum(-1)/active + .35*F.binary_cross_entropy_with_logits(predlog,pp,reduction='none').mean(-1)
        best=cost.argmin(-1); rows=torch.arange(len(ids),device=device)
        assigned_vals[ids]=pv[rows,best]; assigned_pres[ids]=pp[rows,best]
    active=assigned_pres.sum(-1).clamp_min(1)
    root_loss=((((out[:,:5]-assigned_vals)**2)*assigned_pres).sum(-1)/active)[finite].mean() if finite.any() else out.sum()*0
    presence=F.binary_cross_entropy_with_logits(out[:,5:10],assigned_pres)
    total=root_loss + .35*presence + .35*state_loss
    if other_out is not None: total=total + CONSISTENCY_WEIGHT*F.mse_loss(out,other_out)
    return total

# ========================= TRAINING EXAMPLE GENERATORS =========================
def mk(eq,roots=(),system=(),state=FINITE,equiv=None):
    k,n,d,f,src=encode(eq); ek=en=ed=ef=None
    if equiv:
        ek,en,ed,ef,_=encode(equiv)
        if ef!=f: equiv=None
    return dict(eq=src,k=k,n=n,d=d,f=f,roots=list(roots)[:5],system=list(system)[:2],state=state,equiv=equiv,ek=ek,en=en,ed=ed,ef=ef)

def swap(eq):
    if ';' in eq: return ';'.join(swap(x) for x in eq.split(';'))
    a,b=eq.split('=',1); return b+'='+a

def synthetic(rng,max_abs=100):
    t=rng.randrange(100)
    if t<20:
        root=rng.randint(-max_abs,max_abs)/rng.randint(1,4); a=rng.choice([i for i in range(-12,13) if i]); b=rng.randint(-30,30); eq=f"{a}*x{b:+}={a*root+b:g}"
        return mk(eq,[root],equiv=swap(eq))
    if t<48:
        degree=rng.randint(2,5); rr=[float(rng.randint(-12,12)) for _ in range(degree)]; eq='*'.join(f"(x{-v:+g})" for v in rr)+'=0'
        return mk(eq,sorted(set(rr)),equiv=swap(eq))
    if t<70:
        root=rng.randint(-40,40)/4
        mode=rng.randrange(4)
        if mode==0:
            a=rng.randint(1,6); z=rng.randint(1,12); b=z*z-a*root; eq=f"sqrt({a}*x{b:+g})={z}"
        elif mode==1:
            base=rng.randint(2,6); eq=f"{base}^x={base**root:.8g}"
        elif mode==2:
            a=rng.randint(1,5); inner=rng.randint(1,14); b=inner-a*root; eq=f"ln({a}*x{b:+g})={math.log(inner):.8g}"
        else:
            root=max(-1.2,min(1.2,root)); eq=f"tan(x)={math.tan(root):.8g}"
        return mk(eq,[root],equiv=swap(eq))
    if t<90:
        while True:
            a1,b1,a2,b2=[rng.randint(-9,9) for _ in range(4)]
            if a1*b2-a2*b1: break
        x=float(rng.randint(-25,25)); y=float(rng.randint(-25,25)); c1=a1*x+b1*y; c2=a2*x+b2*y
        eq=f"{a1}*x{b1:+}*y={c1:g};{a2}*x{b2:+}*y={c2:g}"
        return mk(eq,system=[x,y],equiv=';'.join(reversed(eq.split(';'))))
    z=rng.randrange(4)
    if z==0:return mk("0*x=1",state=NO_SOLUTION,equiv="1=0*x")
    if z==1:return mk("0*x=0",state=INFINITE,equiv="0=0*x")
    if z==2:return mk("x+y=2;2*x+2*y=4",state=INFINITE,equiv="2*x+2*y=4;x+y=2")
    return mk("x*y=1",state=UNSUPPORTED,equiv="1=x*y")

# Official DeepMind Mathematics Dataset generators. We consume only equation-solving modules that fit v5's output contract.
dm_modules=dm_algebra.train(lambda entropy_range: entropy_range)
DM_NAMES=["linear_1d","linear_2d","polynomial_roots"]

def parse_eq(s):
    a,b=s.split('=',1); return sp.Eq(sp.sympify(a.replace('^','**')),sp.sympify(b.replace('^','**')))
def canonical_equations(eq_strings):
    eqs=[parse_eq(x.strip()) for x in eq_strings]
    syms=sorted(set().union(*(e.free_symbols for e in eqs)),key=lambda z:str(z))
    if not 1<=len(syms)<=2: raise ValueError("symbol count")
    repl={syms[0]:sp.Symbol('x')}
    if len(syms)==2: repl[syms[1]]=sp.Symbol('y')
    out=[]
    for e in eqs:
        out.append(str(e.lhs.subs(repl)).replace('**','^')+'='+str(e.rhs.subs(repl)).replace('**','^'))
    sol=sp.solve(eqs,syms,dict=True)
    return out,syms,sol,repl

def extract_polynomial_equality(q):
    patterns=[
      r"^Let (.+?=.+?)\. (?:What is|Calculate) [A-Za-z]\??$", r"^Suppose (.+?=.+?)\. (?:What is|Calculate) [A-Za-z]\??$",
      r"^What is [A-Za-z] in (.+?=.+?)\?$", r"^Solve (.+?=.+?)(?: for [A-Za-z])?\.$",
      r"^Find [A-Za-z],? (?:such that|given that) (.+?=.+?)\.$", r"^Determine [A-Za-z],? (?:so that|given that) (.+?=.+?)\.$"
    ]
    for pat in patterns:
        m=re.match(pat,q)
        if m:return m.group(1)
    raise ValueError("unparsed polynomial prompt")

def deepmind_example(rng):
    for _ in range(80):
        name=rng.choice(DM_NAMES); problem=dm_modules[name](); q=str(problem.question)
        try:
            if name.startswith('linear_'):
                m=re.match(r"^Solve (.+) for ([A-Za-z])\.$",q)
                if not m: continue
                text=m.group(1).replace(' and ',', '); parts=[x.strip() for x in text.split(',') if '=' in x]
                eqs,syms,sol,repl=canonical_equations(parts)
                if not sol: continue
                if len(syms)==1:
                    val=float(sp.N(sol[0][syms[0]])); eq=eqs[0]; return mk(eq,[val],equiv=swap(eq))
                if len(syms)==2 and all(s in sol[0] for s in syms):
                    vals=[float(sp.N(sol[0][s])) for s in syms]; eq=';'.join(eqs); return mk(eq,system=vals,equiv=';'.join(reversed(eqs)))
            else:
                if q.startswith('Factor '): continue
                raw=extract_polynomial_equality(q); eq0=parse_eq(raw); syms=sorted(eq0.free_symbols,key=lambda z:str(z))
                if len(syms)!=1: continue
                roots=sp.solve(eq0,syms[0]); real=[]
                for r in roots:
                    c=complex(sp.N(r));
                    if abs(c.imag)<1e-8: real.append(float(c.real))
                real=sorted(set(round(x,10) for x in real))
                if not 1<=len(real)<=5: continue
                eq=str(eq0.lhs.subs({syms[0]:sp.Symbol('x')})).replace('**','^')+'='+str(eq0.rhs.subs({syms[0]:sp.Symbol('x')})).replace('**','^')
                return mk(eq,real,equiv=swap(eq))
        except Exception:
            continue
    return synthetic(rng)

# ========================= BATCHING =========================
def collate(examples):
    kinds=torch.tensor(np.stack([e['k'] for e in examples]),device=device,dtype=torch.long)
    numeric=torch.tensor(np.stack([e['n'] for e in examples]),device=device,dtype=torch.float32)
    depth=torch.tensor(np.stack([e['d'] for e in examples]),device=device,dtype=torch.float32)
    family=torch.tensor([e['f'] for e in examples],device=device,dtype=torch.long)
    roots=np.zeros((len(examples),5),np.float32); rc=np.zeros(len(examples),np.int64); systems=np.zeros((len(examples),2),np.float32)
    for i,e in enumerate(examples):
        rc[i]=len(e['roots']); roots[i,:rc[i]]=e['roots']; systems[i,:len(e['system'])]=e['system']
    states=torch.tensor([e['state'] for e in examples],device=device,dtype=torch.long)
    roots=torch.tensor(roots,device=device); rc=torch.tensor(rc,device=device); systems=torch.tensor(systems,device=device)
    if all(e['equiv'] is not None for e in examples):
        ek=torch.tensor(np.stack([e['ek'] for e in examples]),device=device,dtype=torch.long)
        en=torch.tensor(np.stack([e['en'] for e in examples]),device=device,dtype=torch.float32)
        ed=torch.tensor(np.stack([e['ed'] for e in examples]),device=device,dtype=torch.float32)
        ef=torch.tensor([e['ef'] for e in examples],device=device,dtype=torch.long)
        equiv=(ek,en,ed,ef)
    else: equiv=None
    return kinds,numeric,depth,family,roots,rc,systems,states,equiv

# ========================= MAI5 LOAD / SAVE =========================
def _read_i(f): return struct.unpack('>i',f.read(4))[0]
def _read_f(f): return struct.unpack('>f',f.read(4))[0]
def _read_arr(f,shape): return np.frombuffer(f.read(int(np.prod(shape))*4),dtype='>f4').astype(np.float32).reshape(shape)
def _write_i(f,v): f.write(struct.pack('>i',int(v)))
def _write_f(f,v): f.write(struct.pack('>f',float(v)))
def _write_arr(f,a): f.write(np.asarray(a,dtype='>f4').tobytes(order='C'))

def param_index(): return {id(p):i for i,p in enumerate(params)}
def save_triplet(f,p,transpose=False):
    i=param_index()[id(p)]; arrays=[p.detach().cpu().numpy(),moments[i].detach().cpu().numpy(),velocities[i].detach().cpu().numpy()]
    for a in arrays: _write_arr(f,a.T if transpose else a)

def save_mai5(path):
    with open(path,'wb') as f:
        for v in [MAGIC,VERSION,MAX_NODES,TOKEN_VOCAB,EMB,INPUT,SHARED1,SHARED2,HEAD_HIDDEN,HEADS,HEAD_OUT,adam_step]: _write_i(f,v)
        _write_f(f,ROOT_SCALE)
        save_triplet(f,model.embedding.weight)
        save_triplet(f,model.shared1.weight,True); save_triplet(f,model.shared1.bias)
        save_triplet(f,model.shared2.weight,True); save_triplet(f,model.shared2.bias)
        for h in model.heads:
            save_triplet(f,h[0].weight,True); save_triplet(f,h[0].bias); save_triplet(f,h[2].weight,True); save_triplet(f,h[2].bias)

def load_triplet(f,p,shape_on_disk,transpose=False):
    idx=param_index()[id(p)]; arrs=[_read_arr(f,shape_on_disk) for _ in range(3)]
    if transpose: arrs=[a.T.copy() for a in arrs]
    with torch.no_grad(): p.copy_(torch.tensor(arrs[0],device=device)); moments[idx].copy_(torch.tensor(arrs[1],device=device)); velocities[idx].copy_(torch.tensor(arrs[2],device=device))

def load_mai5(path):
    global adam_step
    with open(path,'rb') as f:
        expected=[MAGIC,VERSION,MAX_NODES,TOKEN_VOCAB,EMB,INPUT,SHARED1,SHARED2,HEAD_HIDDEN,HEADS,HEAD_OUT]
        got=[_read_i(f) for _ in expected]
        if got!=expected: raise ValueError('Incompatible MAI5 header: '+str(got))
        adam_step=_read_i(f)
        if abs(_read_f(f)-ROOT_SCALE)>1e-4: raise ValueError('ROOT_SCALE mismatch')
        load_triplet(f,model.embedding.weight,(TOKEN_VOCAB,EMB))
        load_triplet(f,model.shared1.weight,(INPUT,SHARED1),True); load_triplet(f,model.shared1.bias,(SHARED1,))
        load_triplet(f,model.shared2.weight,(SHARED1,SHARED2),True); load_triplet(f,model.shared2.bias,(SHARED2,))
        for h in model.heads:
            load_triplet(f,h[0].weight,(SHARED2,HEAD_HIDDEN),True); load_triplet(f,h[0].bias,(HEAD_HIDDEN,))
            load_triplet(f,h[2].weight,(HEAD_HIDDEN,HEAD_OUT),True); load_triplet(f,h[2].bias,(HEAD_OUT,))
    with torch.no_grad(): model.embedding.weight[PAD].zero_(); moments[0][PAD].zero_(); velocities[0][PAD].zero_()
    print('Resumed MAI5 at Adam step',adam_step)

if RESUME_FROM_MAI5:
    from google.colab import files
    uploaded=files.upload(); path=next(iter(uploaded)); load_mai5(path)

# Fixed external holdout: larger coefficient/solution range, never fed into training.
hold_rng=random.Random(0xA165)
holdout=[synthetic(hold_rng,max_abs=240) for _ in range(160)]

def evaluate():
    model.eval(); sq=ae=0.; cnt=within=state_ok=0
    with torch.no_grad():
        for start in range(0,len(holdout),64):
            batch=holdout[start:start+64]; k,n,d,f,r,rc,sy,st,eqv=collate(batch); out=model(k,n,d,f); state_ok+=(out[:,10:14].argmax(-1)==st).sum().item()
            for i,e in enumerate(batch):
                if e['state']!=FINITE: continue
                if e['f']==SYSTEM:
                    pv=out[i,:2].cpu().numpy()*ROOT_SCALE; ev=np.asarray(e['system']); errs=np.abs(pv-ev)
                else:
                    probs=torch.sigmoid(out[i,5:10]); pv=(out[i,:5][probs>=.5]*ROOT_SCALE).cpu().numpy(); ev=np.asarray(e['roots'])
                    if len(ev)==0: continue
                    used=set(); errs=[]
                    for v in ev:
                        cand=[(abs(float(p)-v),j) for j,p in enumerate(pv) if j not in used]
                        if cand: er,j=min(cand); used.add(j); errs.append(er)
                        else: errs.append(ROOT_SCALE)
                    errs=np.asarray(errs)
                sq+=float((errs**2).sum()); ae+=float(errs.sum()); cnt+=len(errs); within+=int((errs<=1).sum())
    model.train(); return math.sqrt(sq/max(cnt,1)),ae/max(cnt,1),within/max(cnt,1),state_ok/len(holdout)

# ========================= TRAIN =========================
rng=random.Random(SEED+1); model.train()
for step_idx in range(adam_step+1, TOTAL_STEPS+1):
    ex=[]
    for _ in range(BATCH_SIZE):
        ex.append(deepmind_example(rng) if rng.random()<DEEPMIND_RATIO else synthetic(rng))
    k,n,d,f,r,rc,sy,st,eqv=collate(ex)
    out=model(k,n,d,f)
    other=model(*eqv) if eqv is not None else None
    loss=loss_fn(out,r,rc,sy,st,f,other)
    loss.backward(); gnorm=android_adam_step(LEARNING_RATE)
    if step_idx%100==0:
        print(f"step {step_idx:6d}/{TOTAL_STEPS}  loss={float(loss):.6f}  grad={gnorm:.3f}")
    if step_idx%CHECKPOINT_EVERY==0:
        save_mai5(OUTPUT_FILE)
        rmse,mae,acc,sacc=evaluate(); print(f"  HOLDOUT rmse={rmse:.3f} mae={mae:.3f} ±1={acc*100:.1f}% state={sacc*100:.1f}%  saved={OUTPUT_FILE}")

save_mai5(OUTPUT_FILE)
rmse,mae,acc,sacc=evaluate()
print("\nDONE")
print(f"Holdout RMSE={rmse:.4f}  MAE={mae:.4f}  within ±1={acc*100:.2f}%  state accuracy={sacc*100:.2f}%")
print("MAI5:",OUTPUT_FILE,"bytes:",os.path.getsize(OUTPUT_FILE),"Adam step:",adam_step)
if AUTO_DOWNLOAD_AT_END:
    from google.colab import files
    files.download(OUTPUT_FILE)
