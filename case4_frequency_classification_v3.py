"""
Case 4 - fixed-energy local phase-space regularity map (v3)

This version is designed to avoid the main problems of the first prototype:

* It does NOT call every failed spectral test "chaotic".
* It requires sufficiently long time windows for frequency estimation.
* It uses several observables, but ignores nearly constant signals.
* It calibrates numerical frequency-drift and spectral-entropy floors using the
  known Case-4 periodic orbit.
* It reports REGULAR CANDIDATE / CHAOTIC CANDIDATE / UNCERTAIN.
* It keeps the full raw diagnostics so the threshold does not hide the result.
* It includes an optional tangent-map FTLE validator for selected points.

Scientific interpretation
--------------------------
A small, converged frequency drift is evidence of regular/quasi-periodic-like
motion, not a proof of quasi-periodicity. A clearly positive FTLE is evidence
for chaos. Points where the diagnostics disagree are explicitly marked
UNCERTAIN.

The scan varies (x1,y1) while enforcing H=E by reconstructing p_y1. Thus the
scan is a 2-D surface inside the 7-D fixed-energy manifold.

The known Case-4 periodic point is
    x* = 3.8351644331875563
    q* = [[x*,5],[10-x*,5]]
    p* = [[0,20],[0,-20]]

The model matches the supplied notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.signal import find_peaks

# =============================================================================
# Model
# =============================================================================
PARAMETERS = dict(k=1.0, alpha=1.0, beta=1.0, C=1.0, Lx=10.0, Ly=10.0)
MASSES = np.ones(2)
X_STAR = 3.8351644331875563
Y_STAR = 5.0
Q_STAR = np.array([[X_STAR, Y_STAR], [10.0 - X_STAR, Y_STAR]], dtype=float)
P_STAR = np.array([[0.0, 20.0], [0.0, -20.0]], dtype=float)
REFERENCE_PERIOD = 1.0027085198  # from report02 Case 4


def kinetic(p):
    return float(np.sum(p * p / (2.0 * MASSES[:, None])))


def potential(q):
    d = q[0] - q[1]
    r2 = float(np.dot(d, d))
    if r2 <= 0.0:
        return np.inf
    pair = PARAMETERS["k"] * r2 ** (-PARAMETERS["alpha"] / 2.0)
    wall = 0.0
    for x, y in q:
        if not (0.0 < x < PARAMETERS["Lx"] and 0.0 < y < PARAMETERS["Ly"]):
            return np.inf
        b = PARAMETERS["beta"]
        C = PARAMETERS["C"]
        Lx = PARAMETERS["Lx"]
        Ly = PARAMETERS["Ly"]
        wall += C * (x**(-b) + (Lx-x)**(-b) + y**(-b) + (Ly-y)**(-b))
    return float(pair + wall)


def total_energy(q, p):
    return kinetic(p) + potential(q)


def force(q):
    k = PARAMETERS["k"]
    a = PARAMETERS["alpha"]
    b = PARAMETERS["beta"]
    C = PARAMETERS["C"]
    Lx = PARAMETERS["Lx"]
    Ly = PARAMETERS["Ly"]

    d = q[0] - q[1]
    r2 = float(np.dot(d, d))
    if r2 <= 0.0:
        raise FloatingPointError("Particle collision.")
    pair = a * k * d * r2 ** (-a / 2.0 - 1.0)
    F = np.zeros_like(q)
    F[0] += pair
    F[1] -= pair
    for i in range(2):
        x, y = q[i]
        F[i, 0] += C * b * (x**(-b-1.0) - (Lx-x)**(-b-1.0))
        F[i, 1] += C * b * (y**(-b-1.0) - (Ly-y)**(-b-1.0))
    return F


def force_jacobian(q):
    """dF/dq, flattened ordering [x1,y1,x2,y2]."""
    k = PARAMETERS["k"]
    a = PARAMETERS["alpha"]
    b = PARAMETERS["beta"]
    C = PARAMETERS["C"]
    Lx = PARAMETERS["Lx"]
    Ly = PARAMETERS["Ly"]

    d = q[0] - q[1]
    r2 = float(np.dot(d, d))
    if r2 <= 0.0:
        raise FloatingPointError("Particle collision.")

    I = np.eye(2)
    A = a*k*(r2**(-a/2.0-1.0)*I - (a+2.0)*r2**(-a/2.0-2.0)*np.outer(d,d))
    J = np.zeros((4,4))
    J[:2,:2] += A
    J[:2,2:] -= A
    J[2:,:2] -= A
    J[2:,2:] += A

    for i in range(2):
        x,y = q[i]
        wx = -C*b*(b+1.0)*(x**(-b-2.0) + (Lx-x)**(-b-2.0))
        wy = -C*b*(b+1.0)*(y**(-b-2.0) + (Ly-y)**(-b-2.0))
        j=2*i
        J[j,j] += wx
        J[j+1,j+1] += wy
    return J


def same_energy_ic(x1, y1, E):
    q = Q_STAR.copy()
    p = P_STAR.copy()
    q[0] = [x1, y1]
    other_ke = kinetic(p) - 0.5*p[0,1]**2
    available = E - potential(q) - other_ke
    if available <= 0.0:
        raise ValueError("No real positive p_y1 on this fixed-energy slice.")
    p[0,1] = np.sqrt(2.0*available)
    return q,p


# =============================================================================
# Integration
# =============================================================================
def integrate_orbit(q0,p0,h,total_time,sample_stride=10):
    nsteps = int(round(total_time/h))
    q = q0.copy(); p = p0.copy(); E0 = total_energy(q,p)
    ts=[]; qs=[]; ps=[]; Es=[]
    for s in range(nsteps+1):
        if s % sample_stride == 0 or s == nsteps:
            ts.append(s*h); qs.append(q.copy()); ps.append(p.copy()); Es.append(total_energy(q,p))
        if s == nsteps: break
        ph = p + 0.5*h*force(q)
        q = q + h*ph
        p = ph + 0.5*h*force(q)
    return np.asarray(ts),np.asarray(qs),np.asarray(ps),np.asarray(Es),E0


def integrate_with_tangent(q0,p0,h,total_time,sample_stride=10,renorm_steps=20,seed=123):
    """Optional FTLE calculation using the exact tangent map of leapfrog."""
    rng=np.random.default_rng(seed)
    v=rng.normal(size=8)
    gradH=np.concatenate([-force(q0).ravel(),p0.ravel()])
    v -= np.dot(v,gradH)/np.dot(gradH,gradH)*gradH
    v /= np.linalg.norm(v)
    q=q0.copy(); p=p0.copy(); E0=total_energy(q,p)
    nsteps=int(round(total_time/h)); logsum=0.0; nrm_count=0
    for s in range(nsteps):
        F=force(q); J0=force_jacobian(q)
        ph=p+0.5*h*F; qn=q+h*ph; J1=force_jacobian(qn); pn=ph+0.5*h*force(qn)
        dq=v[:4]; dp=v[4:]
        dp_half=dp+0.5*h*(J0@dq)
        dq_new=dq+h*dp_half
        dp_new=dp_half+0.5*h*(J1@dq_new)
        v=np.concatenate([dq_new,dp_new])
        if (s+1)%renorm_steps==0:
            gradH=np.concatenate([-force(qn).ravel(),pn.ravel()])
            denom=np.dot(gradH,gradH)
            if denom>0:
                v -= np.dot(v,gradH)/denom*gradH
            n=np.linalg.norm(v)
            logsum += np.log(n); nrm_count += 1; v/=n
        q,p=qn,pn
    ftle=logsum/(nrm_count*renorm_steps*h) if nrm_count else np.nan
    return float(ftle), float(np.max(np.abs(total_energy(q,p)-E0))/abs(E0))


# =============================================================================
# Frequency extraction
# =============================================================================
def hann(n): return np.hanning(n)


def projection(y,t,w,window):
    y=y-np.mean(y)
    return np.sum(window*y*np.exp(-1j*w*t))/np.sum(window)


def refine_frequency(y,t,seed,resolution):
    dt=float(np.mean(np.diff(t))); nyq=np.pi/dt
    lo=max(0.0,seed-2*resolution); hi=min(nyq,seed+2*resolution)
    win=hann(len(y))
    out=minimize_scalar(lambda w:-abs(projection(y,t,w,win))**2,bounds=(lo,hi),method="bounded",options={"xatol":1e-10})
    return float(out.x)


def fit_line(y,t,w):
    A=np.column_stack([np.ones_like(t),np.cos(w*t),np.sin(w*t)])
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    model=A@c
    amp=float(np.hypot(c[1],c[2]))
    return model,amp


def spectral_lines(y,t,max_lines=6,min_amp_fraction=0.02):
    y=np.asarray(y,float); t=np.asarray(t,float)
    y=y-np.mean(y)
    if np.std(y) < 1e-10:
        return np.array([]),np.array([]),0.0
    dt=float(np.mean(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-7,atol=1e-12):
        raise ValueError("Nonuniform time grid.")
    win=hann(len(y))
    raw=np.abs(np.fft.rfft(win*y)); raw[0]=0
    power=raw**2
    prob=power/np.sum(power)
    entropy=float(-np.sum(prob[prob>0]*np.log(prob[prob>0]))/np.log(len(prob)))
    ff=2*np.pi*np.fft.rfftfreq(len(y),d=dt)
    res=2*np.pi/max(t[-1]-t[0],dt)
    peaks,_=find_peaks(raw)
    order=peaks[np.argsort(raw[peaks])[::-1]] if len(peaks) else []
    residual=y.copy(); found=[]
    for idx in order:
        if len(found)>=max_lines: break
        seed=float(ff[idx])
        if seed<=0: continue
        if any(abs(seed-w)<1.5*res for w,_ in found): continue
        w=refine_frequency(residual,t,seed,res)
        model,amp=fit_line(residual,t,w)
        found.append((w,amp)); residual-=model
    if not found:
        return np.array([]),np.array([]),entropy
    found=np.asarray(found)
    ampmax=np.max(found[:,1])
    found=found[found[:,1]>=min_amp_fraction*ampmax]
    return found[:,0],found[:,1],entropy


def match_lines(a,b):
    if len(a)==0 or len(b)==0: return np.inf,0.0
    C=np.abs(a[:,None]-b[None,:])/np.maximum(np.abs(a[:,None]),1e-12)
    r,c=linear_sum_assignment(C)
    if len(r)==0: return np.inf,0.0
    return float(np.max(C[r,c])),float(len(r)/max(len(a),len(b)))


OBSERVABLES={
    "x1":lambda q:q[:,0,0],
    "y1":lambda q:q[:,0,1],
    "x2":lambda q:q[:,1,0],
    "y2":lambda q:q[:,1,1],
    "px1":lambda q:q[:,0,0],  # position family repeated below only for labels; p handled separately
}


def all_observables(q,p):
    return {
        "x1":q[:,0,0],"y1":q[:,0,1],"x2":q[:,1,0],"y2":q[:,1,1],
        "px1":p[:,0,0],"py1":p[:,0,1],"px2":p[:,1,0],"py2":p[:,1,1],
        "dx":q[:,0,0]-q[:,1,0],"dy":q[:,0,1]-q[:,1,1],
    }


def frequency_diagnostic(t,q,p,n_windows=2,max_lines=6):
    edges=np.linspace(0,len(t),n_windows+1,dtype=int)
    details={}; drifts=[]; persists=[]; entropies=[]
    obs=all_observables(q,p)
    for name,signal in obs.items():
        if np.std(signal) < 1e-8:  # almost constant signal gives meaningless FFT lines
            continue
        fw=[]; aw=[]; ew=[]
        for k in range(n_windows):
            a,b=edges[k],edges[k+1]
            tt=t[a:b]-t[a]
            f,amp,e=spectral_lines(signal[a:b],tt,max_lines=max_lines)
            fw.append(f); aw.append(amp); ew.append(e)
        ld=[];lp=[]
        for k in range(n_windows-1):
            d,pp=match_lines(fw[k],fw[k+1])
            if np.isfinite(d): ld.append(d); lp.append(pp)
        if ld:
            d=float(np.median(ld)); pp=float(np.median(lp))
            drifts.append(d); persists.append(pp); entropies.extend(ew)
        details[name]={"frequencies":fw,"amplitudes":aw,"entropy":ew}
    drift=float(np.median(drifts)) if drifts else np.inf
    persistence=float(np.median(persists)) if persists else 0.0
    entropy=float(np.median(entropies)) if entropies else np.inf
    return drift,persistence,entropy,details


# =============================================================================
# Calibration + classification
# =============================================================================
def calibrate(h,total_time,sample_stride,max_lines,include_ftle=False):
    if total_time/2 < 20*REFERENCE_PERIOD:
        print("WARNING: each frequency window contains fewer than 10 Case-4 periods. Increase total_time.")
    t,q,p,E,E0=integrate_orbit(Q_STAR,P_STAR,h,total_time,sample_stride)
    drift,persist,entropy,_=frequency_diagnostic(t,q,p,n_windows=2,max_lines=max_lines)
    ftle=np.nan
    if include_ftle:
        ftle,_=integrate_with_tangent(Q_STAR,P_STAR,h,total_time,sample_stride)
    return {
        "freq_drift":drift,"persistence":persist,"entropy":entropy,"ftle":ftle,
        "energy_error":np.max(np.abs(E-E0))/abs(E0)
    }


def thresholds_from_reference(ref,freq_factor=20.0,ftle_factor=5.0):
    f0=max(ref["freq_drift"],1e-6)
    l0=max(abs(ref["ftle"]),1e-6) if np.isfinite(ref["ftle"]) else np.nan
    out={
        "freq_regular":freq_factor*f0,
        "freq_chaotic":max(4.0*freq_factor*f0,5e-3),
        "persistence_regular":0.75,
    }
    if np.isfinite(l0):
        out["ftle_regular"]=ftle_factor*l0
        out["ftle_chaotic"]=max(3.0*ftle_factor*l0,2e-2)
    return out


def classify(drift,persistence,ftle,thr):
    reg_freq=drift<=thr["freq_regular"] and persistence>=thr["persistence_regular"]
    chaos_freq=drift>=thr["freq_chaotic"]
    if np.isfinite(ftle) and "ftle_regular" in thr:
        reg_ftle=ftle<=thr["ftle_regular"]
        chaos_ftle=ftle>=thr["ftle_chaotic"]
        if reg_freq and reg_ftle: return 1
        if chaos_freq and chaos_ftle: return 0
        return 2
    # Without FTLE we intentionally refuse to label points as chaotic.
    return 1 if reg_freq else 2


# =============================================================================
# Grid project
# =============================================================================
def run_grid(
    nx=7,ny=7,x_half_width=0.05,y_half_width=0.05,
    h=5e-5,total_time=20.0,sample_stride=20,max_lines=6,
    do_ftle=True,renorm_steps=20,
):
    E0=total_energy(Q_STAR,P_STAR)
    xs=np.linspace(X_STAR-x_half_width,X_STAR+x_half_width,nx)
    ys=np.linspace(Y_STAR-y_half_width,Y_STAR+y_half_width,ny)

    ref=calibrate(h,total_time,sample_stride,max_lines,include_ftle=do_ftle)
    thr=thresholds_from_reference(ref)
    print("Reference periodic orbit:",ref)
    print("Thresholds:",thr)

    cls=np.full((ny,nx),np.nan); drift=np.full((ny,nx),np.nan)
    pers=np.full((ny,nx),np.nan); ent=np.full((ny,nx),np.nan)
    ft=np.full((ny,nx),np.nan); ee=np.full((ny,nx),np.nan)

    for iy,y1 in enumerate(ys):
        for ix,x1 in enumerate(xs):
            try:
                q0,p0=same_energy_ic(x1,y1,E0)
                t,q,p,E,Einit=integrate_orbit(q0,p0,h,total_time,sample_stride)
                d,pp,e,_=frequency_diagnostic(t,q,p,n_windows=2,max_lines=max_lines)
                lam=np.nan
                if do_ftle:
                    lam,_=integrate_with_tangent(q0,p0,h,total_time,sample_stride,renorm_steps)
                c=classify(d,pp,lam,thr) if do_ftle else (1 if d<=thr["freq_regular"] and pp>=thr["persistence_regular"] else 2)
                cls[iy,ix]=c; drift[iy,ix]=d; pers[iy,ix]=pp; ent[iy,ix]=e; ft[iy,ix]=lam
                ee[iy,ix]=np.max(np.abs(E-Einit))/abs(Einit)
                names={0:"CHAOS",1:"REGULAR",2:"UNCERTAIN"}
                print(f"[{iy+1:02d},{ix+1:02d}] x={x1:.6f} y={y1:.6f} drift={d:.3e} FTLE={lam:.3e} -> {names[c]}")
            except Exception as exc:
                print(f"SKIP x={x1:.6f} y={y1:.6f}: {exc}")
    return {"energy":E0,"xs":xs,"ys":ys,"classification":cls,"freq_drift":drift,
            "persistence":pers,"entropy":ent,"ftle":ft,"energy_error":ee,
            "reference":ref,"thresholds":thr}


def plot_results(r):
    xs,ys=r["xs"],r["ys"]
    cmap=ListedColormap(["#4477AA","#55AA55","#CC6677"])
    norm=BoundaryNorm([-0.5,0.5,1.5,2.5],3)
    fig,ax=plt.subplots(figsize=(7,6))
    m=ax.pcolormesh(xs,ys,r["classification"],shading="auto",cmap=cmap,norm=norm)
    ax.scatter([X_STAR],[Y_STAR],marker="*",s=160,color="black")
    ax.set(xlabel="initial x1",ylabel="initial y1",title="Case 4: fixed-energy 2-D slice classification")
    c=fig.colorbar(m,ax=ax,ticks=[0,1,2]); c.ax.set_yticklabels(["chaotic candidate","regular candidate","uncertain"])
    fig.tight_layout()

    fig2,ax2=plt.subplots(figsize=(7,6))
    m2=ax2.pcolormesh(xs,ys,r["freq_drift"],shading="auto")
    ax2.scatter([X_STAR],[Y_STAR],marker="*",s=160,color="black")
    ax2.set(xlabel="initial x1",ylabel="initial y1",title="Median spectral-line drift")
    fig2.colorbar(m2,ax=ax2,label="relative frequency drift")
    fig2.tight_layout()

    fig3,ax3=plt.subplots(figsize=(7,6))
    m3=ax3.pcolormesh(xs,ys,r["ftle"],shading="auto")
    ax3.scatter([X_STAR],[Y_STAR],marker="*",s=160,color="black")
    ax3.set(xlabel="initial x1",ylabel="initial y1",title="Largest finite-time Lyapunov exponent")
    fig3.colorbar(m3,ax=ax3,label="FTLE")
    fig3.tight_layout()
    return fig,fig2,fig3


if __name__=="__main__":
    # IMPORTANT: start with a tiny grid and long enough windows.
    # For a serious run, first test h vs h/2 and T vs 2T on the reference orbit.
    out=run_grid(nx=5,ny=5,x_half_width=0.05,y_half_width=0.05,
                 h=5e-5,total_time=40.0,sample_stride=20,max_lines=6,
                 do_ftle=False,renorm_steps=20)
    plot_results(out)
    plt.show()
