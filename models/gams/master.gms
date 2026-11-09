$Title Master Problem for Single-Bus Microgrid (C&CG)
$Offlisting

* ============================================================
* Sets (textual values to match the .inc file)
* ============================================================
Set T /'1'*'24'/;
Set N /'1'*'10'/;

* ============================================================
* Parameters
* ============================================================
Parameter pv_avail(N, T);
Parameter wind_avail(N, T);
Parameter price(N, T);

$include "C:/Users/Lenovo/Desktop/dfl_wdro_microgrid/models/gams/data_master.inc"

Parameter demand(T) / #T 1.0 /;
Scalar dg_cap /0.5/;
Scalar pv_cap /1.0/;
Scalar wind_cap /1.0/;
Scalar ess_cap /0.5/;
Scalar ess_eff /0.9/;

* ============================================================
* Variables
* ============================================================
Variable z;
Variable u_gen(T);
Positive Variable pv_used(N, T);
Positive Variable wind_used(N, T);
Positive Variable dg_gen(N, T);
Positive Variable ess_ch(N, T);
Positive Variable ess_dis(N, T);
Positive Variable soc(N, T);
Positive Variable grid_buy(N, T);
Positive Variable grid_sell(N, T);

* ============================================================
* Equations
* ============================================================
Equation obj_def;
Equation power_balance(N, T);
Equation pv_limit(N, T);
Equation wind_limit(N, T);
Equation dg_limit(N, T);
Equation soc_dyn(N, T);
Equation z_def(N);

obj_def.. z =e= z;

power_balance(n, t)..
    pv_used(n, t) * pv_cap + wind_used(n, t) * wind_cap + dg_gen(n, t)
    + ess_dis(n, t) - ess_ch(n, t) + grid_buy(n, t) - grid_sell(n, t)
    =e= demand(t);

pv_limit(n, t).. pv_used(n, t) =l= pv_avail(n, t);
wind_limit(n, t).. wind_used(n, t) =l= wind_avail(n, t);
dg_limit(n, t).. dg_gen(n, t) =l= u_gen(t) * dg_cap;

soc_dyn(n, t)..
    soc(n, t) =e= soc(n, t-1) + ess_ch(n, t) * ess_eff - ess_dis(n, t) / ess_eff;

z_def(n)..
    z =g= sum(t, price(n, t) * (grid_buy(n, t) - grid_sell(n, t) + dg_gen(n, t) * 0.1));

* ============================================================
* Bounds
* ============================================================
u_gen.lo(T) = 0;
u_gen.up(T) = 1;
soc.lo(N, T) = 0;
soc.up(N, T) = ess_cap;
grid_buy.up(N, T) = 2.0;
grid_sell.up(N, T) = 2.0;

* ============================================================
* Model and Solve
* ============================================================
Model master /all/;

Option MIP = CPLEX;
Option LP = CPLEX;

master.reslim = 3600;
Solve master using MIP minimizing z;

Display z.l, u_gen.l;