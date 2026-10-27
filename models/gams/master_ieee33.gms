$Title Master Problem for IEEE 33-bus Microgrid (C&CG)
$Offlisting

* ============================================================
* Sets
* ============================================================
Set T /T1*T24/;
Set N /N1*N3/;
Set B /B1*B33/;
Set L /L1*L32/;

* ============================================================
* Parameters
* ============================================================
Parameter pv_avail(N, T);
Parameter wind_avail(N, T);
Parameter price(N, T);
Parameter line_from_num(L);
Parameter line_to_num(L);
Parameter P_load(B);

$include "C:/Users/Lenovo/Desktop/dfl_wdro_microgrid/models/gams/ieee33_data.inc"

* DER capacities
Scalar pv_cap   /0.5/;
Scalar wind_cap /0.5/;
Scalar dg_cap   /0.3/;
Scalar ess_cap  /1.0/;
Scalar ess_eff  /0.9/;

* DER locations
Set pv_buses(B)   / B7,  B15 /;
Set wind_buses(B) / B10, B25 /;
Set dg_buses(B)   / B4,  B19, B31 /;
Set ess_buses(B)  / B13 /;

* ============================================================
* Variables
* ============================================================
Variable z;
Variable u_gen(T, B);
Variable P_flow(N, T, L);
Variable P_grid(N, T);

Positive Variable P_pv(N, T, B);
Positive Variable P_wind(N, T, B);
Positive Variable P_dg(N, T, B);
Positive Variable P_ess_ch(N, T, B);
Positive Variable P_ess_dis(N, T, B);
Positive Variable SOC(N, T, B);

* ============================================================
* Equations
* ============================================================
Equation obj_def;
Equation power_balance(N, T, B);
Equation pv_limit(N, T, B);
Equation wind_limit(N, T, B);
Equation dg_limit(N, T, B);
Equation soc_dyn(N, T, B);
Equation z_def(N);

obj_def.. z =e= z;

* Power balance with explicit grid injection at bus 1
power_balance(n, t, b)..
    sum(l$(line_to_num(l) = ord(b)), P_flow(n, t, l))
  - sum(l$(line_from_num(l) = ord(b)), P_flow(n, t, l))
  + P_grid(n, t)$(ord(b) = 1)
  + P_pv(n, t, b)$pv_buses(b)     * pv_cap
  + P_wind(n, t, b)$wind_buses(b) * wind_cap
  + P_dg(n, t, b)$dg_buses(b)     * dg_cap
  + P_ess_dis(n, t, b)$ess_buses(b)
  - P_ess_ch(n, t, b)$ess_buses(b)
  =e= P_load(b);

pv_limit(n, t, b)$pv_buses(b)..     P_pv(n, t, b)   =l= pv_avail(n, t);
wind_limit(n, t, b)$wind_buses(b).. P_wind(n, t, b) =l= wind_avail(n, t);
dg_limit(n, t, b)$dg_buses(b)..     P_dg(n, t, b)   =l= u_gen(t, b);

soc_dyn(n, t, b)$ess_buses(b)..
    SOC(n, t, b) =e= SOC(n, t-1, b)
                   + P_ess_ch(n, t, b) * ess_eff
                   - P_ess_dis(n, t, b) / ess_eff;

* Worst-case cost = grid exchange + DG fuel cost
z_def(n)..
    z =g= sum(t, price(n, t) * (
            P_grid(n, t)
          + 0.1 * sum(b$dg_buses(b), P_dg(n, t, b) * dg_cap)
        ));

* ============================================================
* Bounds
* ============================================================
u_gen.lo(T, B) = 0;
u_gen.up(T, B) = 1;
u_gen.fx(T, B)$(not dg_buses(B)) = 0;

SOC.lo(N, T, B) = 0;
SOC.up(N, T, B) = ess_cap;
SOC.fx(N, T, B)$(not ess_buses(B)) = 0;

P_pv.fx(N, T, B)$(not pv_buses(B))     = 0;
P_wind.fx(N, T, B)$(not wind_buses(B)) = 0;
P_dg.fx(N, T, B)$(not dg_buses(B))     = 0;
P_ess_ch.fx(N, T, B)$(not ess_buses(B)) = 0;
P_ess_dis.fx(N, T, B)$(not ess_buses(B)) = 0;

P_flow.lo(N, T, L) = -5;
P_flow.up(N, T, L) =  5;

P_grid.lo(N, T) = -2;
P_grid.up(N, T) =  5;

* ============================================================
* Model and Solve
* ============================================================
Model master_33 /all/;

Option MIP = CPLEX;
Option LP  = CPLEX;
master_33.reslim = 3600;

Solve master_33 using MIP minimizing z;

Display z.l, u_gen.l, P_grid.l;