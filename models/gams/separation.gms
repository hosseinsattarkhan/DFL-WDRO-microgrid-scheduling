$Title Separation Problem for Single-Bus Microgrid (Exact MILP)
$Offlisting

* ============================================================
* Sets
* ============================================================
Set T /1*24/;
Set N /1*10/;
Alias(T, t);

* ============================================================
* Parameters
* ============================================================
Parameter u_gen_fixed(T);
Parameter pv_avail(N, T);
Parameter wind_avail(N, T);
Parameter price(N, T);
Parameter demand(T) / #T 1.0 /;
Parameter dg_cap /0.5/;
Parameter pv_cap /1.0/;
Parameter wind_cap /1.0/;
Parameter ess_cap /0.5/;
Parameter ess_eff /0.9/;
Parameter epsilon /0.1/;

* ============================================================
* Variables
* ============================================================
Variable min_cost;
Positive Variable pv_used(T);
Positive Variable wind_used(T);
Positive Variable dg_gen(T);
Positive Variable ess_ch(T);
Positive Variable ess_dis(T);
Positive Variable soc(T);
Positive Variable grid_buy(T);
Positive Variable grid_sell(T);

* ============================================================
* Equations
* ============================================================
Equation obj_def;
Equation power_balance(T);
Equation pv_limit(T);
Equation wind_limit(T);
Equation dg_limit(T);
Equation soc_dyn(T);

* Objective: minimize operational cost for given u_gen and scenario n
obj_def..
    min_cost =e= sum(t, price('1', t) * (grid_buy(t) - grid_sell(t) + dg_gen(t) * 0.1));

* Power balance
power_balance(t)..
    pv_used(t) * pv_cap + wind_used(t) * wind_cap + dg_gen(t)
    + ess_dis(t) - ess_ch(t) + grid_buy(t) - grid_sell(t)
    =e= demand(t);

* PV and Wind limits
pv_limit(t).. pv_used(t) =l= pv_avail('1', t);
wind_limit(t).. wind_used(t) =l= wind_avail('1', t);

* DG limit (u_gen fixed)
dg_limit(t).. dg_gen(t) =l= u_gen_fixed(t) * dg_cap;

* Battery SOC dynamics
soc_dyn(t)..
    soc(t) =e= soc(t-1) + ess_ch(t) * ess_eff - ess_dis(t) / ess_eff;

* ============================================================
* Bounds
* ============================================================
soc.lo(T) = 0;
soc.up(T) = ess_cap;
grid_buy.up(T) = 2.0;
grid_sell.up(T) = 2.0;

* ============================================================
* Model and Solve
* ============================================================
Model separation /all/;

Option LP = CPLEX;
separation.reslim = 300;  * 5 minutes
Solve separation using LP minimizing min_cost;

Display min_cost.l;