from __future__ import annotations
import io, math, re, textwrap
from datetime import date, datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import sympy as sp
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")
st.title("Hispanic Heritage Historical Data Explorer")
st.caption("Hispanic Heritage Data Lab · authentic World Bank data + a student-friendly function laboratory")
st.info("Research question (customize): How has the selected indicator changed over time, and what can a polynomial model tell us—and not tell us—about its trend? Brazil is included as a Latin American comparison country, not as a Spanish-speaking country.")

COUNTRIES = {"Mexico":"MEX","Brazil":"BRA","Argentina":"ARG","Colombia":"COL","Chile":"CHL","Peru":"PER","Venezuela":"VEN","Ecuador":"ECU","Uruguay":"URY"}
INDICATORS = {
 "Life expectancy at birth":("SP.DYN.LE00.IN","years","Average number of years a newborn would live if current mortality patterns persist.","Higher values indicate longer life expectancy."),
 "Unemployment, national estimate":("SL.UEM.TOTL.NE.ZS","% of labor force","Share of the labor force without work but available for and seeking employment, national estimate.","Not the number of unemployed people."),
 "GDP per capita":("NY.GDP.PCAP.CD","current US$ per person","Gross domestic product divided by midyear population, in current US dollars.","Not average wealth or household income; current-dollar values are affected by prices and exchange rates."),
 "Crude birth rate":("SP.DYN.CBRT.IN","births per 1,000 people per year","Annual live births per 1,000 population.","A rate, not a birth count."),
 "Net migration":("SM.POP.NETM","people, net estimate","Net number of migrants over the period as defined by the World Bank indicator.","Not an emigration count; positive/negative values indicate net inflow/outflow."),
 "Intentional homicide rate":("VC.IHR.PSRC.P5","per 100,000 people","Intentional homicides per 100,000 population.","Reporting and coverage may differ across countries and years."),
 "Consumer price inflation":("FP.CPI.TOTL.ZG","% annual change","Annual percentage change in the consumer price index.","Inflation can be negative in some years."),
 "Central government debt":("GC.DOD.TOTL.GD.ZS","% of GDP","Central government debt as a percentage of GDP.","Coverage and reporting can vary."),
 "Total reserves including gold":("FI.RES.TOTL.CD","current US$","Total reserves including gold, reported in current US dollars.","Not a per-person measure."),
 "External balance on goods and services":("NE.RSB.GNFS.ZS","% of GDP","External balance on goods and services as a percentage of GDP.","Not the full current account.")
}
WB_ROOT = "https://api.worldbank.org/v2"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"{WB_ROOT}/country/{country_code}/indicator/{indicator_code}"
    try:
        response = requests.get(url, params={"format":"json","date":f"{start_year}:{end_year}","per_page":20000}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            return pd.DataFrame(columns=["year","value"]), f"World Bank returned no observations (HTTP {response.status_code})."
        rows = [{"year": int(row["date"]), "value": row.get("value")} for row in payload[1] if row.get("date")]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["year","value"]), "Response contained no year rows."
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        return df, ("" if not df.empty else "The indicator has no non-null values in this period.")
    except requests.Timeout:
        return pd.DataFrame(columns=["year","value"]), "World Bank API timed out. Retry later or inspect your network."
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year","value"]), f"Could not retrieve/parse World Bank data: {exc}"

def valid_working_data(df, indicator_name):
    errors=[]
    if df.empty: return ["There are no rows to fit."]
    if not {"year","value"}.issubset(df.columns): return ["Working table must have year and value columns."]
    years=pd.to_numeric(df["year"], errors="coerce")
    vals=pd.to_numeric(df["value"], errors="coerce")
    if years.isna().any() or not np.all(np.equal(years.dropna(), np.floor(years.dropna()))): errors.append("Years must be whole numbers.")
    if years.duplicated().any(): errors.append("Each year must be unique.")
    if vals.isna().any() or not np.isfinite(vals.dropna()).all(): errors.append("Values must be finite numbers (no blanks, text, or infinity).")
    if indicator_name == "Life expectancy at birth" and ((vals.dropna()<0).any() or (vals.dropna()>125).any()): errors.append("Life expectancy values must be between 0 and 125 years.")
    if indicator_name == "Unemployment, national estimate" and ((vals.dropna()<0).any() or (vals.dropna()>100).any()): errors.append("Unemployment rate must be between 0 and 100%.")
    if indicator_name == "Crude birth rate" and (vals.dropna()<0).any(): errors.append("Crude birth rate cannot be negative.")
    if indicator_name == "Intentional homicide rate" and (vals.dropna()<0).any(): errors.append("Homicide rate cannot be negative.")
    return errors

def polynomial_fit(df, degree):
    d=df.sort_values("year").copy()
    base=int(d["year"].min())
    t=d["year"].to_numpy(dtype=float)-base
    y=d["value"].to_numpy(dtype=float)
    if len(np.unique(t)) < degree+1: raise ValueError(f"Degree {degree} requires at least {degree+1} distinct years; only {len(np.unique(t))} available.")
    coeff=np.polyfit(t,y,degree)
    pred=np.polyval(coeff,t)
    return {"base":base,"degree":degree,"coeff":coeff,"years":d["year"].to_numpy(dtype=int),"values":y,"fitted":pred,"r2":float(r2_score(y,pred)) if len(y)>1 else float("nan"),"mae":float(mean_absolute_error(y,pred))}

def poly_value(model, years):
    return np.polyval(model["coeff"], np.asarray(years,dtype=float)-model["base"])

def equation_text(model, variable="t"):
    parts=[]
    d=model["degree"]
    for i,c in enumerate(model["coeff"]):
        power=d-i
        if abs(c)<1e-12: continue
        term=f"{c:.8g}" if power==0 else (f"{c:.8g}{variable}" if power==1 else f"{c:.8g}{variable}^{power}")
        parts.append(term)
    return "f(t) = " + (" + ".join(parts).replace("+ -","- ") if parts else "0") + f", where t = year − {model['base']}."

CONTEXTS = {
 "Business":("Revenue","currency units","Revenue is the modeled output; the derivative estimates change in revenue per time unit.","x + 0.1*x**2"),
 "Engineering":("Position","meters","The first derivative is velocity (m/s) and the second derivative is acceleration (m/s²), when x is seconds.","0.5*x**2"),
 "Medical":("Drug concentration","mg/L","The model output is concentration; its derivative is concentration change per time unit. This is educational, not clinical dosing advice.","10*exp(-0.3*x)"),
 "Sports science":("Jump height","meters","The function represents a measured performance quantity; derivatives describe its rate and rate-of-rate of change.","-0.5*x**2 + 2*x + 1"),
 "Biology":("Population index","organisms","The derivative describes population change per time unit; the second derivative describes how that rate changes.","100*exp(0.08*x)"),
 "Bioengineering":("Signal response","arbitrary units","The derivative describes how quickly the response changes with the input.","sin(x) + 0.2*x"),
 "Economic forecasting":("Price index","index points","The first derivative is index points per time unit; the second derivative indicates whether the rate is accelerating or slowing.","0.02*x**2 + x + 100"),
 "Weather":("Temperature","°C","The first derivative is degrees per time unit and the second derivative is the change in that rate.","10 + 5*sin(x/3)"),
 "Custom":("Output","units","Describe the meaning and units of your own function.","x**2")
}

def parse_function(expr):
    x=sp.symbols("x", real=True)
    # Only allow explicit math names/operators; no eval or Python execution.
    if len(expr)>180: raise ValueError("Function expression is too long.")
    if not re.fullmatch(r"[0-9xX+\-*/().,^ a-zA-Z_]+", expr): raise ValueError("Use mathematical notation with numbers, x, + - * / ** ^, parentheses, and supported functions.")
    expr=expr.replace("^","**").replace("X","x")
    local={"x":x,"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,"ln":sp.log,"sqrt":sp.sqrt,"pi":sp.pi,"abs":sp.Abs}
    try: f=sp.sympify(expr, locals=local, evaluate=True)
    except Exception as e: raise ValueError(f"Could not read that expression: {e}")
    if f.free_symbols - {x}: raise ValueError("Use x as the only variable.")
    return x, f

def numeric_roots(expr, x, lo, hi, samples=1600):
    fn=sp.lambdify(x,expr,"numpy")
    grid=np.linspace(lo,hi,samples)
    try: yy=np.asarray(fn(grid),dtype=float)
    except Exception: return []
    if yy.ndim==0: yy=np.full_like(grid,float(yy))
    yy=np.broadcast_to(yy,grid.shape)
    roots=[]
    for i in range(len(grid)-1):
        if not (np.isfinite(yy[i]) and np.isfinite(yy[i+1])): continue
        if abs(yy[i])<1e-5: roots.append(grid[i])
        elif yy[i]*yy[i+1]<0:
            a,b=grid[i],grid[i+1]
            for _ in range(45):
                m=(a+b)/2
                try: fm=float(fn(m))
                except Exception: break
                if not np.isfinite(fm): break
                if yy[i]*fm<=0: b=m
                else: a=m
            roots.append((a+b)/2)
    out=[]
    for r in roots:
        if not out or abs(r-out[-1])>1e-3: out.append(float(r))
    return out

# Sidebar controls
with st.sidebar:
    st.header("Historical data")
    country_name=st.selectbox("Featured country",list(COUNTRIES),index=0)
    indicator_name=st.selectbox("Indicator",list(INDICATORS),index=0)
    degree=st.slider("Polynomial degree",3,8,3)
    sample_step=st.slider("Model marker / sample-table increment (years)",1,10,5)
    horizon=st.slider("Forward display horizon (years)",0,50,10)
    comparison=st.multiselect("Compare same indicator with", [c for c in COUNTRIES if c != country_name], default=["Brazil"] if country_name!="Brazil" else ["Mexico"])
    printer_mode=st.toggle("Printer-friendly report view",False)
    st.divider()
    st.header("Function lab")
    course=st.selectbox("Mathematics pathway",["Honors Algebra 2","Precalculus","Calculus"])
    context=st.selectbox("Context",list(CONTEXTS))
    mode=st.radio("Function source",["Historical curve of best fit","Suggested context function","Build polynomial","Enter a function"])
    st.caption("Custom expressions support polynomial, rational, exponential, logarithmic and trigonometric forms.")
    expr=st.text_input("Function f(x)",value=CONTEXTS[context][3],help="Use x, for example: x**2, 1/(x-2), exp(0.2*x), log(x), sin(x).") if mode=="Enter a function" else CONTEXTS[context][3]
    custom_degree=st.slider("Custom polynomial degree",1,6,2) if mode=="Build polynomial" else 2
    p1=st.number_input("Parameter a",value=1.0,step=0.5) if mode=="Build polynomial" else 1.0
    p2=st.number_input("Parameter b",value=0.0,step=0.5) if mode=="Build polynomial" else 0.0
    p3=st.number_input("Parameter c",value=0.0,step=0.5) if mode=="Build polynomial" else 0.0

now_year=date.today().year
start_year=now_year-70
end_year=now_year-1
code,units,definition,interpretation=INDICATORS[indicator_name]
st.caption(f"World Bank coverage requested: {start_year}–{end_year} inclusive. Retrieved {date.today().isoformat()}.")

raw_df, fetch_message = fetch_indicator(COUNTRIES[country_name],code,start_year,end_year)
if fetch_message: st.warning(fetch_message)
if raw_df.empty:
    st.error("No authentic observations are available for this selection. No synthetic values will be generated; try another indicator/country or retry later.")
    st.stop()

if "edited_df" not in st.session_state or st.session_state.get("edit_key") != (country_name,indicator_name):
    st.session_state.edited_df=raw_df.copy()
    st.session_state.edit_key=(country_name,indicator_name)
if st.button("Reset working data to source"):
    st.session_state.edited_df=raw_df.copy()
working = st.session_state.edited_df.copy()
st.subheader(f"{country_name}: {indicator_name}")
st.write(definition)
st.markdown(f"**Units:** {units} · **Source:** [World Bank indicator API]({WB_ROOT}/country/{COUNTRIES[country_name]}/indicator/{code}?format=json&date={start_year}%3A{end_year})")
st.caption(f"Source observations available: {len(raw_df)}; actual coverage: {int(raw_df.year.min())}–{int(raw_df.year.max())}. Missing years are not imputed.")

with st.expander("Edit the working year/value table",expanded=False):
    edited = st.data_editor(working, num_rows="dynamic", use_container_width=True, key=f"editor_{country_name}_{indicator_name}")
    st.session_state.edited_df=edited
    errors=valid_working_data(edited,indicator_name)
    if errors:
        for e in errors: st.error(e)
    elif not edited.equals(raw_df):
        st.warning("User-modified scenario: model and exports use the edited table until Reset to source is selected.")
working=st.session_state.edited_df.copy()
errors=valid_working_data(working,indicator_name)
if errors:
    st.error("Fix the working data errors before fitting: " + " ".join(errors))
    st.stop()
working=working.sort_values("year").reset_index(drop=True)
if len(working)<degree+1:
    st.error(f"Degree {degree} needs at least {degree+1} distinct observations. Lower the degree or restore/add valid years.")
    st.stop()
try: model=polynomial_fit(working,degree)
except Exception as e:
    st.error(str(e)); st.stop()
edited_flag=not working.reset_index(drop=True).equals(raw_df.sort_values("year").reset_index(drop=True))
st.subheader("Polynomial model")
m1,m2,m3,m4=st.columns(4)
m1.metric("Observations",len(working))
m2.metric("In-sample R²",f"{model['r2']:.5f}")
m3.metric("In-sample MAE",f"{model['mae']:.5g} {units}")
m4.metric("Observed mean",f"{working.value.mean():.5g} {units}")
st.markdown(f"**Model:** `{equation_text(model)}`")
st.caption(f"Centered time variable t = year − {model['base']}; degree {degree}. Metrics are in-sample and are not proof of predictive accuracy.")
resid=pd.DataFrame({"year":model["years"],"actual":model["values"],"fitted_at_observed_year":model["fitted"],"residual_actual_minus_fitted":model["values"]-model["fitted"]})
with st.expander("Residuals and chronological holdout check"):
    st.dataframe(resid,use_container_width=True)
    if len(working)>=max(12,degree+5):
        split=max(degree+1,int(len(working)*0.8))
        train=working.iloc[:split]; test=working.iloc[split:]
        if len(test) and len(train)>=degree+1:
            hm=polynomial_fit(train,degree)
            hp=poly_value(hm,test.year.to_numpy())
            st.write(f"Chronological holdout (last {len(test)} observations, fit on earlier years): MAE = {mean_absolute_error(test.value,hp):.5g} {units}; R² = {r2_score(test.value,hp):.5f} (holdout score, separate from in-sample metrics).")
    else: st.write("Not enough observations for the configured chronological holdout check; need at least max(12, degree + 5) rows.")

st.subheader("Estimate a year")
if "single_year" not in st.session_state: st.session_state.single_year=int(working.year.max())
with st.form("single_year_form"):
    year_input=st.number_input("Year (1950–2100)",min_value=1950,max_value=2100,value=int(np.clip(st.session_state.single_year,1950,2100)),step=1)
    do_single=st.form_submit_button("Calculate value")
if do_single: st.session_state.single_year=int(year_input)
single_year=int(st.session_state.single_year)
single_est=float(poly_value(model,[single_year])[0])
if do_single or "single_result" not in st.session_state or st.session_state.get("result_key") != (country_name,indicator_name,degree,tuple(working.year),tuple(np.round(working.value,10))):
    st.session_state.single_result=(single_year,single_est)
    st.session_state.result_key=(country_name,indicator_name,degree,tuple(working.year),tuple(np.round(working.value,10)))
within=int(working.year.min())<=single_year<=int(working.year.max())
st.metric(f"Fitted estimate for {single_year}",f"{single_est:.7g} {units}")
st.write(f"Classification: **{'within' if within else 'outside'} the observed fitted range** ({int(working.year.min())}–{int(working.year.max())}). This is a model estimate, not necessarily an observed value.")
if single_year in set(working.year.astype(int)):
    actual=float(working.loc[working.year.astype(int)==single_year,"value"].iloc[0])
    st.write(f"At an observed year: actual = {actual:.7g} {units}; fitted estimate = {single_est:.7g} {units}. These are different quantities.")

st.subheader("Batch estimates")
batch_mode=st.radio("Batch input",["Year range","Custom years"],horizontal=True)
batch_years=[]
if batch_mode=="Year range":
    c1,c2,c3=st.columns(3)
    with c1: bstart=st.number_input("Start year",1950,2100,min(working.year.min(),2100),key="bstart")
    with c2: bend=st.number_input("End year",1950,2100,min(max(working.year.max(),int(bstart)),2100),key="bend")
    with c3: bstep=st.slider("Range step",1,10,1)
    if bstart<=bend: batch_years=list(range(int(bstart),int(bend)+1,int(bstep)))
    else: st.error("Start year must be less than or equal to end year.")
else:
    custom_years=st.text_input("Comma-separated years",value=f"{single_year}, {int(working.year.max())}")
    try:
        tokens=[z.strip() for z in custom_years.split(",") if z.strip()]
        parsed=[int(z) for z in tokens]
        if any(str(y)!=z and str(y)!=z.lstrip("+") for y,z in zip(parsed,tokens)): raise ValueError("Years must be integer values.")
        if any(y<1950 or y>2100 for y in parsed): raise ValueError("Each year must be between 1950 and 2100.")
        if len(set(parsed))!=len(parsed): raise ValueError("Duplicate years are not allowed.")
        batch_years=parsed
    except Exception as e: st.error(f"Invalid custom years: {e}")
batch_results=pd.DataFrame(columns=["year","estimate","units","range_classification"])
if batch_years:
    batch_results=pd.DataFrame({"year":batch_years,"estimate":poly_value(model,batch_years)})
    batch_results["units"]=units
    batch_results["range_classification"]=batch_results.year.apply(lambda y:"within observed range" if working.year.min()<=y<=working.year.max() else "outside observed range")
    st.dataframe(batch_results,use_container_width=True)
    match=batch_results.loc[batch_results.year==single_year,"estimate"]
    if not match.empty: st.caption(f"Agreement check: batch estimate for {single_year} = {match.iloc[0]:.10g}; single-year estimate = {single_est:.10g}.")

st.subheader("Historical trend and polynomial fit")
plot_end=int(working.year.max())+horizon
smooth_years=np.linspace(float(working.year.min()),float(plot_end),max(400,plot_end-int(working.year.min())+1))
smooth_vals=poly_value(model,smooth_years)
fig,ax=plt.subplots(figsize=(11,5.2))
ax.scatter(working.year,working.value,label="Observed / edited data",s=26,zorder=3)
ax.plot(smooth_years,smooth_vals,label="Smooth polynomial fit",linewidth=2)
sample=working[(working.year.astype(int)-int(working.year.min()))%sample_step==0]
ax.scatter(sample.year,poly_value(model,sample.year),marker="x",s=55,label=f"Model markers every {sample_step} year(s)")
ax.axvline(int(working.year.max()),linestyle="--",label="Historical boundary")
if horizon>0:
    hist_end=int(working.year.max())
    mask=smooth_years>=hist_end
    ax.plot(smooth_years[mask],smooth_vals[mask],linestyle=":",linewidth=2.5,label="Extrapolated segment")
if batch_years: ax.scatter(batch_years,poly_value(model,batch_years),marker="D",s=35,label="Batch estimates")
ax.scatter([single_year],[single_est],marker="*",s=150,label=f"Single estimate ({single_year})")
ax.set_xlabel("Year"); ax.set_ylabel(units); ax.grid(alpha=.25); ax.legend(loc="best",fontsize=8)
st.pyplot(fig,use_container_width=True)
st.warning("Polynomial extrapolation can grow implausibly outside the observed years. No estimates are clipped. MAE describes typical in-sample absolute error, not a validated 95% prediction interval.")
st.write(f"Historical model domain shown: {int(working.year.min())}–{int(working.year.max())}; displayed horizon: through {plot_end}. Polynomial end behavior is controlled by the leading coefficient and degree, and may not reflect real-world limits.")

if comparison:
    st.subheader("Country comparison (same indicator)")
    comp_fig,comp_ax=plt.subplots(figsize=(11,4.6))
    comp_ax.plot(working.year,working.value,marker="o",markersize=3,label=f"{country_name} (working data)")
    compare_notes=[]
    for cn in comparison:
        cdf,msg=fetch_indicator(COUNTRIES[cn],code,start_year,end_year)
        if not cdf.empty:
            comp_ax.plot(cdf.year,cdf.value,marker=".",markersize=3,label=cn)
            compare_notes.append(f"{cn}: {len(cdf)} observations, {int(cdf.year.min())}–{int(cdf.year.max())}")
        else: compare_notes.append(f"{cn}: no observations returned ({msg})")
    comp_ax.set_xlabel("Year"); comp_ax.set_ylabel(units); comp_ax.grid(alpha=.25); comp_ax.legend()
    st.pyplot(comp_fig,use_container_width=True)
    st.caption("Coverage can differ because country reporting and indicator availability differ. " + " · ".join(compare_notes))

st.subheader("Function laboratory")
if mode=="Historical curve of best fit":
    x=sp.symbols("x", real=True)
    t_sym=sp.symbols("t", real=True)
    f=sum(sp.Float(float(c),12)*(x-model["base"])**(degree-i) for i,c in enumerate(model["coeff"]))
    f=sp.expand(f)
    x_label="Year"
    context_description=f"World Bank {indicator_name} for {country_name}; x is calendar year and f(x) is {units}."
elif mode=="Suggested context function":
    x,f=parse_function(CONTEXTS[context][3]); x_label="Input"; context_description=CONTEXTS[context][2]
elif mode=="Build polynomial":
    x=sp.symbols("x",real=True)
    f=sp.expand(p1*x**custom_degree+p2*x+p3)
    x_label="Input"; context_description=CONTEXTS[context][2]
else:
    try: x,f=parse_function(expr); x_label="Input"; context_description=CONTEXTS[context][2]
    except Exception as e:
        st.error(f"Function input error: {e}")
        st.stop()

x_name=st.text_input("Independent variable name",value="year" if mode=="Historical curve of best fit" else "time / input")
y_name=st.text_input("Dependent variable name",value=indicator_name if mode=="Historical curve of best fit" else CONTEXTS[context][0])
x_units=st.text_input("Independent variable units",value="calendar year" if mode=="Historical curve of best fit" else "time units")
y_units=st.text_input("Dependent variable units",value=units if mode=="Historical curve of best fit" else CONTEXTS[context][1])
context_description=st.text_area("Context description / interpretation",value=context_description)
if mode=="Historical curve of best fit":
    default_lo=float(working.year.min()); default_hi=float(min(max(working.year.max(),default_lo+1),2100))
else: default_lo=0.0; default_hi=10.0
col_a,col_b=st.columns(2)
with col_a: x_lo=st.number_input("Analysis interval start",value=float(default_lo),key="xlo")
with col_b: x_hi=st.number_input("Analysis interval end",value=float(default_hi),key="xhi")
if x_hi<=x_lo:
    st.error("Analysis interval end must be greater than start."); st.stop()
if course=="Honors Algebra 2": st.write("**Algebra 2 focus:** domain and range on the chosen interval, graph trends, end behavior, and average rates of change.")
elif course=="Precalculus": st.write("**Precalculus focus:** degree and coefficient effects, turning points, concavity, and how average rates change.")
else: st.write("**Calculus focus:** derivatives, tangent/secant lines, critical points, extrema, concavity, and verified inflection points.")

f1=sp.diff(f,x); f2=sp.diff(f1,x)
st.markdown(f"**Function:** `{sp.sstr(f)}`")
st.markdown(f"**First derivative:** `{sp.sstr(f1)}`")
st.markdown(f"**Second derivative:** `{sp.sstr(f2)}`")
st.caption("Symbolic expressions are exact where SymPy can represent them exactly; roots and plotted values below are numerical approximations.")
if "lab_x" not in st.session_state: st.session_state.lab_x=float((x_lo+x_hi)/2)
if "animation_playing" not in st.session_state: st.session_state.animation_playing=False
play_col, pause_col, reset_col, speed_col = st.columns([1,1,1,2])
with play_col:
    if st.button("▶ Play", use_container_width=True): st.session_state.animation_playing=True
with pause_col:
    if st.button("⏸ Pause", use_container_width=True): st.session_state.animation_playing=False
with reset_col:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.animation_playing=False
        st.session_state.lab_x=float((x_lo+x_hi)/2)
with speed_col:
    playback_speed=st.select_slider("Playback speed",options=[0.5,1.0,1.5,2.0,3.0],value=1.0)
if st.session_state.animation_playing:
    st_autorefresh(interval=int(500/playback_speed), key="function_lab_autorefresh")
    delta=(x_hi-x_lo)/100.0*playback_speed
    next_x=float(st.session_state.lab_x)+delta
    st.session_state.lab_x=float(x_lo if next_x>x_hi else next_x)
lab_x=st.slider("Current input x",float(x_lo),float(x_hi),float(np.clip(st.session_state.lab_x,x_lo,x_hi)),step=float((x_hi-x_lo)/200 or .01),key="lab_x_slider")
st.session_state.lab_x=float(lab_x)
# Keep the input and interval endpoints synchronized through visible controls.
ac1,ac2=st.columns(2)
with ac1: a=st.slider("Average-rate interval start a",float(x_lo),float(x_hi),float(x_lo+(x_hi-x_lo)*.25),key="interval_a")
with ac2: b=st.slider("Average-rate interval end b",float(x_lo),float(x_hi),float(x_lo+(x_hi-x_lo)*.75),key="interval_b")
if b<=a: st.warning("For average rate of change, endpoint b must be greater than a.")
def scalar_value(expr, xx):
    fn=sp.lambdify(x,expr,"numpy")
    try:
        val=fn(float(xx))
        return float(val) if np.isreal(val) and np.isfinite(val) else float("nan")
    except Exception: return float("nan")
y0=scalar_value(f,lab_x); d0=scalar_value(f1,lab_x); dd0=scalar_value(f2,lab_x)
if np.isfinite(y0):
    k1,k2,k3=st.columns(3); k1.metric(f"{y_name} f(x)",f"{y0:.6g}"); k2.metric("First derivative f′(x)",f"{d0:.6g}" if np.isfinite(d0) else "undefined"); k3.metric("Second derivative f″(x)",f"{dd0:.6g}" if np.isfinite(dd0) else "undefined")
    st.write(f"**Context interpretation:** {context_description} At x = {lab_x:.5g}, the modeled {y_name.lower()} is {y0:.5g} {y_units}. Its rate is {d0:.5g} {y_units} per {x_units} and the rate-of-rate is {dd0:.5g} {y_units} per {x_units}², where defined.")
else: st.warning("The function is undefined at the selected input; move the slider away from a domain restriction.")
fa=scalar_value(f,a); fb=scalar_value(f,b)
if b>a and np.isfinite(fa) and np.isfinite(fb):
    avg=(fb-fa)/(b-a)
    sec_y=fa+avg*(lab_x-a)
    st.write(f"Average rate of change on [{a:.5g}, {b:.5g}] = ({fb:.6g} − {fa:.6g}) / ({b:.5g} − {a:.5g}) = **{avg:.6g} {y_units} per {x_units}**.")
else: avg=float("nan"); st.write("Average rate unavailable: check interval order and function domain.")

# Graph uses NaNs to break lines at undefined values and asymptotes.
grid=np.linspace(x_lo,x_hi,1000)
def vector_eval(expr, xs):
    fn=sp.lambdify(x,expr,"numpy")
    try:
        vals=np.asarray(fn(xs),dtype=float)
        if vals.ndim==0: vals=np.full_like(xs,float(vals))
        vals=np.broadcast_to(vals,xs.shape).astype(float)
        vals[~np.isfinite(vals)]=np.nan
        vals[np.abs(vals)>1e8]=np.nan
        return vals
    except Exception: return np.full_like(xs,np.nan,dtype=float)
yv=vector_eval(f,grid); d1v=vector_eval(f1,grid); d2v=vector_eval(f2,grid)
show_f=st.checkbox("Show f(x)",True); show_d1=st.checkbox("Show f′(x)",course=="Calculus"); show_d2=st.checkbox("Show f″(x)",course=="Calculus")
fig2,ax2=plt.subplots(figsize=(10,4.8))
if show_f: ax2.plot(grid,yv,label="f(x)")
if show_d1: ax2.plot(grid,d1v,label="f′(x)")
if show_d2: ax2.plot(grid,d2v,label="f″(x)")
if np.isfinite(y0):
    ax2.scatter([lab_x],[y0],s=70,label="Current point",zorder=5)
    if np.isfinite(d0): ax2.plot(grid, y0+d0*(grid-lab_x),linestyle="--",alpha=.8,label="Tangent line")
if b>a and np.isfinite(fa) and np.isfinite(fb):
    ax2.plot([a,b],[fa,fb],linestyle=":",linewidth=2,label="Secant line")
ax2.set_xlabel(f"{x_name} ({x_units})"); ax2.set_ylabel(f"{y_name} ({y_units})"); ax2.grid(alpha=.25); ax2.legend()
st.pyplot(fig2,use_container_width=True)

if course=="Calculus":
    st.markdown("#### Derivative-based analysis on the selected interval")
    crit=numeric_roots(f1,x,x_lo,x_hi)
    inflect_candidates=numeric_roots(f2,x,x_lo,x_hi)
    def classify_critical(r):
        eps=max((x_hi-x_lo)*1e-4,1e-5)
        left=scalar_value(f1,max(x_lo,r-eps)); right=scalar_value(f1,min(x_hi,r+eps))
        if left<0<right: return "local minimum (f′ changes − to +)"
        if left>0>right: return "local maximum (f′ changes + to −)"
        return "stationary point; no sign-change extremum verified"
    crit_rows=[]
    for r in crit:
        crit_rows.append({"x (approx.)":r,"f(x)":scalar_value(f,r),"classification":classify_critical(r)})
    if crit_rows: st.dataframe(pd.DataFrame(crit_rows),use_container_width=True)
    else: st.write("No numerical critical points found in the selected interval. Domain breaks may still exist.")
    # Increasing/decreasing and concavity sample intervals are reported from signs between roots.
    breaks=sorted([x_lo]+[r for r in crit if x_lo+1e-4<r<x_hi-1e-4]+[x_hi])
    trend=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        mid=(lo+hi)/2; dv=scalar_value(f1,mid)
        trend.append(f"{'Increasing' if dv>0 else 'Decreasing' if dv<0 else 'approximately constant'} on ({lo:.4g}, {hi:.4g})")
    st.write("**Monotonicity (numerically sampled):** " + ("; ".join(trend) if trend else "Unable to determine."))
    infl_rows=[]
    for r in inflect_candidates:
        eps=max((x_hi-x_lo)*1e-4,1e-5)
        left=scalar_value(f2,max(x_lo,r-eps)); right=scalar_value(f2,min(x_hi,r+eps))
        changes=np.isfinite(left) and np.isfinite(right) and left*right<0
        infl_rows.append({"x candidate (approx.)":r,"f(x)":scalar_value(f,r),"concavity changes?": "yes, numerically verified" if changes else "not verified"})
    st.write("**Concavity / inflection candidates:**")
    if infl_rows: st.dataframe(pd.DataFrame(infl_rows),use_container_width=True)
    else: st.write("No numerical zeros of f″ found in the interval.")
    endpoints=[(x_lo,scalar_value(f,x_lo)),(x_hi,scalar_value(f,x_hi))]
    valid_ext=[q for q in endpoints if np.isfinite(q[1])]
    valid_ext += [(r,scalar_value(f,r)) for r in crit if np.isfinite(scalar_value(f,r))]
    if valid_ext:
        st.write("**Absolute extrema candidates on the closed interval** (numerical; include endpoints and differentiable critical points):")
        st.write(f"Minimum candidate: x={min(valid_ext,key=lambda q:q[1])[0]:.6g}, f(x)={min(valid_ext,key=lambda q:q[1])[1]:.6g}; maximum candidate: x={max(valid_ext,key=lambda q:q[1])[0]:.6g}, f(x)={max(valid_ext,key=lambda q:q[1])[1]:.6g}.")
    st.caption("Root finding is numerical and may miss even-multiplicity roots or roots near singularities. These are candidates, not a proof of a complete symbolic analysis.")
elif course=="Precalculus":
    st.write("**Turning-point preview:** inspect where the first derivative changes sign; concavity is indicated by the sign of the second derivative. Numerical root search is approximate.")
else:
    st.write(f"**Interval behavior:** f({x_lo:.4g}) = {scalar_value(f,x_lo):.5g}; f({x_hi:.4g}) = {scalar_value(f,x_hi):.5g}. Average rates across subintervals describe the graph's trend; polynomial end behavior is governed by its leading term.")

st.subheader("Visual simulation")
visual=st.selectbox("Choose a context visual",["Walking person with prosthetic leg","Moving object / projectile","Tank or drug concentration","Population / cell colony","Thermometer / weather","Business / economic indicator"])
scale=st.slider("Visual scale (display only)",0.1,5.0,1.0,0.1)
st.caption("Simulation is a schematic mapping of the selected function value to a visual indicator; scale changes only the display, not the mathematical model.")
visual_fig,visual_ax=plt.subplots(figsize=(8,1.8))
if np.isfinite(y0):
    normalized=float(np.clip((y0 if abs(y0)<1e6 else 0)*scale,-10,10))
else: normalized=0
visual_ax.set_xlim(0,10); visual_ax.set_ylim(0,2); visual_ax.axis("off")
visual_ax.text(.2,1.45,visual,fontsize=10,weight="bold")
visual_ax.plot([.5,9.5],[.65,.65],linewidth=8,alpha=.2,solid_capstyle="round")
visual_ax.plot([.5,9.5],[.65,.65],linewidth=1)
pos=float(np.clip(5+normalized*.35,.6,9.4))
visual_ax.scatter([pos],[.65],s=230)
visual_ax.text(pos,.12,f"scaled f(x) = {normalized:.3g}",ha="center",fontsize=9)
st.pyplot(visual_fig,use_container_width=True)
st.write("Playback uses timed Streamlit reruns while Play is active. Pause stops the reruns; Reset returns the moving input to the interval midpoint. The graph point, tangent, displayed values, and schematic are recalculated from the same current input.")

# Exports: all represent current selection/model and working edits.
summary = pd.DataFrame([{"country":country_name,"country_code":COUNTRIES[country_name],"indicator":indicator_name,"indicator_code":code,"units":units,"definition":definition,"retrieved_on":date.today().isoformat(),"requested_years":f"{start_year}-{end_year}","actual_coverage":f"{int(working.year.min())}-{int(working.year.max())}","observation_count":len(working),"degree":degree,"base_year":model["base"],"equation":equation_text(model),"in_sample_R2":model["r2"],"in_sample_MAE":model["mae"],"user_modified_scenario":edited_flag}])
export_work=working.copy()
export_work["fitted_estimate"]=poly_value(model,export_work.year)
csv_bytes=export_work.to_csv(index=False).encode("utf-8")
xlsx_io=io.BytesIO()
with pd.ExcelWriter(xlsx_io,engine="xlsxwriter") as writer:
    summary.to_excel(writer,index=False,sheet_name="Summary")
    export_work.to_excel(writer,index=False,sheet_name="Working data")
    batch_results.to_excel(writer,index=False,sheet_name="Calculated years")
    workbook=writer.book
    for sheet in ["Summary","Working data","Calculated years"]:
        ws=writer.sheets[sheet]; ws.freeze_panes(1,0); ws.set_column(0,12,20)
pdf_io=io.BytesIO()
pdf_doc=SimpleDocTemplate(pdf_io,pagesize=letter,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
styles=getSampleStyleSheet(); story=[]
story.append(Paragraph("Hispanic Heritage Historical Data Explorer",styles["Title"]))
story.append(Paragraph(f"{country_name} · {indicator_name}",styles["Heading2"]))
story.append(Paragraph(f"Retrieved: {date.today().isoformat()} | Units: {units} | Coverage: {int(working.year.min())}–{int(working.year.max())} | Observations: {len(working)}",styles["BodyText"]))
story.append(Spacer(1,8))
story.append(Paragraph(definition,styles["BodyText"]))
story.append(Spacer(1,8))
story.append(Paragraph(equation_text(model),styles["BodyText"]))
story.append(Paragraph(f"Degree {degree}; in-sample R²={model['r2']:.6g}; in-sample MAE={model['mae']:.6g} {units}; working data user-modified: {edited_flag}.",styles["BodyText"]))
story.append(Paragraph("Limitations: polynomial fits can overfit and extrapolate implausibly. MAE is not a prediction interval. World Bank missing values are not imputed.",styles["BodyText"]))
story.append(Spacer(1,8))
img_buf=io.BytesIO(); fig.savefig(img_buf,format="png",dpi=130,bbox_inches="tight"); img_buf.seek(0)
story.append(RLImage(img_buf,width=500,height=240))
story.append(Paragraph("Working data preview",styles["Heading2"]))
preview=[["Year","Value","Fitted estimate"]]+[[str(int(r.year)),f"{r.value:.6g}",f"{r.fitted_estimate:.6g}"] for r in export_work.head(18).itertuples()]
tbl=Table(preview,repeatRows=1)
tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
story.append(tbl)
if not batch_results.empty:
    story.append(Paragraph("Calculated-year results",styles["Heading2"]))
    bprev=[["Year","Estimate","Units","Range"]]+[[str(int(r.year)),f"{r.estimate:.6g}",r.units,r.range_classification] for r in batch_results.head(20).itertuples()]
    story.append(Table(bprev,repeatRows=1))
pdf_doc.build(story)
d1,d2,d3=st.columns(3)
d1.download_button("Download CSV",csv_bytes,file_name="historical_working_data.csv",mime="text/csv")
d2.download_button("Download Excel workbook",xlsx_io.getvalue(),file_name="historical_explorer.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
d3.download_button("Download PDF report",pdf_io.getvalue(),file_name="historical_explorer_report.pdf",mime="application/pdf")
if printer_mode:
    st.divider()
    st.subheader("Printer-friendly summary")
    st.write(f"{country_name} — {indicator_name}; coverage {int(working.year.min())}–{int(working.year.max())}; n={len(working)}.")
    st.write(equation_text(model))
    st.write(f"In-sample R² {model['r2']:.5g}; MAE {model['mae']:.5g} {units}.")
    st.write("Caution: this fitted polynomial is descriptive; extrapolation is uncertain and should not be interpreted as a causal explanation.")
st.caption("Source data: World Bank API. Model outputs are educational estimates, not official forecasts or causal claims.")
