import json
import time

import holidays
import numpy as np
import pandas as pd
import requests

from .config import (ANO_FIM, ANO_INICIO, BASE_CDD, BASE_HDD, BCB_SGS, BRUTOS, CIDADES, DATOS_AR,
                     DEFASAGEM_DIVULGACAO_DIAS, GRUPOS_CLIMA, INTERMEDIARIOS, NORMAL_CLIMA, ONS_S3, OPEN_METEO)


# Baixa um arquivo da internet e salva em disco; se já existir, usa o arquivo salvo (cache).
def baixar(url, destino, params=None):
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        for tentativa in range(5):
            r = requests.get(url, params=params, timeout=300)
            if r.ok:
                break
            time.sleep(30 * (tentativa + 1))
        r.raise_for_status()
        destino.write_bytes(r.content)
    return destino


# Salva uma tabela limpa em data/interim/ no formato Parquet.
def salvar(df, nome):
    df.to_parquet(INTERMEDIARIOS / f"{nome}.parquet", index=False)
    print(f"{nome}: {len(df):,} linhas")


# Converte textos de data do ONS (com ou sem horário) para datas sem horário.
def data(serie):
    return pd.to_datetime(serie, format="ISO8601").dt.normalize()


# Baixa e junta os CSVs anuais de um conjunto de dados do ONS.
def ler_ons(chave, pasta, prefixo, anos):
    arquivos = [baixar(f"{ONS_S3}/{pasta}/{prefixo}_{a}.csv", BRUTOS / "ons" / chave / f"{prefixo}_{a}.csv")
                for a in anos]
    df = pd.concat([pd.read_csv(a, sep=";") for a in arquivos], ignore_index=True)
    df.columns = df.columns.str.lower()
    for c in df.select_dtypes("object"):
        df[c] = df[c].str.strip()
    return df


# Coleta e limpa os dados do ONS: intercâmbio, ENA, EAR, carga, CMO e energia vertida turbinável.
def coletar_ons():
    anos = range(ANO_INICIO, ANO_FIM + 1)

    i = ler_ons("intercambio_internacional", "intercambio_internacional_ho", "INTERCAMBIO_INTERNACIONAL", anos)
    i["din_instante"] = pd.to_datetime(i["din_instante"], format="ISO8601")
    i["nom_paisdestino"] = i["nom_paisdestino"].str.upper()
    salvar(i.drop_duplicates(["din_instante", "nom_paisdestino"], keep="last")
            .sort_values(["nom_paisdestino", "din_instante"]), "ons_intercambio_internacional_horario")

    ena = ler_ons("ena_subsistema", "ena_subsistema_di", "ENA_DIARIO_SUBSISTEMA", anos)
    ena = ena.rename(columns={"ena_data": "data", "ena_bruta_regiao_mwmed": "ena_bruta_mwmed",
                              "ena_bruta_regiao_percentualmlt": "ena_bruta_pct_mlt"})
    ena["data"] = data(ena["data"])
    salvar(ena[["data", "id_subsistema", "ena_bruta_mwmed", "ena_bruta_pct_mlt"]]
           .drop_duplicates(["data", "id_subsistema"], keep="last"), "ons_ena_subsistema_diario")

    ear = ler_ons("ear_subsistema", "ear_subsistema_di", "EAR_DIARIO_SUBSISTEMA", anos)
    ear = ear.rename(columns={"ear_data": "data", "ear_verif_subsistema_percentual": "ear_pct"})
    ear["data"] = data(ear["data"])
    salvar(ear[["data", "id_subsistema", "ear_pct"]].drop_duplicates(["data", "id_subsistema"], keep="last"),
           "ons_ear_subsistema_diario")

    carga = ler_ons("carga_subsistema", "carga_energia_di", "CARGA_ENERGIA", anos)
    carga = carga.rename(columns={"din_instante": "data", "val_cargaenergiamwmed": "carga_mwmed"})
    carga["data"] = data(carga["data"])
    carga["id_subsistema"] = carga["id_subsistema"].str.upper()
    salvar(carga[["data", "id_subsistema", "carga_mwmed"]].drop_duplicates(["data", "id_subsistema"], keep="last"),
           "ons_carga_subsistema_diario")

    cmo = ler_ons("cmo_semanal", "cmo_se", "CMO_SEMANAL", range(2005, ANO_FIM + 1))
    cmo = cmo.rename(columns={"din_instante": "data_referencia_semana", "val_cmomediasemanal": "cmo_medio_semanal"})
    cmo["data_referencia_semana"] = data(cmo["data_referencia_semana"])
    salvar(cmo[["data_referencia_semana", "id_subsistema", "cmo_medio_semanal"]]
           .drop_duplicates(["data_referencia_semana", "id_subsistema"], keep="last"), "ons_cmo_semanal")

    nomes = [f"ENERGIA_VERTIDA_TURBINAVEL_{a}" for a in range(2015, 2024)]
    nomes += [f"ENERGIA_VERTIDA_TURBINAVEL_2024_{m:02d}" for m in range(1, 13)]
    partes = []
    for nome in nomes:
        parte = INTERMEDIARIOS / "vertida_partes" / f"{nome}.parquet"
        if not parte.exists():
            bruto = baixar(f"{ONS_S3}/energia_vertida_turbinavel_ho/{nome}.csv",
                           BRUTOS / "ons" / "vertida_turbinavel" / f"{nome}.csv")
            v = pd.read_csv(bruto, sep=";", usecols=["id_subsistema", "din_instante", "val_energiavertidaturbinavel"])
            v["id_subsistema"] = v["id_subsistema"].str.strip()
            v["din_instante"] = pd.to_datetime(v["din_instante"], format="ISO8601")
            hora = v.groupby(["id_subsistema", "din_instante"], as_index=False)["val_energiavertidaturbinavel"].sum()
            hora["data"] = hora["din_instante"].dt.normalize()
            dia = (hora.groupby(["data", "id_subsistema"])["val_energiavertidaturbinavel"].mean()
                   .rename("evt_mwmed").reset_index())
            parte.parent.mkdir(exist_ok=True)
            dia.to_parquet(parte, index=False)
            bruto.unlink()
        partes.append(pd.read_parquet(parte, columns=["data", "id_subsistema", "evt_mwmed"]))
    salvar(pd.concat(partes).drop_duplicates(["data", "id_subsistema"], keep="last"), "ons_vertida_turbinavel_diario")


# Calcula a média 2000–2020 de cada dia do ano (normal climatológica), suavizada em ±7 dias.
def normal_climatologica(serie):
    d = serie.index
    doy = pd.Series(d.dayofyear - (d.is_leap_year & (d.month > 2)), index=d)
    doy[(d.month == 2) & (d.day == 29)] = 59
    base = (d.year >= NORMAL_CLIMA[0]) & (d.year <= NORMAL_CLIMA[1])
    clim = serie[base].groupby(doy[base]).mean().reindex(range(1, 366))
    suave = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    suave.index = range(1, 366)
    return doy.map(suave)


# Baixa a temperatura das 8 cidades e calcula, por grupo de cidades, médias ponderadas, graus-dia, anomalias e extremos (H1).
def coletar_clima():
    linhas = []
    for cidade, (lat, lon, pop) in CIDADES.items():
        for ini in range(ANO_INICIO, ANO_FIM + 1, 5):
            fim = min(ini + 4, ANO_FIM)
            arq = BRUTOS / "clima" / f"{cidade}_{ini}_{fim}.json"
            novo = not arq.exists()
            baixar(OPEN_METEO, arq, params={
                "latitude": lat, "longitude": lon, "start_date": f"{ini}-01-01", "end_date": f"{fim}-12-31",
                "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min", "timezone": "America/Sao_Paulo"})
            if novo:
                time.sleep(8)
            d = pd.DataFrame(json.loads(arq.read_text())["daily"])
            d = d.rename(columns={"time": "data", "temperature_2m_mean": "tmed",
                                  "temperature_2m_max": "tmax", "temperature_2m_min": "tmin"})
            linhas.append(d[["data", "tmed", "tmax", "tmin"]].assign(cidade=cidade, populacao=pop))
    c = pd.concat(linhas, ignore_index=True)
    c["data"] = pd.to_datetime(c["data"])
    c["hdd"] = (BASE_HDD - c["tmed"]).clip(lower=0)
    c["cdd"] = (c["tmed"] - BASE_CDD).clip(lower=0)
    salvar(c, "clima_diario_cidades")

    blocos, limiares = [], []
    for grupo, membros in GRUPOS_CLIMA.items():
        g = c[c["cidade"].isin(membros)]
        peso = g["populacao"] / g.groupby("data")["populacao"].transform("sum")
        agg = g[["tmed", "tmax", "tmin", "hdd", "cdd"]].mul(peso, axis=0).groupby(g["data"]).sum()
        agg["anom_tmed"] = agg["tmed"] - normal_climatologica(agg["tmed"])
        anos, meses = agg.index.year, agg.index.month
        base = (anos >= NORMAL_CLIMA[0]) & (anos <= NORMAL_CLIMA[1])
        verao, inverno = meses.isin([12, 1, 2]), meses.isin([6, 7, 8])
        p90 = agg.loc[base & verao, "tmax"].quantile(0.90)
        p10 = agg.loc[base & inverno, "tmin"].quantile(0.10)
        agg["extremo_quente"] = (verao & (agg["tmax"] > p90)).astype("int8")
        agg["extremo_frio"] = (inverno & (agg["tmin"] < p10)).astype("int8")
        agg["extremo_termico"] = agg["extremo_quente"] | agg["extremo_frio"]
        limiares.append({"grupo": grupo, "p90_tmax_verao": round(p90, 2), "p10_tmin_inverno": round(p10, 2)})
        blocos.append(agg.add_prefix(f"clima_{grupo}_"))
    salvar(pd.concat(blocos, axis=1).rename_axis("data").reset_index(), "clima_diario_grupos")
    pd.DataFrame(limiares).to_csv(INTERMEDIARIOS / "clima_limiares_extremos.csv", index=False)


# Baixa uma série do Banco Central (SGS) em blocos de 5 anos.
def serie_bcb(nome, codigo):
    partes = []
    for ini in range(ANO_INICIO, ANO_FIM + 1, 5):
        fim = min(ini + 4, ANO_FIM)
        arq = baixar(BCB_SGS.format(codigo), BRUTOS / "macro" / f"bcb_{nome}_{ini}_{fim}.json",
                     {"formato": "json", "dataInicial": f"01/01/{ini}", "dataFinal": f"31/12/{fim}"})
        partes.append(pd.DataFrame(json.loads(arq.read_text())))
    df = pd.concat(partes)
    s = pd.Series(df["valor"].astype(float).values, index=pd.to_datetime(df["data"], format="%d/%m/%Y"))
    return s[~s.index.duplicated(keep="last")].sort_index()


# Baixa uma série da API de séries de tempo do governo argentino (datos.gob.ar).
def serie_ar(nome, serie_id):
    dados, inicio, total = [], 0, 1
    while inicio < total:
        arq = baixar(DATOS_AR, BRUTOS / "macro" / f"ar_{nome}_{inicio}.json",
                     {"ids": serie_id, "start_date": f"{ANO_INICIO}-01-01", "end_date": f"{ANO_FIM}-12-31",
                      "limit": 5000, "start": inicio, "format": "json"})
        js = json.loads(arq.read_text())
        dados += js["data"]
        total, inicio = js["count"], inicio + 5000
    df = pd.DataFrame(dados, columns=["data", "valor"]).dropna()
    s = pd.Series(df["valor"].astype(float).values, index=pd.to_datetime(df["data"]))
    return s[~s.index.duplicated(keep="last")].sort_index()


# Coleta o câmbio peso/dólar e os índices de atividade (IBC-Br e EMAE) com a data em que cada mês é divulgado.
def coletar_macro():
    ars = serie_ar("ars_usd", "168.1_T_CAMBI500_D_0_0_17")
    salvar(ars.rename("ars_usd").rename_axis("data").reset_index(), "macro_cambio_diario")
    linhas = []
    for nome, s in {"ibcbr": serie_bcb("ibcbr", 24363), "emae": serie_ar("emae", "143.3_NO_PR_2004_A_21")}.items():
        d = s.rename("valor").rename_axis("mes_referencia").reset_index()
        d["data_disponivel"] = d["mes_referencia"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=DEFASAGEM_DIVULGACAO_DIAS)
        linhas.append(d.assign(indicador=nome))
    salvar(pd.concat(linhas, ignore_index=True), "macro_atividade_mensal")


# Monta o calendário diário: feriados BR/AR, pontos facultativos, fim de semana, dia útil e férias argentinas.
def coletar_calendario():
    df = pd.DataFrame({"data": pd.date_range(f"{ANO_INICIO}-01-01", f"{ANO_FIM + 1}-01-31")})
    anos = range(ANO_INICIO, ANO_FIM + 2)
    br, ar = holidays.Brazil(years=anos), holidays.Argentina(years=anos)
    opcional = holidays.Brazil(years=anos, categories=("optional",))
    d, dias = df["data"].dt, df["data"].dt.date
    df["ano"], df["mes"], df["dia_semana"] = d.year, d.month, d.dayofweek
    df["sen_anual"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    df["cos_anual"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    df["fim_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["feriado_br"] = dias.map(lambda x: x in br).astype("int8")
    df["nome_feriado_br"] = dias.map(lambda x: br.get(x, ""))
    df["ponto_facultativo_br"] = dias.map(
        lambda x: x in opcional and x not in br and "Servant" not in opcional.get(x, "")).astype("int8")
    df["feriado_ar"] = dias.map(lambda x: x in ar).astype("int8")
    df["nome_feriado_ar"] = dias.map(lambda x: ar.get(x, ""))
    folga_br = df[["feriado_br", "ponto_facultativo_br", "fim_de_semana"]].any(axis=1)
    folga_ar = df[["feriado_ar", "fim_de_semana"]].any(axis=1)
    df["dia_util_ambos"] = (~folga_br & ~folga_ar).astype("int8")
    df["tipo_dia"] = np.select([df["fim_de_semana"] == 1, folga_br | folga_ar], ["fim_de_semana", "feriado"], "dia_util")
    seg3_julho = df[(df["mes"] == 7) & (df["dia_semana"] == 0)].groupby("ano")["data"].nth(2)
    inverno_ar = pd.DatetimeIndex(np.concatenate([pd.date_range(s, periods=12) for s in seg3_julho]))
    df["ferias_escolares_ar"] = (((df["mes"] == 12) & (d.day >= 20)) | df["mes"].isin([1, 2])
                                 | df["data"].isin(inverno_ar)).astype("int8")
    salvar(df, "calendario_diario")


# Executa todas as coletas.
def coletar_tudo():
    coletar_ons()
    coletar_clima()
    coletar_macro()
    coletar_calendario()
