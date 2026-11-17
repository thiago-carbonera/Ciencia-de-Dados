import json

import numpy as np
import pandas as pd

from .config import (ANO_FIM, ANO_INICIO, FIM_TESTE, FIM_TREINO, INICIO_MODELAGEM, INICIO_RECORTE_ORIGINAL,
                     INTERMEDIARIOS, LIMIAR_FLUXO_MW, PROCESSADOS, VARIAVEIS)

IDX = pd.date_range(f"{ANO_INICIO}-01-01", f"{ANO_FIM}-12-31", name="data")


# Lê uma tabela de data/interim/.
def ler(nome):
    return pd.read_parquet(INTERMEDIARIOS / f"{nome}.parquet")


# Classifica o fluxo em exportação, importação ou sem fluxo.
def sentido(v):
    s = np.select([v > LIMIAR_FLUXO_MW, v < -LIMIAR_FLUXO_MW], ["exportacao", "importacao"], "sem_fluxo")
    return pd.Series(s, index=v.index).where(v.notna())


# Soma uma coluna por dia (de um subsistema ou do SIN inteiro) no calendário completo.
def diario(df, coluna, subsistema=None):
    if subsistema:
        df = df[df["id_subsistema"] == subsistema]
    return df.groupby("data")[coluna].sum(min_count=1).reindex(IDX)


# Monta a base horária do intercâmbio com a Argentina, com marcadores de calendário.
def base_horaria(cal):
    i = ler("ons_intercambio_internacional_horario")
    h = i[i["nom_paisdestino"] == "ARGENTINA"][["din_instante", "val_intercambiomwmed"]]
    h = h.rename(columns={"val_intercambiomwmed": "intercambio_mwmed"}).sort_values("din_instante")
    h["data"] = h["din_instante"].dt.normalize()
    h["ano"], h["mes"], h["hora"] = h["din_instante"].dt.year, h["din_instante"].dt.month, h["din_instante"].dt.hour
    h["dia_semana"] = h["din_instante"].dt.dayofweek
    h["sentido"] = sentido(h["intercambio_mwmed"])
    colunas = ["data", "feriado_br", "feriado_ar", "ponto_facultativo_br", "fim_de_semana", "dia_util_ambos", "tipo_dia"]
    return h.merge(cal[colunas], on="data", how="left", validate="many_to_one").reset_index(drop=True)


# Calcula, para cada dia, a média do intercâmbio e estatísticas do perfil horário.
def alvo_diario(h):
    v, dia = h["intercambio_mwmed"], h["data"]
    noite = h["hora"] >= 18
    a = pd.DataFrame({
        "intercambio_mwmed": v.groupby(dia).mean(),
        "intensidade_horaria_mwmed": v.abs().groupby(dia).mean(),
        "horas_exportacao": (v > LIMIAR_FLUXO_MW).groupby(dia).sum(),
        "horas_importacao": (v < -LIMIAR_FLUXO_MW).groupby(dia).sum(),
        "media_ultimas_6h": v[noite].groupby(dia[noite]).mean(),
    }).reindex(IDX)
    a["intercambio_abs_mwmed"] = a["intercambio_mwmed"].abs()
    a["sentido_dia"] = sentido(a["intercambio_mwmed"])
    return a


# Junta ENA, EAR, vertimento e carga do ONS em uma tabela diária.
def sistema_diario():
    ena, ear, evt, carga = (ler(f"ons_{n}") for n in
                            ("ena_subsistema_diario", "ear_subsistema_diario", "vertida_turbinavel_diario",
                             "carga_subsistema_diario"))
    mlt = ena.assign(mlt=ena["ena_bruta_mwmed"] / (ena["ena_bruta_pct_mlt"] / 100))
    t = pd.DataFrame(index=IDX)
    t["hidro_ena_mwmed_sin"] = diario(ena, "ena_bruta_mwmed")
    t["hidro_ena_pct_mlt_sin"] = 100 * t["hidro_ena_mwmed_sin"] / diario(mlt, "mlt")
    t["hidro_ena_pct_mlt_se"] = diario(ena, "ena_bruta_pct_mlt", "SE")
    t["hidro_ear_pct_s"] = diario(ear, "ear_pct", "S")
    t["hidro_ear_pct_se"] = diario(ear, "ear_pct", "SE")
    t["hidro_evt_mwmed_se"] = diario(evt, "evt_mwmed", "SE")
    t["hidro_evt_mwmed_sin"] = diario(evt, "evt_mwmed")
    t["carga_mwmed_s"] = diario(carga, "carga_mwmed", "S")
    t["carga_mwmed_sin"] = carga.groupby("data")["carga_mwmed"].sum(min_count=4).reindex(IDX)
    return t


# Leva o CMO semanal do Sul para cada dia (semana corrente ou última semana já encerrada).
def cmo_sul(semana_encerrada):
    cmo = ler("ons_cmo_semanal")
    s = cmo[cmo["id_subsistema"] == "S"].set_index("data_referencia_semana")["cmo_medio_semanal"]
    if semana_encerrada:
        ontem = IDX - pd.Timedelta(days=1)
        sexta = ontem - pd.to_timedelta((ontem.dayofweek - 4) % 7, unit="D")
    else:
        sexta = IDX + pd.to_timedelta((4 - IDX.dayofweek) % 7, unit="D")
    return pd.Series(s.reindex(sexta).values, index=IDX)


# Leva o câmbio e os índices de atividade para cada dia (pelo mês de referência ou só após a divulgação).
def macro_diario(divulgado):
    ars = ler("macro_cambio_diario").set_index("data")["ars_usd"]
    m = pd.DataFrame({"macro_ars_usd": ars.reindex(ars.index.union(IDX)).ffill().reindex(IDX)})
    for nome, g in ler("macro_atividade_mensal").groupby("indicador"):
        g = g.sort_values("mes_referencia").assign(var12m=lambda x: x["valor"].pct_change(12, fill_method=None) * 100)
        if divulgado:
            s = g.set_index("data_disponivel")["var12m"]
            m[f"macro_{nome}_var12m"] = s.reindex(s.index.union(IDX)).ffill().reindex(IDX)
        else:
            m[f"macro_{nome}_var12m"] = g.set_index("mes_referencia")["var12m"].reindex(IDX.to_period("M").to_timestamp()).values
    return m


# Monta a base de previsão t+1: alvo do dia d e as 41 variáveis com informação até d−1.
def base_modelagem(alvo, cal, clima, sistema):
    y = alvo["intercambio_mwmed"]
    yf = y.ffill(limit=3)
    y1 = yf.shift(1)
    ontem = lambda s: s.shift(1)
    media7 = lambda s: s.shift(1).rolling(7, min_periods=5).mean()
    macro = macro_diario(divulgado=True)
    X = pd.DataFrame({
        "hist_media_ult6h_l1": ontem(alvo["media_ultimas_6h"]),
        "hist_lag1": y1,
        "hist_lag2": yf.shift(2),
        "hist_lag7": yf.shift(7),
        "hist_horas_import_l1": ontem(alvo["horas_importacao"]),
        "hist_horas_export_l1": ontem(alvo["horas_exportacao"]),
        "hist_abs_mm7": y1.abs().rolling(7, min_periods=5).mean(),
        "hist_mm7": y1.rolling(7, min_periods=4).mean(),
        "cal_mes": cal["mes"], "cal_sen_anual": cal["sen_anual"], "cal_cos_anual": cal["cos_anual"],
        "cal_dia_semana": cal["dia_semana"], "cal_fim_de_semana": cal["fim_de_semana"],
        "cal_feriado_br": cal["feriado_br"], "cal_feriado_ar": cal["feriado_ar"],
        "cal_dia_util_ambos": cal["dia_util_ambos"], "cal_ferias_escolares_ar": cal["ferias_escolares_ar"],
        "clima_ar_hdd_l1": ontem(clima["clima_ar_hdd"]),
        "clima_ar_hdd_7d": ontem(clima["clima_ar_hdd"]).rolling(7).sum(),
        "clima_br_cdd_7d": ontem(clima["clima_br_cdd"]).rolling(7).sum(),
        "clima_ar_tmax_l1": ontem(clima["clima_ar_tmax"]),
        "clima_ar_tmin_l1": ontem(clima["clima_ar_tmin"]),
        "clima_br_se_anom_tmed_mm3": ontem(clima["clima_br_se_anom_tmed"]).rolling(3).mean(),
        "clima_br_s_hdd_l1": ontem(clima["clima_br_s_hdd"]),
        "clima_br_se_hdd_l1": ontem(clima["clima_br_se_hdd"]),
        "clima_br_extremo_termico_l1": ontem(clima["clima_br_extremo_termico"]),
        "clima_ar_extremo_termico_l1": ontem(clima["clima_ar_extremo_termico"]),
        "hidro_ear_pct_se_l1": ontem(sistema["hidro_ear_pct_se"]),
        "hidro_ear_pct_s_l1": ontem(sistema["hidro_ear_pct_s"]),
        "hidro_ear_pct_se_var7": ontem(sistema["hidro_ear_pct_se"]) - sistema["hidro_ear_pct_se"].shift(8),
        "hidro_ena_mwmed_sin_l1": ontem(sistema["hidro_ena_mwmed_sin"]),
        "hidro_ena_pct_mlt_se_l1": ontem(sistema["hidro_ena_pct_mlt_se"]),
        "hidro_ena_pct_mlt_sin_mm7": media7(sistema["hidro_ena_pct_mlt_sin"]),
        "hidro_evt_mwmed_se_mm7": media7(sistema["hidro_evt_mwmed_se"]),
        "hidro_evt_mwmed_sin_mm7": media7(sistema["hidro_evt_mwmed_sin"]),
        "preco_cmo_s_ult_semana": cmo_sul(semana_encerrada=True),
        "carga_mwmed_s_l1": ontem(sistema["carga_mwmed_s"]),
        "carga_mwmed_sin_mm7": media7(sistema["carga_mwmed_sin"]),
        "macro_ars_usd_var30": 100 * np.log(macro["macro_ars_usd"].shift(1) / macro["macro_ars_usd"].shift(31)),
        "macro_ibcbr_var12m_divulgado": ontem(macro["macro_ibcbr_var12m"]),
        "macro_emae_var12m_divulgado": ontem(macro["macro_emae_var12m"]),
    })
    X = X[[c for colunas in VARIAVEIS.values() for c in colunas]]
    df = pd.concat([pd.DataFrame({"alvo_intercambio_mwmed": y, "alvo_abs_mwmed": y.abs(), "alvo_sentido": sentido(y)}),
                    X], axis=1).loc[INICIO_MODELAGEM:FIM_TESTE]
    df = df[df["alvo_intercambio_mwmed"].notna()].copy()
    df["particao"] = np.where(df.index <= FIM_TREINO, "treino", "teste")
    df["recorte_original_2021"] = (df.index >= INICIO_RECORTE_ORIGINAL).astype("int8")
    return df.reset_index()


# Gera e salva as três bases finais em data/processed/.
def construir_tudo():
    cal = ler("calendario_diario")
    h = base_horaria(cal)
    alvo = alvo_diario(h)
    cal = cal.set_index("data").reindex(IDX)
    clima = ler("clima_diario_grupos").set_index("data").reindex(IDX)
    sistema = sistema_diario()

    diaria = pd.concat([alvo, cal, clima, sistema, macro_diario(divulgado=False)], axis=1)
    diaria["preco_cmo_s"] = cmo_sul(semana_encerrada=False)
    diaria["periodo"] = np.select([IDX < INICIO_MODELAGEM, IDX <= FIM_TREINO], ["exploracao", "treino"], "teste")
    diaria = diaria.reset_index()
    modelo = base_modelagem(alvo, cal, clima, sistema)

    h.to_parquet(PROCESSADOS / "intercambio_argentina_horario.parquet", index=False)
    diaria.to_parquet(PROCESSADOS / "base_diaria_integrada.parquet", index=False)
    modelo.to_parquet(PROCESSADOS / "base_modelagem_t1.parquet", index=False)
    (PROCESSADOS / "grupos_variaveis.json").write_text(json.dumps(VARIAVEIS, indent=2), encoding="utf-8")
    for nome, df in (("horária", h), ("diária", diaria), ("modelagem t+1", modelo)):
        print(f"base {nome}: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    return {"horaria": h, "diaria": diaria, "modelagem": modelo}
