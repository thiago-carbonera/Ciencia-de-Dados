from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BRUTOS = RAIZ / "data" / "raw"
INTERMEDIARIOS = RAIZ / "data" / "interim"
PROCESSADOS = RAIZ / "data" / "processed"
for pasta in (BRUTOS, INTERMEDIARIOS, PROCESSADOS):
    pasta.mkdir(parents=True, exist_ok=True)

ANO_INICIO, ANO_FIM = 2000, 2024
NORMAL_CLIMA = (2000, 2020)
INICIO_MODELAGEM, FIM_TREINO, FIM_TESTE = "2015-01-01", "2023-12-31", "2024-12-31"
INICIO_RECORTE_ORIGINAL = "2021-01-01"

ONS_S3 = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
BCB_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
DATOS_AR = "https://apis.datos.gob.ar/series/api/series/"

CIDADES = {
    "sao_paulo":      (-23.55, -46.63, 11_451_245),
    "rio_de_janeiro": (-22.91, -43.17, 6_211_423),
    "belo_horizonte": (-19.92, -43.94, 2_315_560),
    "curitiba":       (-25.43, -49.27, 1_773_733),
    "porto_alegre":   (-30.03, -51.23, 1_332_570),
    "buenos_aires":   (-34.61, -58.38, 13_985_794),
    "cordoba":        (-31.42, -64.18, 1_565_112),
    "rosario":        (-32.95, -60.65, 1_342_619),
}
GRUPOS_CLIMA = {
    "br": ["sao_paulo", "rio_de_janeiro", "belo_horizonte", "curitiba", "porto_alegre"],
    "br_se": ["sao_paulo", "rio_de_janeiro", "belo_horizonte"],
    "br_s": ["curitiba", "porto_alegre"],
    "ar": ["buenos_aires", "cordoba", "rosario"],
}
BASE_HDD, BASE_CDD = 18.0, 22.0
LIMIAR_FLUXO_MW = 1.0
DEFASAGEM_DIVULGACAO_DIAS = 60

VARIAVEIS = {
    "historico": ["hist_media_ult6h_l1", "hist_lag1", "hist_horas_import_l1", "hist_lag2",
                  "hist_horas_export_l1", "hist_abs_mm7", "hist_mm7", "hist_lag7"],
    "calendario": ["cal_mes", "cal_cos_anual", "cal_sen_anual", "cal_dia_semana", "cal_dia_util_ambos",
                   "cal_ferias_escolares_ar", "cal_fim_de_semana", "cal_feriado_br", "cal_feriado_ar"],
    "clima": ["clima_ar_hdd_7d", "clima_ar_hdd_l1", "clima_br_cdd_7d", "clima_ar_tmax_l1", "clima_ar_tmin_l1",
              "clima_br_se_anom_tmed_mm3", "clima_br_s_hdd_l1", "clima_br_se_hdd_l1",
              "clima_br_extremo_termico_l1", "clima_ar_extremo_termico_l1"],
    "hidro_preco": ["hidro_ear_pct_se_l1", "hidro_ear_pct_s_l1", "hidro_ena_pct_mlt_sin_mm7",
                    "hidro_ena_mwmed_sin_l1", "hidro_evt_mwmed_se_mm7", "hidro_ena_pct_mlt_se_l1",
                    "hidro_ear_pct_se_var7", "hidro_evt_mwmed_sin_mm7", "preco_cmo_s_ult_semana"],
    "carga": ["carga_mwmed_sin_mm7", "carga_mwmed_s_l1"],
    "macro": ["macro_ars_usd_var30", "macro_ibcbr_var12m_divulgado", "macro_emae_var12m_divulgado"],
}
