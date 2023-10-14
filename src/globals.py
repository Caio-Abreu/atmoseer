INMET_API_BASE_URL = "https://apitempo.inmet.gov.br"

# Weather stations datasource directories
WS_INMET_DATA_DIR = "./data/ws/inmet/"
WS_ALERTARIO_DATA_DIR = "./data/ws/alertario/ws/"
GS_ALERTARIO_DATA_DIR = "./data/ws/alertario/rain_gauge_era5_fused/"

WS_GOES_DATA_DIR = "atmoseer/data/ws/goes16"

# Atmospheric sounding datasource directory
NWP_DATA_DIR = "./data/NWP/"

# Atmospheric sounding datasource directory
AS_DATA_DIR = "./data/as/"

# Directory to store the train/val/test datasets for each weather station of interest
DATASETS_DIR = './data/datasets/'

# Directory to store the generated models and their corresponding reports
MODELS_DIR = './models/'

# see https://portal.inmet.gov.br/paginas/catalogoaut
INMET_WEATHER_STATION_IDS = ('A636', # Jacarepagua
                     'A621', # Vila militar
                     'A602', # Marambaia
                     'A652', # Copacabana
                     'A627')

ALERTARIO_GAUGE_STATION_IDS = (
                         'anchieta', 
                         'av_brasil_mendanha', 
                         'bangu', 
                         'barrinha', 
                         'campo_grande', 
                         'cidade_de_deus', 
                         'copacabana', 
                         'grajau_jacarepagua', 
                         'grajau', 
                         'grande_meier', 
                         'grota_funda', 
                         'ilha_do_governador', 
                         'laranjeiras', 
                         'madureira', 
                         'penha', 
                         'piedade', 
                         'recreio', 
                         'rocinha',
                         'santa_teresa',
                         'saude', 
                         'sepetiba', 
                         'tanque', 
                         'tijuca_muda', 
                         'tijuca', 
                         'urca',
                         'alto_da_boa_vista', #**
                         'iraja', #**
                         'jardim_botanico', #**
                         'riocentro', #**
                         'santa_cruz', #**
                         'vidigal' #**
                         )

ALERTARIO_WEATHER_STATION_IDS = (
                         'guaratiba', #**
                         'sao_cristovao' #**
                         )

# hyper_params_dict_bc = {
#     "N_EPOCHS" : 3500,
#     "PATIENCE" : 1000,
#     "BATCH_SIZE" : 1024,
#     "WEIGHT_DECAY" : 0,
#     "LEARNING_RATE" : 0.0003,
#     "DROPOUT_RATE" : 0.5,
#     "SLIDING_WINDOW_SIZE" : 6
# }

# hyper_params_dict_oc = {
#     "N_EPOCHS" : 6000,
#     "PATIENCE" : 1000,
#     "BATCH_SIZE" : 1024,
#     "WEIGHT_DECAY" : 0,
#     "LEARNING_RATE" : 3e-6,
#     "DROPOUT_RATE" : 0.5,
#     "SLIDING_WINDOW_SIZE" : 6
# }


# Observed variables for INMET weather stations:
# ,DC_NOME,
# PRE_INS,
# TEM_SEN,
# VL_LATITUDE,
# PRE_MAX,UF,
# RAD_GLO,
# PTO_INS,
# TEM_MIN,
# VL_LONGITUDE,
# UMD_MIN,
# PTO_MAX,
# VEN_DIR,
# DT_MEDICAO,
# CHUVA,
# PRE_MIN,
# UMD_MAX,
# VEN_VEL,
# PTO_MIN,
# TEM_MAX,
# TEN_BAT,
# VEN_RAJ,
# TEM_CPU,
# TEM_INS,
# UMD_INS,
# CD_ESTACAO,
# HR_MEDICAO