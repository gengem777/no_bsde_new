import options as opt

PATH_DEPENDENT_LIST = [
    opt.GeometricAsian, 
    opt.LookbackOption,
]

SDE_LIST = ["GBM", "TGBM", "SV", "HW", "SVJ", "GBMJ"]

JUMP_LIST = ["GBMJ", "SVJ"]

OPTION_LIST = [
    "European",
    "Lookback",
    "Asian",
    "Basket",
    "BasketnoPI",
    "Swap",
    "Bond",
    "Swaption",
    "SwaptionFirst",
    "SwaptionLast",
    "TimeEuropean",
    "BermudanPut",
]
DIM_LIST = [1, 3, 5, 10, 20]