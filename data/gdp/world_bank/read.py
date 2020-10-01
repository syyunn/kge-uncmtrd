import pandas as pd

df = pd.read_csv('./API_NY.GDP.MKTP.CD_DS2_en_csv_v2_1345540.csv')

mmbrs_iso3 = ['IND', 'CHN', 'USA', 'AUS']
year_codes = [str(year) for year in range(1988, 2020)]

for iso3 in mmbrs_iso3:
    idx = df.index[df['Country Code'] == iso3].tolist()
    gdps = df.loc[idx[0]][year_codes].to_list()
    pass
if __name__ == "__main__":
    pass
