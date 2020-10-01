import psycopg2
import data.uncmtrd.auth as auth
import pandas as pd
from pathlib import Path

import json

with open('./partner.json', 'r') as f:
    decoded_hand = json.loads(f.read())

connection = psycopg2.connect(
    user=auth.user,
    password=auth.password,
    host=auth.host,
    port=auth.port,
    database=auth.database,
)  # please make sure to run ssh -L <local_port>:localhost:<remote_port> <user_at_remote>@<remote_address>

conn_status = connection.closed  # 0

years = [i for i in range(1988, 2020)]

for year in years:
    for code in mmbr_codes:
        try:
            Path(f"./Export_ptrIso3/{code}").mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        try:
            Path(f"./Import_ptrIso3/{code}").mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        for imex in IM_EX:
            try:
                crsr = connection.cursor()
                crsr.execute(
                    f"""
                    select * from raw___uncmtrd.annual a
                    where aggregate_level = 6 
                    and year = 2019 
and trade_flow != 'Re-Import' and trade_flow != 'Re-Export' -- and partner = 'Rep. of Korea' and commodity_code = '281111' and trade_flow = 'Import'
                """
                )
                df = pd.DataFrame(crsr.fetchall())
                # print(df)
                df.to_csv(
                    f"./{imex}_ptrIso3/{code}/y{year}_ch{'03'}.csv",
                    sep=",",
                    index=False,
                )
                print("done", code, year, imex)

            except psycopg2.OperationalError:
                print("error occurred", year)
                pass

if __name__ == "__main__":
    pass
