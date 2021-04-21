import pandas as pd

new_feats = [
    "dst_host_srv_count", "dst_host_diff_srv_rate", "flag", "srv_serror_rate",
    "serror_rate"
]

names = ['train']
for name in names:
    file = pd.read_csv(f"{name}_5feats.csv")
    with open("2.csv", "w") as f:
        df = pd.DataFrame(file,
                          columns=[
                              "dst_host_srv_count", "dst_host_diff_srv_rate",
                              'Malicious'
                          ])
        df.to_csv(f, index=False, header=True, line_terminator='\n')
        print(df)
