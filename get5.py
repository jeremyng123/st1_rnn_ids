import pandas as pd

new_feats = [
    "dst_host_srv_count", "dst_host_diff_srv_rate", "flag", "srv_serror_rate",
    "serror_rate", 'Malicious'
]

name = "test"
file = pd.read_csv(f"{name}_9feats.csv")
with open(f"{name}_5feats.csv", "w") as f:
    df = pd.DataFrame(file, columns=new_feats)
    df.to_csv(f, index=False, header=True, line_terminator='\n')
    print(df)

name = "train"
file = pd.read_csv(f"{name}_9feats.csv")
with open(f"{name}_5feats.csv", "w") as f:
    df = pd.DataFrame(file, columns=new_feats)
    df.to_csv(f, index=False, header=True, line_terminator='\n')
    print(df)