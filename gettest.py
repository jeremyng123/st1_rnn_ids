import pandas as pd

file = pd.read_csv("KDDTest+.csv")
with open("test_9feats.csv", "w") as f:
    df = pd.DataFrame(file,
                      columns=[
                          "dst_host_srv_serror_rate", "dst_host_serror_rate",
                          "serror_rate", "srv_serror_rate", "count", "flag",
                          "same_srv_rate", "dst_host_srv_count",
                          "dst_host_diff_srv_rate", "Malicious"
                      ])
    df.to_csv(f, index=False, header=True, line_terminator='\n')
    print(df)