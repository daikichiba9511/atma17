import pathlib

import polars as pl

pathlib.Path("./output/ensemble").mkdir(parents=True, exist_ok=True)

sub1 = pl.read_csv("./output/exp002/submission.csv")
sub2 = pl.read_csv("./output/exp006/submission.csv")
sub3 = pl.read_csv("./output/exp004/submission.csv")

# LB: 0.9715
# sub = 0.5 * sub1["target"] + 0.5 * sub2["target"]
# sub = pl.DataFrame({"target": sub})
# print(sub)
# sub.write_csv("./output/ensemble/submission_0206_w05.csv")


# LB: 0.9712
# sub = (0.4 * sub1["target"]) + (0.2 * sub2["target"]) + (0.4 * sub3["target"])
# sub = pl.DataFrame({"target": sub})
# print(sub)
# sub.write_csv("./output/ensemble/submission_020604_w0424.csv")

sub = 0.5 * sub1["target"] + 0.5 * sub3["target"]
sub = pl.DataFrame({"target": sub})
print(sub)
sub.write_csv("./output/ensemble/submission_0204_w05.csv")
