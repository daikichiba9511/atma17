import pathlib

import polars as pl

pathlib.Path("./output/ensemble").mkdir(parents=True, exist_ok=True)

sub1 = pl.read_csv("./output/exp002/submission.csv")
sub2 = pl.read_csv("./output/exp006/submission.csv")
sub3 = pl.read_csv("./output/exp004/submission.csv")
sub4 = pl.read_csv("./output/exp007/submission.csv")

# LB: 0.9715, Private: 0.9768
# sub = 0.5 * sub1["target"] + 0.5 * sub2["target"]
# sub = pl.DataFrame({"target": sub})
# print(sub)
# sub.write_csv("./output/ensemble/submission_0206_w05.csv")


# LB: 0.9712, Private: 0.9768
# sub = (0.4 * sub1["target"]) + (0.2 * sub2["target"]) + (0.4 * sub3["target"])
# sub = pl.DataFrame({"target": sub})
# print(sub)
# sub.write_csv("./output/ensemble/submission_020604_w0424.csv")

# LB: 0.9710, Private: 0.9763
# sub = 0.5 * sub1["target"] + 0.5 * sub3["target"]
# sub = pl.DataFrame({"target": sub})
# print(sub)
# sub.write_csv("./output/ensemble/submission_0204_w05.csv")

# ------- Late Sub -----

# LB: 0.9716, Private: 0.9764
# sub = 0.5 * sub1["target"] + 0.5 * sub4["target"]
# sub = pl.DataFrame({"target": sub})
# print(sub)
# sub.write_csv("./output/ensemble/submission_0207_w05.csv")

# LB: 0.9717, Private: 0.9769
# sub = pl.DataFrame({
#     "target": (0.25 * sub1["target"]) + (0.25 * sub2["target"]) + (0.25 * sub3["target"]) + (0.25 * sub4["target"])
# })
# print(sub)
# sub.write_csv("./output/ensemble/submission_02060407_w025.csv")

sub = pl.DataFrame({"target": (0.4 * sub1["target"]) + (0.2 * sub2["target"]) + (0.4 * sub4["target"])})
print(sub)
sub.write_csv("./output/ensemble/submission_020607_w040204.csv")
