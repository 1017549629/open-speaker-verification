import sys

rfile=sys.argv[1]
wfile=sys.argv[2]


clean = []
with open(rfile) as f:
    for line in f:
        key = line.strip().split()[0]
        if key.endswith("-babble") or key.endswith("-noise") \
            or key.endswith("-reverb") or key.endswith("-music"):
            pass
        else:
            clean.append(line)

with open(wfile, "w") as f:
    f.writelines(clean)