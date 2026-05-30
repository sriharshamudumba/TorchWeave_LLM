import json
import sys

j = json.load(sys.stdin)


def table(rows):
    print("| batch | prefill | decode | TTFT (ms) | TPS | VRAM (GB) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['batch']} | {r['prefill']} | {r['decode']} | {r['ttft_ms']} | {r['tps']} | {r.get('vram_gb','-')} |"
        )


print(f"### Benchmark — {j['model']} on {j['device']}\n")
print(f"- Artifact: `{j['artifact']}`")
print(f"- Optimize duration: **{j['optimize_sec']}s**\n")

print("**Before**")
table(j["before"]["by_batch"])
print("\n**After**")
table(j["after"]["by_batch"])
