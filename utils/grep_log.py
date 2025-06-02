import re
import sys

def grep_episode_returns(filename):
    pattern = re.compile(r"Episode (\d+): return ([\d\.\-eE]+)")
    with open(filename, 'r', encoding='utf-8') as f:
        out_f = open(filename + "_grep", "a", encoding="utf-8")
        for line in f:
            match = pattern.search(line)
            if match:
                    out_f.write(f"{match.group(1)}, {match.group(2)}\n")

def grep_episode_rewards(filename):
    pattern = re.compile(r"reward:\s*([\d\.\-eE]+)")
    with open(filename, 'r', encoding='utf-8') as f:
        out_f = open(filename + "_episodes", "a", encoding="utf-8")
        for line in f:
            match = pattern.search(line)
            if match:
                out_f.write(f"{match.group(1)}\n")

    # Remove repetitive, adjacent lines
    out_f.flush()
    with open(filename + "_episodes", "r", encoding="utf-8") as temp_f:
        lines = temp_f.readlines()
    with open(filename + "_episodes", "w", encoding="utf-8") as temp_f:
        prev = None
        for line in lines:
            if line != prev:
                temp_f.write(line)
            prev = line
    out_f.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python grep_log.py <logfile>")
    else:
        grep_episode_returns(sys.argv[1])
        grep_episode_rewards(sys.argv[1])