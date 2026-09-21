#!/usr/bin/env bash
set -u

usage() {
    cat <<'EOF'
Usage: ./dump_pycharm.sh [--pid PID] [--samples N] [--interval SECONDS] [--top N] [--out DIR]

Collects a PyCharm JVM diagnostic bundle:
  - repeated per-thread CPU samples from /proc
  - jcmd thread dumps with Java thread names and stack snippets
  - JVM flags/properties/heap/native-memory summaries where available
  - process/thread counts, child processes, and recent PyCharm log lines

Defaults: --samples 4 --interval 3 --top 30 --out /tmp/pycharm-diagnostics-YYYYmmdd-HHMMSS
EOF
}

SAMPLES=4
INTERVAL=3
TOP=30
OUT_DIR=""
PID=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --pid)
            PID="${2:-}"
            shift 2
            ;;
        --samples)
            SAMPLES="${2:-}"
            shift 2
            ;;
        --interval)
            INTERVAL="${2:-}"
            shift 2
            ;;
        --top)
            TOP="${2:-}"
            shift 2
            ;;
        --out)
            OUT_DIR="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

require_cmd awk
require_cmd date
require_cmd ps

if [ -z "$PID" ]; then
    if command -v jps >/dev/null 2>&1; then
        PID=$(jps -lv | awk '/pycharm|com\.intellij\.idea\.Main|jetbrains-toolbox/ {print $1; exit}')
    fi
fi

if [ -z "$PID" ]; then
    PID=$(ps -eo pid=,args= | awk '
        /[p]ycharm|[c]om\.intellij\.idea\.Main|[j]etbrains\.idea/ {print $1; exit}
    ')
fi

if [ -z "$PID" ] || [ ! -d "/proc/$PID" ]; then
    echo "PyCharm JVM not found. Try: $0 --pid <PID>" >&2
    exit 1
fi

if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]] || [ "$SAMPLES" -lt 1 ]; then
    echo "--samples must be a positive integer" >&2
    exit 2
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--interval must be a positive number" >&2
    exit 2
fi

if ! [[ "$TOP" =~ ^[0-9]+$ ]] || [ "$TOP" -lt 1 ]; then
    echo "--top must be a positive integer" >&2
    exit 2
fi

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="/tmp/pycharm-diagnostics-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$OUT_DIR"

HZ=$(getconf CLK_TCK 2>/dev/null || echo 100)

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_capture() {
    local name="$1"
    shift

    {
        echo "# $*"
        echo "# $(date '+%F %T')"
        "$@"
    } > "$OUT_DIR/$name" 2>&1 || true
}

thread_ticks_snapshot() {
    local output="$1"

    for stat in /proc/"$PID"/task/*/stat; do
        [ -r "$stat" ] || continue
        awk '
            {
                tid=$1
                open=index($0, "(")
                cpos=0
                for (i=length($0); i>0; i--) {
                    if (substr($0, i, 1) == ")") {
                        cpos=i
                        break
                    }
                }
                rest=substr($0, cpos + 2)
                split(rest, f, " ")
                # Fields after comm start at stat field 3, so utime=14 is f[12], stime=15 is f[13].
                print tid, f[12] + f[13]
            }
        ' "$stat"
    done | sort -n > "$output"
}

capture_thread_dump() {
    local name="$1"

    if command -v jcmd >/dev/null 2>&1; then
        jcmd "$PID" Thread.print -l > "$OUT_DIR/$name" 2>&1 || true
    elif command -v jstack >/dev/null 2>&1; then
        jstack -l "$PID" > "$OUT_DIR/$name" 2>&1 || true
    else
        echo "Neither jcmd nor jstack is installed." > "$OUT_DIR/$name"
    fi
}

pycharm_log_candidates() {
    find "$HOME"/.cache/JetBrains "$HOME"/.config/JetBrains -path '*/log/idea.log' -type f 2>/dev/null
}

log "Collecting PyCharm diagnostics for PID $PID into $OUT_DIR"

run_capture process.txt ps -p "$PID" -o pid,ppid,pgid,sid,stat,pcpu,pmem,nlwp,etime,lstart,args
run_capture threads-initial.txt ps -L -p "$PID" -o pid,tid,psr,stat,pcpu,pmem,time,comm --sort=-pcpu
run_capture children.txt ps --ppid "$PID" -o pid,ppid,stat,pcpu,pmem,etime,args

if [ -r "/proc/$PID/status" ]; then
    cp "/proc/$PID/status" "$OUT_DIR/proc-status.txt"
fi

if [ -r "/proc/$PID/limits" ]; then
    cp "/proc/$PID/limits" "$OUT_DIR/proc-limits.txt"
fi

if [ -r "/proc/$PID/smaps_rollup" ]; then
    cp "/proc/$PID/smaps_rollup" "$OUT_DIR/proc-smaps-rollup.txt"
fi

if [ -r "/proc/$PID/cmdline" ]; then
    tr '\0' ' ' < "/proc/$PID/cmdline" > "$OUT_DIR/cmdline.txt"
    printf '\n' >> "$OUT_DIR/cmdline.txt"
fi

if command -v jcmd >/dev/null 2>&1; then
    run_capture jcmd-help.txt jcmd "$PID" help
    run_capture vm-command-line.txt jcmd "$PID" VM.command_line
    run_capture vm-flags.txt jcmd "$PID" VM.flags
    run_capture vm-system-properties.txt jcmd "$PID" VM.system_properties
    run_capture gc-heap-info.txt jcmd "$PID" GC.heap_info
    run_capture compiler-codecache.txt jcmd "$PID" Compiler.codecache
    run_capture native-memory-summary.txt jcmd "$PID" VM.native_memory summary scale=MB
fi

log "Taking $SAMPLES CPU samples, $INTERVAL second(s) apart"

for sample in $(seq 1 "$SAMPLES"); do
    sample_dir="$OUT_DIR/sample-$sample"
    mkdir -p "$sample_dir"

    thread_ticks_snapshot "$sample_dir/ticks-before.txt"
    capture_thread_dump "sample-$sample/thread-dump.txt"
    sleep "$INTERVAL"
    thread_ticks_snapshot "$sample_dir/ticks-after.txt"

    awk -v hz="$HZ" -v interval="$INTERVAL" '
        NR == FNR {
            before[$1]=$2
            next
        }
        $1 in before {
            delta=$2 - before[$1]
            if (delta < 0) {
                delta=0
            }
            cpu=(delta / hz) / interval * 100
            printf "%d %.1f %d\n", $1, cpu, delta
        }
    ' "$sample_dir/ticks-before.txt" "$sample_dir/ticks-after.txt" |
        sort -k2,2nr > "$sample_dir/thread-cpu.txt"

    ps -L -p "$PID" -o tid=,psr=,stat=,pcpu=,time=,comm= --sort=-pcpu > "$sample_dir/ps-threads.txt" 2>&1 || true

    awk '
        function nid_to_tid(nid) {
            if (nid ~ /^0x/) {
                return strtonum(nid)
            }
            return nid + 0
        }
        function nid_display(nid) {
            if (nid ~ /^0x/) {
                return nid
            }
            return sprintf("0x%x", nid + 0)
        }
        /^"/ {
            name=$0
            sub(/^"/, "", name)
            sub(/".*/, "", name)
            nid=""
            if (match($0, /nid=(0x[0-9a-fA-F]+|[0-9]+)/)) {
                nid=substr($0, RSTART + 4, RLENGTH - 4)
                printf "%d\t%s\t%s\n", nid_to_tid(nid), nid_display(nid), name
            }
        }
    ' "$sample_dir/thread-dump.txt" > "$sample_dir/java-thread-map.tsv" 2>/dev/null || true

    awk -v map="$sample_dir/java-thread-map.tsv" -v dump="$sample_dir/thread-dump.txt" -v top="$TOP" '
        function dump_tid(line,    nid) {
            if (match(line, /nid=(0x[0-9a-fA-F]+|[0-9]+)/)) {
                nid=substr(line, RSTART + 4, RLENGTH - 4)
                if (nid ~ /^0x/) {
                    return strtonum(nid)
                }
                return nid + 0
            }
            return ""
        }
        BEGIN {
            while ((getline line < map) > 0) {
                split(line, fields, "\t")
                tid_to_nid[fields[1]]=fields[2]
                tid_to_name[fields[1]]=fields[3]
            }
            close(map)

            current_tid=""
            current_block=""
            while ((getline dline < dump) > 0) {
                if (dline ~ /^"/) {
                    if (current_tid != "") {
                        block[current_tid]=current_block
                    }
                    current_tid=dump_tid(dline)
                    current_block=dline "\n"
                } else if (current_tid != "") {
                    current_block=current_block dline "\n"
                }
            }
            if (current_tid != "") {
                block[current_tid]=current_block
            }
            close(dump)
        }
        FNR <= top {
            tid=$1
            cpu=$2
            delta=$3
            name=(tid in tid_to_name) ? tid_to_name[tid] : "(not found in JVM dump)"
            nid=(tid in tid_to_nid) ? tid_to_nid[tid] : "-"
            printf "%-10s %-10s %-8s %s\n", "TID", "NID", "CPU%", "JAVA THREAD"
            printf "%-10d %-10s %-8.1f %s\n\n", tid, nid, cpu, name
            if (tid in block) {
                printf "%s", block[tid]
            } else {
                print "(No Java stack block found. This may be the process leader or a JVM/native helper thread.)"
            }
            print ""
            print "--------------------------------------------------------------------------------"
            print ""
        }
    ' "$sample_dir/thread-cpu.txt" > "$sample_dir/top-thread-stacks.txt"

    {
        echo "JVM thread states:"
        awk '
            /java.lang.Thread.State:/ {
                state=$2
                sub(/\(.*/, "", state)
                count[state]++
            }
            END {
                for (state in count) {
                    printf "%-18s %d\n", state, count[state]
                }
            }
        ' "$sample_dir/thread-dump.txt" | sort
        echo
        echo "OS thread states:"
        awk '{count[$3]++} END {for (state in count) printf "%-18s %d\n", state, count[state]}' "$sample_dir/ps-threads.txt" | sort
        echo
        echo "Most common Java thread name prefixes:"
        awk -F'\t' '
            {
                name=$3
                sub(/[ -]?[0-9]+$/, "", name)
                count[name]++
            }
            END {
                for (name in count) {
                    printf "%5d %s\n", count[name], name
                }
            }
        ' "$sample_dir/java-thread-map.tsv" | sort -nr | head -25
    } > "$sample_dir/thread-states.txt"

    log "Sample $sample complete"
done

{
    echo "PyCharm PID: $PID"
    echo "Output directory: $OUT_DIR"
    echo "Samples: $SAMPLES"
    echo "Interval seconds: $INTERVAL"
    echo
    echo "Process summary:"
    cat "$OUT_DIR/process.txt"
    echo
    echo "Thread state/count summary:"
    if [ -r "$OUT_DIR/proc-status.txt" ]; then
        awk '/^Threads:|^voluntary_ctxt_switches:|^nonvoluntary_ctxt_switches:/ {print}' "$OUT_DIR/proc-status.txt"
    fi
    echo
    echo "Top threads by sampled CPU:"
    for cpu_file in "$OUT_DIR"/sample-*/thread-cpu.txt; do
        sample_name=$(basename "$(dirname "$cpu_file")")
        echo
        echo "[$sample_name]"
        awk -v map="$(dirname "$cpu_file")/java-thread-map.tsv" -v top="$TOP" '
            BEGIN {
                while ((getline line < map) > 0) {
                split(line, fields, "\t")
                    tid_to_nid[fields[1]]=fields[2]
                    tid_to_name[fields[1]]=fields[3]
                }
                close(map)
                printf "%-10s %-10s %-8s %s\n", "TID", "NID", "CPU%", "JAVA THREAD"
                printf "%-10s %-10s %-8s %s\n", "----------", "----------", "--------", "--------------------------------"
            }
            FNR <= top {
                tid=$1
                nid=(tid in tid_to_nid) ? tid_to_nid[tid] : "-"
                name=(tid in tid_to_name) ? tid_to_name[tid] : "(not found in JVM dump)"
                printf "%-10d %-10s %-8.1f %s\n", tid, nid, $2, name
            }
        ' "$cpu_file"
    done
    echo
    echo "Look next at sample-*/top-thread-stacks.txt for the Java stacks behind hot threads."
    echo "Use sample-*/thread-states.txt to see whether thread count is active work or mostly parked helpers."
} > "$OUT_DIR/summary.txt"

latest_log=$(pycharm_log_candidates | xargs -r ls -t 2>/dev/null | head -1)
if [ -n "${latest_log:-}" ]; then
    {
        echo "# $latest_log"
        echo "# Last 400 lines containing common high-CPU/indexing/error clues"
        grep -Ei 'index|scanning|vfs|freeze|slow|thread|cpu|memory|oom|exception|error' "$latest_log" | tail -400
    } > "$OUT_DIR/pycharm-log-clues.txt" 2>&1 || true
fi

cat "$OUT_DIR/summary.txt"

log "Done. Diagnostic bundle: $OUT_DIR"
