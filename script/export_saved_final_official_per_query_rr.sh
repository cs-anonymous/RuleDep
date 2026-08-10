#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SUFFIX="${RUN_SUFFIX:-main_table_per_query_rr_20260809}"
EXPORT_DIR="${EXPORT_DIR:-${ROOT_DIR}/reports/official_query_subset/true_official_per_query_rr/${RUN_SUFFIX}}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
fi

export_one() {
    local dataset="$1"
    local gpu="$2"
    local experiment="$3"
    shift 3
    local saved_dir="${ROOT_DIR}/data/${dataset}/aggregation/reproduction"
    local relation_args=(--relation -1)
    if [ -n "${RELATION_IDS:-}" ]; then
        relation_args+=(--relation_ids "${RELATION_IDS}")
    fi
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${saved_dir}"
        export PYTHONUNBUFFERED=1
        python src/ruledep/aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            "${relation_args[@]}" \
            --export_saved_final_per_query_rr \
            --saved_mrr_dir "${saved_dir}" \
            --export_per_query_rr_dir "${EXPORT_DIR}" \
            --export_experiment_name "${experiment}" \
            "$@"
    )
}

case "${1:-all}" in
    FB15k-237)
        export_one FB15k-237 "${GPU:-0}" tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 \
            --synergy --redundancy --type_grouping r2d3 --pos auto_ratio \
            --rule_init_mode conf --dependency_static_norm none --dep_l1_lambda 1e-5
        ;;
    WN18RR)
        export_one WN18RR "${GPU:-0}" structural_rd \
            --synergy --redundancy --type_grouping rd
        ;;
    KG20C)
        export_one KG20C "${GPU:-0}" tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 \
            --synergy --redundancy --type_grouping r2d3 --pos auto_ratio \
            --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
        ;;
    YAGO3-10)
        export_one YAGO3-10 "${GPU:-0}" tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5 \
            --synergy --redundancy --type_grouping r3d6 --pos auto_sqrt \
            --rule_init_mode surprisal --dependency_static_norm none --dep_l1_lambda 1e-5
        ;;
    codex-m)
        export_one codex-m "${GPU:-0}" tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 \
            --synergy --redundancy --type_grouping rd --pos auto_sqrt \
            --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
        ;;
    codex-l)
        export_one codex-l "${GPU:-0}" tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 \
            --synergy --redundancy --type_grouping rd --pos auto_sqrt \
            --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
        ;;
    all)
        echo "Run one dataset per process, e.g. GPU=0 $0 FB15k-237" >&2
        exit 2
        ;;
    *)
        echo "Unknown dataset: $1" >&2
        exit 2
        ;;
esac
