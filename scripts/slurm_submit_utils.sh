#!/bin/bash

slurm_submit_prefix() {
  local gpu_profile="${1:-large}"
  local gpu_partition="${GPU_PARTITION:-}"
  local qos="${GPU_QOS:-gpu_access}"
  local gres="${GPU_GRES:-gpu:1}"

  if [ -n "$gpu_partition" ]; then
    printf '%s\n' "sbatch --parsable -p ${gpu_partition} --qos=${qos} --gres=${gres}"
    return
  fi

  case "$gpu_profile" in
    small)
      printf '%s\n' "sbatch --parsable -p l40-gpu --qos=${qos} --gres=${gres}"
      ;;
    medium)
      printf '%s\n' "sbatch --parsable -p gpu --qos=${qos} --gres=${gres}"
      ;;
    large)
      printf '%s\n' "sbatch --parsable -p a100-gpu --qos=${qos} --gres=${gres}"
      ;;
    *)
      echo "Unsupported gpu profile: $gpu_profile" >&2
      return 2
      ;;
  esac
}
