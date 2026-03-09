#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="mss1"

# 检查会话是否存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo ">>> Session '$SESSION_NAME' already exists"
  tmux attach -t "$SESSION_NAME"
else
  # 创建新会话
  tmux new-session -d -s "$SESSION_NAME" -n train
  echo ">>> Session '$SESSION_NAME' created with window 'train'"
  tmux attach -t "$SESSION_NAME"
fi
