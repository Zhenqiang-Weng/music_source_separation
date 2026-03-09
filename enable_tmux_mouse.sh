#!/usr/bin/env bash
set -euo pipefail

TMUX_CONF="$HOME/.tmux.conf"

# 创建或备份现有配置
if [ -f "$TMUX_CONF" ]; then
  cp "$TMUX_CONF" "$TMUX_CONF.bak"
  echo ">>> Backed up existing config to $TMUX_CONF.bak"
fi

# 添加鼠标支持配置
cat >> "$TMUX_CONF" <<'EOF'

# Enable mouse support
set -g mouse on
EOF

echo ">>> Mouse support enabled in $TMUX_CONF"
echo ">>> Restart tmux or run: tmux kill-server && tmux"
