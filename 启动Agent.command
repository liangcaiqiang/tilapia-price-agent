#!/bin/bash
# ─────────────────────────────────────────────
#  罗非鱼鱼价预测 Agent — 一键启动
#  双击此文件即可运行（macOS）
# ─────────────────────────────────────────────

cd "$(dirname "$0")"

echo ""
echo "══════════════════════════════════════════"
echo "   🐟 罗非鱼鱼价预测 Agent 启动中..."
echo "══════════════════════════════════════════"
echo ""

# 检查 Python3
if ! command -v python3 &>/dev/null; then
    echo "❌ 未找到 Python3，请先安装："
    echo "   https://www.python.org/downloads/"
    read -p "按回车键退出..."; exit 1
fi

echo "✓ Python 版本: $(python3 --version)"
echo "✓ 工作目录:    $(pwd)"
echo ""

# 检查依赖文件
for f in tilapia_price_agent.py news_intelligence.py run_agent.py config.json; do
    if [ ! -f "$f" ]; then
        echo "❌ 缺少文件: $f"
        read -p "按回车键退出..."; exit 1
    fi
done
echo "✓ 所有文件就绪"
echo ""

# ── 虚拟环境处理（解决 macOS Homebrew Python 限制）──
VENV_DIR="$(pwd)/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️  首次运行：创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ 虚拟环境创建失败"
        read -p "按回车键退出..."; exit 1
    fi
    echo "✓ 虚拟环境已创建"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo "✓ 虚拟环境已激活"

# 安装依赖（只在缺少时安装）
if ! python3 -c "import pandas, numpy, matplotlib" 2>/dev/null; then
    echo ""
    echo "⚙️  安装依赖库（首次约需1分钟）..."
    pip install pandas numpy matplotlib --quiet
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败"
        read -p "按回车键退出..."; exit 1
    fi
    echo "✓ 依赖安装完成"
fi

echo ""
echo "──────────────────────────────────────────"
python3 run_agent.py
echo ""
echo "──────────────────────────────────────────"
echo "Agent 已退出。按回车键关闭窗口..."
read
