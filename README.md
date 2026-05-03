# Agent Quant · 量化交易系统

多因子、多策略量化交易回测系统，支持趋势策略、均值回归策略、机器学习辅助策略的独立运行与融合决策。

---

## 系统架构

```
agent_quant/
├── api_server.py              # Flask REST API 服务
├── dashboard.html              # Web 可视化仪表板
├── scripts/                    # 前端 JS 资源
├── styles/                     # CSS 样式
├── multi_agent_quant/
│   ├── agents/                 # 交易智能体
│   │   ├── data_agent.py       # 数据获取（akshare 多源 fallback）
│   │   ├── trend_agent.py      # 趋势策略（MA 金叉/死叉）
│   │   ├── mean_reversion_agent.py  # 均值回归（Bollinger Bands）
│   │   ├── ml_agent.py         # ML 策略（XGBoost 二分类）
│   │   └── coordinator_agent.py # 融合协调器（动态权重）
│   ├── backtest/
│   │   └── backtest_engine.py  # 回测引擎
│   ├── data/                   # 数据目录
│   ├── experiments/            # 实验脚本
│   ├── models/                # 模型（XGBoost）
│   ├── outputs/               # 回测结果输出
│   └── utils/                 # 工具函数
```

---

## 核心策略

| 策略 | 说明 |
|------|------|
| **Trend-only** | 双均线交叉策略（MA5 / MA20），短期 MA > 长期 MA 买入，反之卖出 |
| **MeanRev-only** | Bollinger Bands 均值回归，price < 下轨买入，price > 上轨卖出 |
| **ML-only** | XGBoost 二分类，预测次日涨跌概率，p_up >= 阈值买入，p_down >= 阈值卖出 |
| **Fusion** | Trend + MeanRev + ML 三策略加权融合，动态权重分配 |

---

## Agent 设计

### DataAgent
- 数据源 fallback 链：东财 → 腾讯 → 新浪
- 自动缓存（parquet 优先），支持分段拉取（按年）
- 自动添加技术指标：MA5/10/20、RSI(14)、Bollinger Bands、波动率

### TrendAgent
- 双均线差值作为置信度来源
- 趋势持续期间持续发出信号，不只是交叉那一刻

### MeanReversionAgent
- 基于 Z-score 的布林带策略
- 偏离越极端，置信度越高

### MLAgent
- 模型：XGBoost 二分类（预测次日涨/跌）
- 特征：lookback 收益、波动率、均线偏离度
- 直接下单，p_up >= 阈值买入，p_down >= 阈值卖出

### CoordinatorAgent
- 融合决策：静态先验权重 + 动态表现权重（EWMA）
- 市场状态感知：趋势市场给趋势策略加权重，均值回归市场反向
- 三策略加权融合：Trend + MeanRev + ML 各自输出信号后加权求和决定最终决策

---

## 安装

```bash
pip install akshare pandas numpy flask flask-cors

# 可选（加速缓存）
pip install pyarrow fastparquet
```

---

## 快速启动

### 启动 API 服务

```bash
cd agent_quant-main
python api_server.py
# 或指定端口：python api_server.py --port 5000 --debug
```

启动后访问：
- 仪表板：http://127.0.0.1:5000/
- 健康检查：http://127.0.0.1:5000/api/health

### 使用实验脚本

```bash
# 单策略回测
python multi_agent_quant/experiments/run_strategy.py

# 多策略对比
python multi_agent_quant/experiments/compare_strategies.py

# 多 Agent 融合回测
python multi_agent_quant/experiments/run_multi_agent.py
```

---

## API 接口

### POST /api/backtest
启动后台回测任务，返回 task_id。

**请求体：**
```json
{
  "symbol": "600519",
  "start_date": "20200101",
  "end_date": "20241231",
  "adjust": "qfq",
  "strategies": ["Trend-only", "MeanRev-only", "ML-only", "Fusion"],
  "init_cash": 100000,
  "fee_rate": 0.0003,
  "slippage_bps": 5
}
```

### GET /api/backtest-status/{task_id}
查询回测进度。

### GET /api/results
获取最新回测结果（含 equity curve、交易列表、指标）。

---

## 回测指标

| 指标 | 说明 |
|------|------|
| totalReturn | 总收益率 |
| annualReturn | 年化收益率 |
| maxDrawdown | 最大回撤 |
| sharpe | 夏普比率 |
| sortino | Sortino 比率 |
| calmar | Calmar 比率 |
| numTrades | 交易次数 |
| maxConsecutiveLosses | 最大连续亏损次数 |
| monthlyReturns | 月度收益率 |

---

## 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| init_cash | 100000 | 初始资金 |
| fee_rate | 0.0003 | 手续费率（3bp）|
| slippage_bps | 5 | 滑点（bps）|
| trend_short | 5 | 趋势策略短期均线窗口 |
| trend_long | 20 | 趋势策略长期均线窗口 |
| mr_window | 20 | 均值回归布林带窗口 |
| mr_num_std | 1.2 | 均值回归标准差倍数 |
| ml_lookback | 10 | ML 特征回看窗口 |
| ml_prob_threshold | 0.55 | ML 概率阈值 |
| w_trend | 0.60 | 趋势策略初始权重 |
| w_mr | 0.40 | 均值回归策略初始权重 |
| w_ml | 0.25 | ML 辅助权重 |

---

## 数据格式

回测结果输出到 `multi_agent_quant/outputs/`：

- `equity_curve_{策略名}.csv` — 每日权益曲线
- `comparison_table.csv` — 多策略对比表
