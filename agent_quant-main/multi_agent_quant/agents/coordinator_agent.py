from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import math


@dataclass
class CoordinatorAgent:
    # 静态先验权重
    agent_weights: Dict[str, float]

    # 融合决策阈值
    min_edge: float = 0.05
    min_score_to_trade: float = 0.05
    regime_boost: float = 0.25
    min_conf_when_trade: float = 0.20

    # 动态权重参数
    perf_alpha: float = 0.10
    perf_clip: float = 0.03
    perf_temperature: float = 0.50
    dyn_blend: float = 0.60
    min_weight_floor: float = 0.02

    # 内部状态：每个 agent 的近期表现分数（EWMA）
    perf_ewma: Dict[str, float] = field(default_factory=dict)

    # ──── 工具方法 ────
    def _normalize(self, w: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(v, 0.0) for v in w.values())
        if s <= 0:
            n = len(w) if len(w) else 1
            return {k: 1.0 / n for k in w.keys()}
        return {k: max(v, 0.0) / s for k, v in w.items()}

    def _apply_regime(self, weights: Dict[str, float], market_state: Optional[str]) -> Dict[str, float]:
        """市场状态偏置：趋势市场偏 trend，震荡市场偏 mean_reversion"""
        w = dict(weights)
        if market_state == "trend":
            if "trend" in w:
                w["trend"] = w["trend"] + self.regime_boost
            if "mean_reversion" in w:
                w["mean_reversion"] = max(0.0, w["mean_reversion"] - self.regime_boost)
        elif market_state == "range":
            if "mean_reversion" in w:
                w["mean_reversion"] = w["mean_reversion"] + self.regime_boost
            if "trend" in w:
                w["trend"] = max(0.0, w["trend"] - self.regime_boost)
        return self._normalize(w)

    # ──── 动态权重 ────
    def _ensure_perf_keys(self) -> None:
        for k in self.agent_weights.keys():
            self.perf_ewma.setdefault(k, 0.0)

    def update_performance(self, agent_outputs: Dict[str, Dict[str, Any]], next_ret: float) -> None:
        """根据下一根 bar 的实际收益更新各 Agent 的 EWMA 表现分数"""
        self._ensure_perf_keys()
        r = max(-self.perf_clip, min(self.perf_clip, float(next_ret)))

        for name, out in agent_outputs.items():
            if name not in self.agent_weights:
                continue
            sig = str(out.get("signal", "hold")).lower()
            conf = max(0.0, min(1.0, float(out.get("confidence", 0.0))))

            if sig == "buy":
                reward = +r * conf
            elif sig == "sell":
                reward = -r * conf
            else:
                reward = 0.0

            old = self.perf_ewma.get(name, 0.0)
            self.perf_ewma[name] = (1.0 - self.perf_alpha) * old + self.perf_alpha * reward

    def _softmax_weights(self) -> Dict[str, float]:
        """将 EWMA 表现分数通过 softmax 转为动态权重"""
        self._ensure_perf_keys()
        tau = max(1e-6, float(self.perf_temperature))
        keys = list(self.agent_weights.keys())
        vals = {k: self.perf_ewma.get(k, 0.0) / tau for k in keys}
        m = max(vals.values()) if vals else 0.0
        exps = {k: math.exp(v - m) for k, v in vals.items()}
        s = sum(exps.values()) if exps else 0.0
        if s <= 0:
            return self._normalize(dict(self.agent_weights))
        w = {k: exps[k] / s for k in exps.keys()}
        w = {k: max(self.min_weight_floor, w[k]) for k in w.keys()}
        return self._normalize(w)

    def _mix_static_dynamic(self, base: Dict[str, float]) -> Dict[str, float]:
        """混合静态权重和动态权重"""
        dyn = self._softmax_weights()
        b = max(0.0, min(1.0, float(self.dyn_blend)))
        mixed = {}
        keys = set(base.keys()) | set(dyn.keys())
        for k in keys:
            mixed[k] = (1.0 - b) * float(base.get(k, 0.0)) + b * float(dyn.get(k, 0.0))
        return self._normalize(mixed)

    # ──── 融合决策 ────
    def aggregate(self, agent_outputs: Dict[str, Dict[str, Any]], market_state: Optional[str] = None) -> Dict[str, Any]:
        """
        所有 Agent 共同投票，加权融合。
        1. 静态权重 + 市场状态偏置
        2. 混合动态表现权重
        3. 加权投票（Score = Σ weight_i × confidence_i）
        4. 边缘检测（edge < min_edge → hold）
        """
        # 1. 静态权重
        base = self._normalize(dict(self.agent_weights))

        # 2. 市场状态偏置
        w_regime = self._apply_regime(base, market_state)

        # 3. 混合动态权重
        w_main = self._mix_static_dynamic(w_regime)

        # 4. 加权投票
        score = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        details: Dict[str, Any] = {}

        for name, out in agent_outputs.items():
            sig = str(out.get("signal", "hold")).lower()
            if sig not in score:
                sig = "hold"
            conf = max(0.0, min(1.0, float(out.get("confidence", 0.0))))
            weight = float(w_main.get(name, 0.0))
            contrib = weight * conf
            score[sig] += contrib
            details[name] = {"signal": sig, "confidence": conf, "weight": weight, "contrib": contrib}

        # 5. 选 best
        best_sig = max(score, key=score.get)
        best = float(score[best_sig])
        second = sorted(score.values(), reverse=True)[1]
        edge = float(best - second)

        if best <= 0.0:
            best_sig = "hold"
            edge = 0.0

        min_trade = float(self.min_score_to_trade)
        if market_state == "high_vol":
            min_trade += 0.10

        if best_sig != "hold":
            if best < min_trade or edge < float(self.min_edge):
                best_sig = "hold"

        # 6. 输出置信度
        if best_sig == "hold":
            out_conf = 0.05
        else:
            supporters = [float(v.get("confidence", 0.0)) for v in details.values() if v.get("signal") == best_sig]
            out_conf = max(supporters) if supporters else best
            out_conf = max(float(self.min_conf_when_trade), min(1.0, float(out_conf)))

        return {
            "signal": best_sig,
            "confidence": float(out_conf),
            "meta": {
                "market_state": market_state,
                "score": score,
                "edge": edge,
                "weights": w_main,
                "perf_ewma": dict(self.perf_ewma),
                "details": details,
            },
        }
