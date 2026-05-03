# agents/ml_agent.py
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class MLAgent:
    """
    机器学习 Agent（XGBoost 二分类）
    输出：{signal, confidence}，与其他 Agent 一起参与投票融合
    """

    def __init__(self, lookback: int = 10, min_train_size: int = 60, prob_threshold: float = 0.55):
        self.lookback = int(lookback)
        self.min_train_size = int(min_train_size)
        self.prob_threshold = float(prob_threshold)
        self._model = None
        self._trained = False
        self._train_failed = False

    def _build_features(self, data: pd.DataFrame) -> tuple:
        """
        从 data 构建特征矩阵和标签。
        特征：lookback 收益、波动率、均线偏离度
        标签：次日涨=1，跌=0
        """
        close = pd.to_numeric(data["close"], errors="coerce").dropna()
        if len(close) < max(self.min_train_size, self.lookback + 5):
            return None, None, None

        lb = self.lookback

        # 特征：近 lb 收益
        ret_lb = (close.iloc[lb:] / close.iloc[lb:].shift(lb) - 1).dropna()

        # 波动率
        rets = close.pct_change().dropna()
        vol = rets.rolling(lb).std().iloc[lb:]

        # 均线偏离度
        ma = close.rolling(lb).mean().iloc[lb:]

        # 标签：次日涨跌（收益率 > 0 为涨）
        future_ret = (close.shift(-1) / close - 1).iloc[lb:]
        labels = (future_ret > 0).astype(int)

        # 对齐
        common_len = min(len(ret_lb), len(vol), len(ma), len(labels))
        ret_lb = ret_lb.iloc[:common_len]
        vol = vol.iloc[:common_len]
        ma = ma.iloc[:common_len]
        labels = labels.iloc[:common_len]

        dist = (close.iloc[lb:common_len + lb] / ma - 1).values
        X = np.column_stack([ret_lb.values, vol.values, dist])
        y = labels.values
        return X, y, ["ret_lb", "vol", "dist_ma"]

    def _train(self, data: pd.DataFrame) -> None:
        """用历史数据训练 XGBoost 模型，训练成功则不再训练"""
        if self._trained or self._train_failed:
            return

        X, y, _ = self._build_features(data)
        if X is None or len(X) < self.min_train_size:
            return

        # 检查标签是否只有一类
        if len(np.unique(y)) < 2:
            print(f"[MLAgent] 训练跳过: y只有一类 y={np.unique(y)} y[:10]={y[:10]}")
            self._train_failed = True
            return

        self._model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        self._model.fit(X, y)
        self._trained = True

    def generate_signal(self, data: pd.DataFrame) -> dict:
        # 首次调用时尝试训练
        if not self._trained and not self._train_failed:
            self._train(data)

        # 基础检查
        if data is None or "close" not in data.columns:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "missing_close"}}

        close = pd.to_numeric(data["close"], errors="coerce").dropna()
        if len(close) < max(5, self.min_train_size):
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "insufficient_data"}}

        lb = min(self.lookback, len(close) - 2)

        # 构建特征
        ret_lb = (close.iloc[-1] / close.iloc[-1 - lb] - 1.0) if lb > 0 else 0.0
        rets = close.pct_change().dropna()
        vol = float(rets.iloc[-lb:].std()) if len(rets) >= lb else 0.0
        ma = float(close.iloc[-min(20, len(close)):].mean())
        dist = (float(close.iloc[-1]) / ma - 1.0) if ma != 0 else 0.0

        # XGBoost 预测 or fallback
        if self._model is not None:
            try:
                X = np.array([[ret_lb, vol, dist]])
                p_up = float(self._model.predict_proba(X)[0][1])
            except Exception:
                p_up = 0.5
        else:
            # 训练失败，没有模型 → 输出 hold
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "ml_not_trained"}}

        # 信号
        if p_up >= self.prob_threshold:
            signal = "buy"
        elif p_up <= (1.0 - self.prob_threshold):
            signal = "sell"
        else:
            signal = "hold"

        confidence = float(min(1.0, abs(p_up - 0.5) * 2.0))

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "meta": {
                "p_up": round(float(p_up), 4),
                "features": {
                    "ret_lb": round(float(ret_lb), 4),
                    "dist_ma": round(float(dist), 4),
                    "vol": round(float(vol), 4),
                },
            },
        }
