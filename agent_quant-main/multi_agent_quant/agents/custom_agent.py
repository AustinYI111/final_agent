# agents/custom_agent.py
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np
import math
import json


def _clean_nan(obj):
    """递归替换 NaN/inf 为 None，确保 JSON 序列化正常"""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _js_to_python(obj):
    """递归将 JSObject 转换为 Python 对象"""
    import py_mini_racer
    if isinstance(obj, py_mini_racer.JSObject):
        obj = dict(obj)  # JSObject → dict
    if isinstance(obj, dict):
        return {k: _js_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_js_to_python(v) for v in obj]
    return obj


def _parse_js_result(result):
    """把 JS 返回值转为 Python dict（优先用 JSON 字符串，fallback 到 JSObject 转换）"""
    import py_mini_racer
    s = str(result)
    # 尝试 JSON 解析（JS 代码 return JSON.stringify(...) 时走到这里）
    try:
        import json as _json
        return _json.loads(s)
    except Exception:
        pass
    # fallback: JSObject → dict
    if isinstance(result, py_mini_racer.JSObject):
        return _js_to_python(result)
    return None


@dataclass
class CustomAgent:
    """
    使用 PyMiniRacer 执行前端传入的 JavaScript 代码。
    JS 代码接收 df 作为数组（{date, open, high, low, close, volume}），
    返回 {signal, confidence}。
    """
    code: str = ""
    name: str = "CustomAgent"

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        # 准备传递给 JS 的数据格式
        df_for_js = data.rename(columns=str.lower).reset_index()
        if "date" in df_for_js.columns:
            df_for_js["date"] = df_for_js["date"].astype(str)

        records = df_for_js.to_dict(orient="records")
        # 清理 NaN/inf，避免 JSON 序列化出错
        records = _clean_nan(records)

        # 用 json.dumps 序列化数据，避免 f-string 拼接产生 NaN 问题
        data_json = json.dumps(records, default=str)

        js_code = f"""
        (function() {{
            {self.code}
            var data = {data_json};
            var r;
            if (typeof generateSignal === 'function') {{
                r = generateSignal(data);
            }} else if (typeof generate_signal === 'function') {{
                r = generate_signal(data);
            }} else {{
                r = {{ signal: 'hold', confidence: 0.0, meta: {{ error: 'no generateSignal function found' }} }};
            }}
            return JSON.stringify(r);
        }})();
        """

        try:
            from py_mini_racer import MiniRacer
            ctx = MiniRacer()
            raw = ctx.eval(js_code)
            result = _parse_js_result(raw)

            if isinstance(result, dict):
                signal = str(result.get("signal", "hold")).lower()
                if signal not in ("buy", "sell", "hold"):
                    signal = "hold"
                confidence = float(result.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))
                meta = result.get("meta", {})
                ret = {
                    "signal": signal,
                    "confidence": confidence,
                    "meta": {**meta, "agent": self.name},
                }
                print(f"[CustomAgent] {self.name}: signal={ret['signal']} conf={ret['confidence']:.3f}")
                return ret
            else:
                print(f"[CustomAgent] {self.name}: invalid return type {type(result)} = {result}")
                return {"signal": "hold", "confidence": 0.0, "meta": {"error": "invalid return type"}}
        except Exception as e:
            print(f"[CustomAgent] {self.name}: ERROR {e}")
            return {"signal": "hold", "confidence": 0.0, "meta": {"error": str(e)}}
