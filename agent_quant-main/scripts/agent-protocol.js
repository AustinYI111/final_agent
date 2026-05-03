/**
 * Agent-to-Agent Protocol & Registry
 * 统一所有 Agent 的注册、消息通信、信号格式
 */

// ─────────────────────────────────────
//  Message Types
// ─────────────────────────────────────
const ACTION_SIGNAL_REQUEST  = "signal_request";
const ACTION_SIGNAL_RESPONSE = "signal_response";
const ACTION_VETO_REQUEST    = "veto_request";
const ACTION_VETO_RESPONSE   = "veto_response";
const ACTION_REGISTER        = "register";
const ACTION_STATUS_QUERY    = "status_query";
const ACTION_STATUS_REPLY    = "status_reply";

// ─────────────────────────────────────
//  Signal Payload 格式
// ─────────────────────────────────────
/**
 * @typedef {Object} SignalPayload
 * @property {"buy"|"sell"|"hold"} signal
 * @property {number} confidence 0.0 ~ 1.0
 * @property {Object} meta
 * @property {string} meta.agent_type
 * @property {number|null} meta.p_up
 */

/**
 * @typedef {Object} AgentMessage
 * @property {string} id
 * @property {string} from_agent
 * @property {string} to_agent
 * @property {string} action
 * @property {Object} payload
 * @property {number} timestamp
 */

// ─────────────────────────────────────
//  默认内置 Agent 配置
// ─────────────────────────────────────
const DEFAULT_AGENTS = [
    {
        id: "trend-default",
        name: "TrendAgent",
        type: "trend",
        enabled: true,
        isVeto: false,
        params: {
            short_window: 5,
            long_window: 20,
        },
        description: "双均线交叉趋势策略",
    },
    {
        id: "meanrev-default",
        name: "MeanRevAgent",
        type: "mean_reversion",
        enabled: true,
        isVeto: false,
        params: {
            window: 20,
            num_std: 1.2,
        },
        description: "布林带均值回归策略",
    },
    {
        id: "ml-default",
        name: "MLAgent",
        type: "ml",
        enabled: true,
        isVeto: true,  // 特殊：纯否决器
        params: {
            lookback: 10,
            min_train: 60,
            prob_threshold: 0.55,
        },
        description: "XGBoost 机器学习否决器",
    },
];

// ─────────────────────────────────────
//  Agent Registry
// ─────────────────────────────────────
class AgentRegistry {
    constructor() {
        this._agents = [];
        this._listeners = [];
        this._load();
    }

    // ── 存储 ──
    _storageKey() {
        return "agent_quant_v4_agents";
    }

    _load() {
        try {
            const stored = localStorage.getItem(this._storageKey());
            if (stored) {
                this._agents = JSON.parse(stored);
            } else {
                // 首次：加载默认 agents
                this._agents = [...DEFAULT_AGENTS];
                this._save();
            }
        } catch (e) {
            this._agents = [...DEFAULT_AGENTS];
        }
    }

    _save() {
        try {
            localStorage.setItem(this._storageKey(), JSON.stringify(this._agents));
        } catch (e) {
            console.error("Failed to save agents:", e);
        }
    }

    // ── CRUD ──
    getAll() {
        return [...this._agents];
    }

    getById(id) {
        return this._agents.find(a => a.id === id) || null;
    }

    getVetoAgent() {
        return this._agents.find(a => a.isVeto) || null;
    }

    getVotingAgents() {
        return this._agents.filter(a => !a.isVeto && a.enabled);
    }

    getEnabledAgents() {
        return this._agents.filter(a => a.enabled);
    }

    add(agent) {
        const id = "agent-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6);
        const newAgent = {
            id,
            name: agent.name || "新Agent",
            type: agent.type || "custom",
            enabled: true,
            isVeto: agent.isVeto || false,
            // Custom agents store code, built-in agents store params
            code: agent.code || null,
            params: agent.params || {},
            description: agent.description || "",
        };
        this._agents.push(newAgent);
        this._save();
        this._notify();
        return newAgent;
    }

    update(id, updates) {
        const idx = this._agents.findIndex(a => a.id === id);
        if (idx === -1) return null;
        this._agents[idx] = { ...this._agents[idx], ...updates };
        this._save();
        this._notify();
        return this._agents[idx];
    }

    remove(id) {
        // 不允许删除默认 agent
        if (id.endsWith("-default")) return false;
        const idx = this._agents.findIndex(a => a.id === id);
        if (idx === -1) return false;
        this._agents.splice(idx, 1);
        this._save();
        this._notify();
        return true;
    }

    reset() {
        this._agents = [...DEFAULT_AGENTS];
        this._save();
        this._notify();
    }

    // ── 监听变化 ──
    addListener(fn) {
        this._listeners.push(fn);
    }

    removeListener(fn) {
        this._listeners = this._listeners.filter(l => l !== fn);
    }

    _notify() {
        this._listeners.forEach(fn => fn(this.getAll()));
    }
}

// ─────────────────────────────────────
//  A2A Message Bus
// ─────────────────────────────────────
class AgentMessageBus {
    constructor(registry) {
        this.registry = registry;
        this._history = [];
    }

    // ── 构造消息 ──
    createMessage(from, to, action, payload) {
        return {
            id: "msg-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6),
            from_agent: from,
            to_agent: to,
            action: action,
            payload: payload,
            timestamp: Date.now() / 1000,
        };
    }

    // ── 广播信号请求给所有投票 Agent ──
    broadcastSignalRequest(df, market_state) {
        const agents = this.registry.getVotingAgents();
        const messages = [];

        for (const agent of agents) {
            const msg = this.createMessage(
                "coordinator",
                agent.name,
                ACTION_SIGNAL_REQUEST,
                { df, market_state }
            );
            messages.push(msg);
        }

        // ML Agent 单独发 veto_request
        const mlAgent = this.registry.getVetoAgent();
        if (mlAgent) {
            const vetoMsg = this.createMessage(
                "coordinator",
                mlAgent.name,
                ACTION_VETO_REQUEST,
                { market_state }
            );
            messages.push(vetoMsg);
        }

        return messages;
    }

    // ── 解析信号响应 ──
    parseSignalResponse(message) {
        return {
            from: message.from_agent,
            signal: message.payload.signal,
            confidence: message.payload.confidence,
            meta: message.payload.meta || {},
        };
    }

    // ── 解析否决响应 ──
    parseVetoResponse(message) {
        return {
            from: message.from_agent,
            vetoed: message.payload.vetoed,
            reason: message.payload.reason,
            p_up: message.payload.p_up,
        };
    }

    // ── 记录历史（调试用） ──
    getHistory() {
        return [...this._history];
    }

    clearHistory() {
        this._history = [];
    }
}

// ─────────────────────────────────────
//  全局单例
// ─────────────────────────────────────
let _registry = null;
let _messageBus = null;

function getAgentRegistry() {
    if (!_registry) {
        _registry = new AgentRegistry();
    }
    return _registry;
}

function getMessageBus() {
    if (!_messageBus) {
        _messageBus = new AgentMessageBus(getAgentRegistry());
    }
    return _messageBus;
}

// ─────────────────────────────────────
//  Agent 类型定义
// ─────────────────────────────────────
const AGENT_TYPES = {
    trend: {
        label: "TrendAgent",
        color: "#3b82f6",
        icon: "📈",
        params: [
            { key: "short_window", label: "短期均线 (天)", type: "number", default: 5, min: 2, max: 50 },
            { key: "long_window", label: "长期均线 (天)", type: "number", default: 20, min: 5, max: 200 },
        ],
    },
    mean_reversion: {
        label: "MeanRevAgent",
        color: "#f59e0b",
        icon: "↕️",
        params: [
            { key: "window", label: "窗口 (天)", type: "number", default: 20, min: 5, max: 100 },
            { key: "num_std", label: "布林带倍数", type: "number", default: 1.2, min: 0.5, max: 4, step: 0.1 },
        ],
    },
    ml: {
        label: "MLAgent",
        color: "#a855f7",
        icon: "🧠",
        isVeto: true,
        params: [
            { key: "lookback", label: "回看窗口 (天)", type: "number", default: 10, min: 3, max: 30 },
            { key: "min_train", label: "最小训练样本", type: "number", default: 60, min: 20, max: 500 },
            { key: "prob_threshold", label: "预测阈值", type: "number", default: 0.55, min: 0.5, max: 1.0, step: 0.01 },
        ],
    },
    custom: {
        label: "CustomAgent",
        color: "#10b981",
        icon: "🔧",
        params: [
            { key: "code", label: "策略代码", type: "textarea", default: "// 输入你的策略代码\nfunction generateSignal(df) {\n  // df: DataFrame with close, open, high, low, volume\n  // 返回: { signal: 'buy'|'sell'|'hold', confidence: 0.0~1.0 }\n}" },
        ],
    },
};
