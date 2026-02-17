# Twilio SIP Realtime 示例

这个示例展示如何通过 Twilio SIP 来电触发 OpenAI Realtime（SIP）会话，并使用 Agents SDK 在电话中进行多智能体协作。

## 目录结构
- `agents.go`：FAQ/记录工具与三位 realtime agents（分诊/FAQ/记录）
- `server.go`：Webhook 接收、验签、接受来电、启动 Realtime SIP 观察
- `server_test.go`：任务去重逻辑测试

## 运行前准备

### 1) OpenAI 配置
需要以下环境变量：
- `OPENAI_API_KEY`：OpenAI API Key
- `OPENAI_WEBHOOK_SECRET`：OpenAI Webhook Secret（用于验签）

可选：
- `OPENAI_BASE_URL`：私有网关/代理时才需要

### 2) Twilio SIP 配置
需要在 Twilio 中配置 SIP Domain 或 SIP Trunk，使来电触发 OpenAI Realtime 的 webhook（由 OpenAI 配置）并通过你的服务回调到 `/openai/webhook`。这通常包含：
- 在 OpenAI 控制台创建 Realtime 的 Twilio SIP 集成
- 在 Twilio 中配置 Webhook URL 指向本服务的 `/openai/webhook`

> 注意：Twilio 和 OpenAI 的集成步骤可能因控制台更新而调整，请以官方文档为准。

## Webhook 验签头说明
本示例使用官方 Go SDK 的 `client.Webhooks.Unwrap` 来验签和解析事件。请求头中应包含：
- `Webhook-Id`
- `Webhook-Timestamp`
- `Webhook-Signature`

这些头由 OpenAI Webhooks 自动发送。若缺失或签名不匹配，示例会返回 `400`。

## 如何运行

```bash
export OPENAI_API_KEY="..."
export OPENAI_WEBHOOK_SECRET="..."
export PORT=8080

go run ./examples/realtime/twilio_sip
```

服务启动后：
- 健康检查：`GET /`
- Webhook：`POST /openai/webhook`

## 常见问题（FAQ）

### 1) 收到 `invalid webhook signature`
- 检查 `OPENAI_WEBHOOK_SECRET` 是否正确
- 确保请求头包含 `Webhook-Id` / `Webhook-Timestamp` / `Webhook-Signature`

### 2) `failed to accept call` 或 404
- 404 通常表示来电已挂断，示例会忽略并返回 200
- 其他错误请检查 OpenAI API Key 是否可用

### 3) Realtime 会话没有输出
- 确认 Twilio SIP 与 OpenAI Realtime 集成完成
- 检查服务日志是否出现 `realtime session error`
- 确认公网可访问 `POST /openai/webhook`

### 4) 如何退出会话
- 当 WebSocket 断开时会触发 `disconnected`，示例会结束观察任务

## 备注
- 示例内置 3 个 agents（分诊/FAQ/记录），可根据业务扩展
- `observeCall` 使用 `OpenAIRealtimeSIPModel` 连接到现有 call_id
