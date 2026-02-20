// Copyright 2026 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agents_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/denggeng/openai-agents-go-plus/agentstesting"
	"github.com/denggeng/openai-agents-go-plus/memory"
	"github.com/denggeng/openai-agents-go-plus/modelsettings"
	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	toolEcho = "approved_echo"
	toolNote = "approved_note"
)

var userMessages = []string{
	"Fetch profile for customer 104.",
	"Update note for customer 104.",
	"Delete note for customer 104.",
}

type scenarioStep struct {
	label          string
	message        string
	toolName       string
	approval       string
	expectedOutput string
}

type scenarioResult struct {
	approval agents.ToolApprovalItem
	items    []agents.TResponseInputItem
}

type scenarioModel struct {
	counter int
}

type queryArgs struct {
	Query string `json:"query"`
}

func (m *scenarioModel) GetResponse(_ context.Context, params agents.ModelResponseParams) (*agents.ModelResponse, error) {
	if inputHasRejection(params.Input) {
		return &agents.ModelResponse{
			Output: []agents.TResponseOutputItem{agentstesting.GetTextMessage(agents.DefaultApprovalRejectionMessage)},
			Usage:  usage.NewUsage(),
		}, nil
	}

	toolName := toolEcho
	if tc, ok := params.ModelSettings.ToolChoice.(modelsettings.ToolChoiceString); ok {
		if tc != "" && tc != modelsettings.ToolChoiceAuto && tc != modelsettings.ToolChoiceRequired && tc != modelsettings.ToolChoiceNone {
			toolName = tc.String()
		}
	}

	userMessage := extractUserMessage(params.Input)
	m.counter++
	callID := fmt.Sprintf("call_%d", m.counter)
	args, _ := json.Marshal(map[string]string{"query": userMessage})

	toolCall := agents.TResponseOutputItem{
		ID:        callID,
		CallID:    callID,
		Name:      toolName,
		Type:      "function_call",
		Arguments: string(args),
	}

	return &agents.ModelResponse{
		Output: []agents.TResponseOutputItem{toolCall},
		Usage:  usage.NewUsage(),
	}, nil
}

func (m *scenarioModel) StreamResponse(_ context.Context, _ agents.ModelResponseParams, _ agents.ModelStreamResponseCallback) error {
	return fmt.Errorf("streaming not supported in scenario model")
}

type scenarioListSession struct {
	id    string
	items []agents.TResponseInputItem
}

func (s *scenarioListSession) SessionID(context.Context) string {
	if s.id == "" {
		return "memory"
	}
	return s.id
}

func (s *scenarioListSession) GetItems(_ context.Context, limit int) ([]agents.TResponseInputItem, error) {
	if limit <= 0 || limit >= len(s.items) {
		return append([]agents.TResponseInputItem(nil), s.items...), nil
	}
	start := len(s.items) - limit
	if start < 0 {
		start = 0
	}
	return append([]agents.TResponseInputItem(nil), s.items[start:]...), nil
}

func (s *scenarioListSession) AddItems(_ context.Context, items []agents.TResponseInputItem) error {
	s.items = append(s.items, items...)
	return nil
}

func (s *scenarioListSession) PopItem(context.Context) (*agents.TResponseInputItem, error) {
	if len(s.items) == 0 {
		return nil, nil
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return &item, nil
}

func (s *scenarioListSession) ClearSession(context.Context) error {
	s.items = nil
	return nil
}

type conversationTransport struct {
	mu             sync.Mutex
	items          []json.RawMessage
	conversationID string
}

func (t *conversationTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	path := strings.TrimPrefix(req.URL.Path, "/")
	path = strings.TrimPrefix(path, "v1/")

	switch {
	case req.Method == http.MethodPost && path == "conversations":
		t.mu.Lock()
		if t.conversationID == "" {
			t.conversationID = "conv_test"
		}
		convID := t.conversationID
		t.mu.Unlock()
		return jsonResponse(http.StatusOK, map[string]any{
			"id":         convID,
			"created_at": 0,
			"metadata":   map[string]any{},
			"object":     "conversation",
		}), nil

	case req.Method == http.MethodPost && strings.HasSuffix(path, "/items") && strings.HasPrefix(path, "conversations/"):
		body, _ := io.ReadAll(req.Body)
		_ = req.Body.Close()
		var payload struct {
			Items []json.RawMessage `json:"items"`
		}
		_ = json.Unmarshal(body, &payload)
		t.mu.Lock()
		t.items = append(t.items, payload.Items...)
		t.mu.Unlock()
		return jsonResponse(http.StatusOK, map[string]any{
			"object":   "list",
			"data":     []any{},
			"first_id": "",
			"last_id":  "",
			"has_more": false,
		}), nil

	case req.Method == http.MethodGet && strings.HasSuffix(path, "/items") && strings.HasPrefix(path, "conversations/"):
		t.mu.Lock()
		items := append([]json.RawMessage(nil), t.items...)
		t.mu.Unlock()
		return jsonResponse(http.StatusOK, struct {
			Object  string            `json:"object"`
			Data    []json.RawMessage `json:"data"`
			FirstID string            `json:"first_id"`
			LastID  string            `json:"last_id"`
			HasMore bool              `json:"has_more"`
		}{
			Object:  "list",
			Data:    items,
			FirstID: "",
			LastID:  "",
			HasMore: false,
		}), nil

	case req.Method == http.MethodDelete && strings.HasPrefix(path, "conversations/"):
		t.mu.Lock()
		convID := t.conversationID
		t.items = nil
		t.mu.Unlock()
		return jsonResponse(http.StatusOK, map[string]any{
			"id":      convID,
			"deleted": true,
			"object":  "conversation.deleted",
		}), nil
	}

	return jsonResponse(http.StatusNotFound, map[string]any{"error": "not found"}), nil
}

func jsonResponse(status int, payload any) *http.Response {
	data, _ := json.Marshal(payload)
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(bytes.NewReader(data)),
		Header: http.Header{
			"Content-Type": []string{"application/json"},
		},
	}
}

func TestHITLSessionScenarioMemory(t *testing.T) {
	executeCounts := map[string]int{}
	model := &scenarioModel{}
	session := &scenarioListSession{id: "memory"}
	runner := agents.Runner{Config: agents.RunConfig{Session: session}}

	steps := []scenarioStep{
		{
			label:          "turn 1",
			message:        userMessages[0],
			toolName:       toolEcho,
			approval:       "approve",
			expectedOutput: fmt.Sprintf("approved:%s", userMessages[0]),
		},
		{
			label:          "turn 2 (rehydrated)",
			message:        userMessages[1],
			toolName:       toolNote,
			approval:       "approve",
			expectedOutput: fmt.Sprintf("approved_note:%s", userMessages[1]),
		},
		{
			label:          "turn 3 (rejected)",
			message:        userMessages[2],
			toolName:       toolEcho,
			approval:       "reject",
			expectedOutput: agents.DefaultApprovalRejectionMessage,
		},
	}

	for i, step := range steps {
		result := runScenarioStep(t, runner, model, &executeCounts, step)
		assertCounts(t, result.items, i+1)
		assertStepOutput(t, result.items, result.approval, step)
	}

	assert.Equal(t, 1, executeCounts[toolEcho])
	assert.Equal(t, 1, executeCounts[toolNote])
}

func TestHITLSessionScenarioOpenAIConversations(t *testing.T) {
	executeCounts := map[string]int{}
	model := &scenarioModel{}

	transport := &conversationTransport{conversationID: "conv_test"}
	client := openai.NewClient(
		option.WithAPIKey("test"),
		option.WithBaseURL("http://example.test/v1/"),
		option.WithHTTPClient(&http.Client{Transport: transport}),
	)

	session := memory.NewOpenAIConversationsSession(memory.OpenAIConversationsSessionParams{
		ConversationID: "conv_test",
		Client:         &client,
	})

	runner := agents.Runner{Config: agents.RunConfig{Session: session}}

	steps := []scenarioStep{
		{
			label:          "turn 1",
			message:        userMessages[0],
			toolName:       toolEcho,
			approval:       "approve",
			expectedOutput: fmt.Sprintf("approved:%s", userMessages[0]),
		},
		{
			label:          "turn 2 (rehydrated)",
			message:        userMessages[1],
			toolName:       toolNote,
			approval:       "approve",
			expectedOutput: fmt.Sprintf("approved_note:%s", userMessages[1]),
		},
		{
			label:          "turn 3 (rejected)",
			message:        userMessages[2],
			toolName:       toolEcho,
			approval:       "reject",
			expectedOutput: agents.DefaultApprovalRejectionMessage,
		},
	}

	for i, step := range steps {
		result := runScenarioStep(t, runner, model, &executeCounts, step)
		assertCounts(t, result.items, i+1)
		assertStepOutput(t, result.items, result.approval, step)
	}

	assert.Equal(t, 1, executeCounts[toolEcho])
	assert.Equal(t, 1, executeCounts[toolNote])

	require.NoError(t, session.ClearSession(t.Context()))
}

func runScenarioStep(
	t *testing.T,
	runner agents.Runner,
	model *scenarioModel,
	executeCounts *map[string]int,
	step scenarioStep,
) scenarioResult {
	t.Helper()

	echoTool := agents.NewFunctionTool(toolEcho, "Echoes back the provided query after approval.", func(_ context.Context, args queryArgs) (string, error) {
		counts := *executeCounts
		counts[toolEcho] = counts[toolEcho] + 1
		return fmt.Sprintf("approved:%s", args.Query), nil
	})
	echoTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	noteTool := agents.NewFunctionTool(toolNote, "Records the provided query after approval.", func(_ context.Context, args queryArgs) (string, error) {
		counts := *executeCounts
		counts[toolNote] = counts[toolNote] + 1
		return fmt.Sprintf("approved_note:%s", args.Query), nil
	})
	noteTool.NeedsApproval = agents.FunctionToolNeedsApprovalEnabled()

	agent := &agents.Agent{
		Name:            fmt.Sprintf("Scenario %s", step.label),
		Instructions:    agents.InstructionsStr(fmt.Sprintf("Always call %s before responding.", step.toolName)),
		Model:           paramOptAgentModel(model),
		Tools:           []agents.Tool{echoTool, noteTool},
		ModelSettings:   modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceString(step.toolName)},
		ToolUseBehavior: agents.StopOnFirstTool(),
	}

	first, err := runner.Run(t.Context(), agent, step.message)
	require.NoError(t, err)
	require.Len(t, first.Interruptions, 1)

	approval := first.Interruptions[0]
	state := agents.NewRunStateFromResult(*first, 1, 10)
	if step.approval == "reject" {
		require.NoError(t, state.RejectTool(approval, ""))
	} else {
		require.NoError(t, state.ApproveTool(approval))
	}

	resumed, err := runner.RunFromState(t.Context(), agent, state)
	require.NoError(t, err)
	assert.Empty(t, resumed.Interruptions)
	assert.Equal(t, step.expectedOutput, resumed.FinalOutput)

	items, err := runner.Config.Session.GetItems(t.Context(), 0)
	require.NoError(t, err)
	return scenarioResult{approval: approval, items: items}
}

func paramOptAgentModel(model agents.Model) param.Opt[agents.AgentModel] {
	return param.NewOpt(agents.NewAgentModel(model))
}

func inputHasRejection(input agents.Input) bool {
	items, ok := input.(agents.InputItems)
	if !ok {
		return false
	}
	for _, item := range items {
		if item.OfFunctionCallOutput == nil {
			continue
		}
		if item.OfFunctionCallOutput.Output.OfString.Valid() &&
			item.OfFunctionCallOutput.Output.OfString.Value == agents.DefaultApprovalRejectionMessage {
			return true
		}
	}
	return false
}

func extractUserMessage(input agents.Input) string {
	switch v := input.(type) {
	case agents.InputString:
		return v.String()
	case agents.InputItems:
		for i := len(v) - 1; i >= 0; i-- {
			item := v[i]
			if item.OfMessage == nil {
				continue
			}
			if item.OfMessage.Role != responses.EasyInputMessageRoleUser {
				continue
			}
			if item.OfMessage.Content.OfString.Valid() {
				return item.OfMessage.Content.OfString.Value
			}
		}
	}
	return ""
}

func assertCounts(t *testing.T, items []agents.TResponseInputItem, turn int) {
	t.Helper()
	payloads := toMapItems(t, items)
	assert.Equal(t, turn, countUserMessages(payloads))
	assert.Equal(t, turn, countFunctionCalls(payloads))
	assert.Equal(t, turn, countFunctionOutputs(payloads))
}

func assertStepOutput(t *testing.T, items []agents.TResponseInputItem, approval agents.ToolApprovalItem, step scenarioStep) {
	t.Helper()
	payloads := toMapItems(t, items)
	lastUser := getLastUserText(payloads)
	assert.Equal(t, step.message, lastUser)

	lastCall := findLastFunctionCall(payloads)
	lastResult := findLastFunctionOutput(payloads)
	approvalCallID := extractApprovalCallID(approval)

	require.NotNil(t, lastCall)
	require.NotNil(t, lastResult)
	assert.Equal(t, step.toolName, lastCall["name"])
	assert.Equal(t, approvalCallID, lastCall["call_id"])
	assert.Equal(t, approvalCallID, lastResult["call_id"])
	assert.Equal(t, step.expectedOutput, extractOutputText(lastResult))
}

func toMapItems(t *testing.T, items []agents.TResponseInputItem) []map[string]any {
	t.Helper()
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		data, err := item.MarshalJSON()
		require.NoError(t, err)
		var payload map[string]any
		require.NoError(t, json.Unmarshal(data, &payload))
		out = append(out, payload)
	}
	return out
}

func countUserMessages(items []map[string]any) int {
	count := 0
	for _, item := range items {
		if role, ok := item["role"].(string); ok && role == "user" {
			count++
		}
	}
	return count
}

func countFunctionCalls(items []map[string]any) int {
	count := 0
	for _, item := range items {
		if typ, ok := item["type"].(string); ok && typ == "function_call" {
			count++
		}
	}
	return count
}

func countFunctionOutputs(items []map[string]any) int {
	count := 0
	for _, item := range items {
		if typ, ok := item["type"].(string); ok && typ == "function_call_output" {
			count++
		}
	}
	return count
}

func findLastFunctionCall(items []map[string]any) map[string]any {
	for i := len(items) - 1; i >= 0; i-- {
		if typ, ok := items[i]["type"].(string); ok && typ == "function_call" {
			return items[i]
		}
	}
	return nil
}

func findLastFunctionOutput(items []map[string]any) map[string]any {
	for i := len(items) - 1; i >= 0; i-- {
		if typ, ok := items[i]["type"].(string); ok && typ == "function_call_output" {
			return items[i]
		}
	}
	return nil
}

func getLastUserText(items []map[string]any) string {
	for i := len(items) - 1; i >= 0; i-- {
		if role, ok := items[i]["role"].(string); ok && role == "user" {
			return extractUserText(items[i])
		}
	}
	return ""
}

func extractUserText(item map[string]any) string {
	content, ok := item["content"]
	if !ok {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var out strings.Builder
		for _, entry := range v {
			part, ok := entry.(map[string]any)
			if !ok {
				continue
			}
			if part["type"] == "input_text" {
				if text, ok := part["text"].(string); ok {
					out.WriteString(text)
				}
			}
		}
		return out.String()
	default:
		return ""
	}
}

func extractApprovalCallID(approval agents.ToolApprovalItem) string {
	switch v := approval.RawItem.(type) {
	case responses.ResponseFunctionToolCall:
		if v.CallID != "" {
			return v.CallID
		}
		return v.ID
	case agents.ResponseFunctionToolCall:
		call := responses.ResponseFunctionToolCall(v)
		if call.CallID != "" {
			return call.CallID
		}
		return call.ID
	case map[string]any:
		if callID, ok := v["call_id"].(string); ok && callID != "" {
			return callID
		}
		if id, ok := v["id"].(string); ok {
			return id
		}
	}
	return ""
}

func extractOutputText(item map[string]any) string {
	if item == nil {
		return ""
	}
	output, ok := item["output"]
	if !ok {
		return ""
	}
	switch v := output.(type) {
	case string:
		return v
	case []any:
		for _, entry := range v {
			part, ok := entry.(map[string]any)
			if !ok {
				continue
			}
			if part["type"] == "input_text" {
				if text, ok := part["text"].(string); ok {
					return text
				}
			}
		}
	case map[string]any:
		if v["type"] == "input_text" {
			if text, ok := v["text"].(string); ok {
				return text
			}
		}
	}
	return ""
}
