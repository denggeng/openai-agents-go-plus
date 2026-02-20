package agents

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDropOrphanFunctionCallsPreservesNonMappingEntries(t *testing.T) {
	payload := []any{
		"plain-text-input",
		map[string]any{"type": "message", "role": "user", "content": "hello"},
		map[string]any{
			"type":      "function_call",
			"call_id":   "orphan_call",
			"name":      "orphan",
			"arguments": "{}",
		},
		map[string]any{
			"type":      "function_call",
			"call_id":   "paired_call",
			"name":      "paired",
			"arguments": "{}",
		},
		map[string]any{"type": "function_call_output", "call_id": "paired_call", "output": "ok"},
		map[string]any{"call_id": "not-a-tool-call"},
	}

	filtered := dropOrphanFunctionCalls(payload)

	assert.Contains(t, filtered, "plain-text-input")
	require.IsType(t, map[string]any{}, filtered[1])
	assert.Equal(t, "message", filtered[1].(map[string]any)["type"])
	assert.True(t, containsToolCall(filtered, "paired_call"))
	assert.False(t, containsToolCall(filtered, "orphan_call"))
}

func TestNormalizeAndEnsureInputItemFormatKeepNonDictEntries(t *testing.T) {
	item := "raw-item"
	assert.Equal(t, item, ensureInputItemFormat(item))
	assert.Equal(t, []any{item}, normalizeInputItemsForAPI([]any{item}))
}

func TestFingerprintInputItemHandlesEdgeCases(t *testing.T) {
	_, ok := fingerprintInputItem(nil, false)
	assert.False(t, ok)

	fingerprint, ok := fingerprintInputItem(
		map[string]any{"id": "id-1", "type": "message", "role": "user", "content": "hi"},
		true,
	)
	require.True(t, ok)
	assert.NotContains(t, fingerprint, "\"id\"")

	_, ok = fingerprintInputItem(brokenModelDump{}, false)
	assert.False(t, ok)

	_, ok = modelDumpWithoutWarnings(struct{}{})
	assert.False(t, ok)

	fingerprint, ok = fingerprintInputItem(opaqueItem{
		ID:      "internal-id",
		Type:    "message",
		Role:    "user",
		Content: "x",
	}, true)
	require.True(t, ok)
	assert.NotContains(t, fingerprint, "\"id\"")
}

func TestDeduplicateInputItemsHandlesFakeIDsAndApprovalRequestIDs(t *testing.T) {
	items := []TResponseInputItem{
		functionCallOutputItem("call-1", "first", param.NewOpt(FakeResponsesID)),
		functionCallOutputItem("call-1", "latest", param.NewOpt(FakeResponsesID)),
		mcpApprovalResponseItem("req-1", true),
		mcpApprovalResponseItem("req-1", false),
		plainMessageInput("plain"),
	}

	deduplicated := deduplicateInputItems(items)
	require.Len(t, deduplicated, 3)
	assert.NotNil(t, deduplicated[len(deduplicated)-1].OfMessage)

	latest := deduplicateInputItemsPreferringLatest(items[:2])
	require.Len(t, latest, 1)
	require.NotNil(t, latest[0].OfFunctionCallOutput)
	assert.Equal(t, "latest", latest[0].OfFunctionCallOutput.Output.OfString.Value)
}

func TestExtractMCPRequestIDSupportsDictsAndObjects(t *testing.T) {
	assert.Equal(t, "provider-id", extractMCPRequestID(map[string]any{
		"provider_data": map[string]any{"id": "provider-id"},
		"id":            "fallback-id",
	}))
	assert.Equal(t, "call-id", extractMCPRequestID(map[string]any{"call_id": "call-id"}))

	assert.Equal(t, "from-provider", extractMCPRequestID(withProviderData{
		ProviderData: map[string]any{"id": "from-provider"},
	}))

	assert.Equal(t, "", extractMCPRequestID(brokenMarshal{}))
}

func TestExtractMCPRequestIDFromRunVariants(t *testing.T) {
	assert.Equal(t, "provider-dict", extractMCPRequestIDFromRun(runWithRequestItem{
		RequestItem: map[string]any{
			"provider_data": map[string]any{"id": "provider-dict"},
			"id":            "fallback",
		},
	}))
	assert.Equal(t, "dict-id", extractMCPRequestIDFromRun(runWithRequestItem{
		RequestItem: map[string]any{"id": "dict-id"},
	}))
	assert.Equal(t, "provider-object", extractMCPRequestIDFromRun(runWithRequestItem{
		RequestItem: requestObject{
			ProviderData: map[string]any{"id": "provider-object"},
			ID:           "object-id",
			CallID:       "object-call-id",
		},
	}))
	assert.Equal(t, "camel-call", extractMCPRequestIDFromRun(runWithRequestItem{
		RequestItemCamel: map[string]any{"call_id": "camel-call"},
	}))
}

func TestRunItemToInputItemPreservesReasoningItemIDsByDefault(t *testing.T) {
	agent := &Agent{Name: "A"}
	reasoning := ReasoningItem{
		Agent: agent,
		RawItem: responses.ResponseReasoningItem{
			Type:    constant.ValueOf[constant.Reasoning](),
			ID:      "rs_123",
			Summary: []responses.ResponseReasoningItemSummary{},
		},
	}

	result, ok := runItemToInputItem(reasoning, "")
	require.True(t, ok)

	payload := map[string]any{}
	require.NoError(t, json.Unmarshal(mustMarshal(result), &payload))
	assert.Equal(t, "reasoning", payload["type"])
	assert.Equal(t, "rs_123", payload["id"])
}

func TestRunItemToInputItemOmitsReasoningItemIDsWhenConfigured(t *testing.T) {
	agent := &Agent{Name: "A"}
	reasoning := ReasoningItem{
		Agent: agent,
		RawItem: responses.ResponseReasoningItem{
			Type:    constant.ValueOf[constant.Reasoning](),
			ID:      "rs_456",
			Summary: []responses.ResponseReasoningItemSummary{},
		},
	}

	result, ok := runItemToInputItem(reasoning, ReasoningItemIDPolicyOmit)
	require.True(t, ok)

	payload := map[string]any{}
	require.NoError(t, json.Unmarshal(mustMarshal(result), &payload))
	assert.Equal(t, "reasoning", payload["type"])
	_, hasID := payload["id"]
	assert.False(t, hasID)
}

type brokenModelDump struct{}

func (brokenModelDump) ModelDump(_ bool, _ bool) (map[string]any, error) {
	return nil, errors.New("broken")
}

type opaqueItem struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content string `json:"content"`
}

type withProviderData struct {
	ProviderData map[string]any `json:"provider_data"`
}

type brokenMarshal struct{}

func (brokenMarshal) MarshalJSON() ([]byte, error) {
	return nil, errors.New("boom")
}

type runWithRequestItem struct {
	RequestItem      any `json:"request_item,omitempty"`
	RequestItemCamel any `json:"requestItem,omitempty"`
}

type requestObject struct {
	ProviderData map[string]any `json:"provider_data"`
	ID           string         `json:"id"`
	CallID       string         `json:"call_id"`
}

func functionCallOutputItem(callID, output string, id param.Opt[string]) TResponseInputItem {
	outputUnion := responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
		OfString: param.NewOpt(output),
	}
	paramValue := responses.ResponseInputItemFunctionCallOutputParam{
		CallID: callID,
		Output: outputUnion,
		ID:     id,
		Type:   constant.ValueOf[constant.FunctionCallOutput](),
	}
	return TResponseInputItem{OfFunctionCallOutput: &paramValue}
}

func mcpApprovalResponseItem(requestID string, approve bool) TResponseInputItem {
	paramValue := responses.ResponseInputItemMcpApprovalResponseParam{
		ApprovalRequestID: requestID,
		Approve:           approve,
		Type:              constant.ValueOf[constant.McpApprovalResponse](),
	}
	return TResponseInputItem{OfMcpApprovalResponse: &paramValue}
}

func plainMessageInput(text string) TResponseInputItem {
	message := responses.EasyInputMessageParam{
		Content: responses.EasyInputMessageContentUnionParam{
			OfString: param.NewOpt(text),
		},
		Role: responses.EasyInputMessageRoleUser,
		Type: responses.EasyInputMessageTypeMessage,
	}
	return TResponseInputItem{OfMessage: &message}
}

func containsToolCall(items []any, callID string) bool {
	for _, entry := range items {
		mapping, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		if mapping["type"] == "function_call" && mapping["call_id"] == callID {
			return true
		}
	}
	return false
}

func mustMarshal(value any) []byte {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return data
}
