package agents

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type dummyRunItem struct {
	RawItem any
	Type    string
}

func (dummyRunItem) isRunItem() {}

func (d dummyRunItem) ToInputItem() TResponseInputItem {
	if item, ok := d.RawItem.(TResponseInputItem); ok {
		return item
	}
	if item, ok := d.RawItem.(responses.ResponseInputItemFunctionCallOutputParam); ok {
		return TResponseInputItem{OfFunctionCallOutput: &item}
	}
	if item, ok := d.RawItem.(responses.ResponseInputItemItemReferenceParam); ok {
		return TResponseInputItem{OfItemReference: &item}
	}
	return TResponseInputItem{}
}

type fakeModel struct {
	nextOutput []TResponseOutputItem
	lastArgs   ModelResponseParams
}

func newFakeModel() *fakeModel {
	return &fakeModel{}
}

func (m *fakeModel) SetNextOutput(output []TResponseOutputItem) {
	m.nextOutput = output
}

func (m *fakeModel) GetResponse(_ context.Context, params ModelResponseParams) (*ModelResponse, error) {
	m.lastArgs = params
	return &ModelResponse{
		Output:     m.nextOutput,
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}, nil
}

func (m *fakeModel) StreamResponse(
	ctx context.Context,
	params ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	m.lastArgs = params
	response := responses.Response{
		ID:     "resp-test",
		Output: m.nextOutput,
	}
	return yield(ctx, TResponseStreamEvent{
		Type:           "response.completed",
		Response:       response,
		SequenceNumber: 0,
	})
}

func TestServerConversationTrackerPrepareInputFiltersItemsSeenByServerAndToolCalls(t *testing.T) {
	tracker := NewOpenAIServerConversationTracker("conv", "", false)

	originalInput := []TResponseInputItem{
		inputItemFromMap(t, map[string]any{"id": "input-1", "type": "message"}),
		inputItemFromMap(t, map[string]any{"id": "input-2", "type": "message"}),
	}
	newRawItem := inputItemFromMap(t, map[string]any{"type": "message", "content": "hello"})
	generatedItems := []RunItem{
		dummyRunItem{RawItem: inputItemFromMap(t, map[string]any{"id": "server-echo", "type": "message"}), Type: "message_output_item"},
		dummyRunItem{RawItem: newRawItem, Type: "message_output_item"},
		dummyRunItem{RawItem: inputItemFromMap(t, map[string]any{"call_id": "call-1", "output": "done", "type": "function_call_output"}), Type: "function_call_output_item"},
	}
	modelResponse := ModelResponse{
		Output: []TResponseOutputItem{
			outputItemFromMap(t, map[string]any{"call_id": "call-1", "output": "prior", "type": "function_call_output"}),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "resp-1",
	}
	sessionItems := []TResponseInputItem{
		inputItemFromMap(t, map[string]any{"id": "session-1", "type": "message"}),
	}

	tracker.HydrateFromState(
		InputItems(originalInput),
		generatedItems,
		[]ModelResponse{modelResponse},
		sessionItems,
	)

	prepared := tracker.PrepareInput(
		InputItems(originalInput),
		generatedItems,
	)

	require.Len(t, prepared, 1)
	assertInputItemEqual(t, newRawItem, prepared[0])
	assert.True(t, tracker.sentInitialInput)
	assert.Nil(t, tracker.remainingInitialInput)
}

func TestServerConversationTrackerMarkInputAsSentAndRewindInputRespectsRemainingInitialInput(t *testing.T) {
	tracker := NewOpenAIServerConversationTracker("conv2", "", false)

	pending1 := inputItemFromMap(t, map[string]any{"id": "p-1", "type": "message"})
	pending2 := inputItemFromMap(t, map[string]any{"id": "p-2", "type": "message"})
	tracker.remainingInitialInput = []TResponseInputItem{pending1, pending2}

	tracker.MarkInputAsSent([]TResponseInputItem{
		pending1,
		inputItemFromMap(t, map[string]any{"id": "p-2", "type": "message"}),
	})
	assert.Nil(t, tracker.remainingInitialInput)

	tracker.RewindInput([]TResponseInputItem{pending1})
	require.Len(t, tracker.remainingInitialInput, 1)
	assertInputItemEqual(t, pending1, tracker.remainingInitialInput[0])
}

func TestServerConversationTrackerTrackServerItemsFiltersRemainingInitialInputByFingerprint(t *testing.T) {
	tracker := NewOpenAIServerConversationTracker("conv3", "", false)

	pendingKept := inputItemFromMap(t, map[string]any{"id": "keep-me", "type": "message"})
	pendingFiltered := inputItemFromMap(t, map[string]any{
		"type":    "function_call_output",
		"call_id": "call-2",
		"output":  "x",
	})
	tracker.remainingInitialInput = []TResponseInputItem{pendingKept, pendingFiltered}

	modelResponse := ModelResponse{
		Output: []TResponseOutputItem{
			outputItemFromMap(t, map[string]any{
				"type":    "function_call_output",
				"call_id": "call-2",
				"output":  "x",
			}),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "resp-2",
	}

	tracker.TrackServerItems(&modelResponse)
	require.Len(t, tracker.remainingInitialInput, 1)
	assertInputItemEqual(t, pendingKept, tracker.remainingInitialInput[0])
}

func TestServerConversationTrackerPrepareInputDoesNotSkipFakeResponseIDs(t *testing.T) {
	tracker := NewOpenAIServerConversationTracker("conv5", "", false)

	modelResponse := ModelResponse{
		Output: []TResponseOutputItem{
			outputItemFromMap(t, map[string]any{"id": FakeResponsesID, "type": "message"}),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "resp-3",
	}
	tracker.TrackServerItems(&modelResponse)

	rawItem := inputItemFromMap(t, map[string]any{"id": FakeResponsesID, "type": "message", "content": "hello"})
	generatedItems := []RunItem{
		dummyRunItem{RawItem: rawItem, Type: "message_output_item"},
	}

	prepared := tracker.PrepareInput(InputItems(nil), generatedItems)
	require.Len(t, prepared, 1)
	assertInputItemEqual(t, rawItem, prepared[0])
}

func TestServerConversationTrackerGetNewResponseMarksFilteredInputAsSent(t *testing.T) {
	model := newFakeModel()
	model.SetNextOutput([]TResponseOutputItem{textOutputMessage("ok")})
	agent := New("test").WithModelInstance(model)

	tracker := NewOpenAIServerConversationTracker("conv4", "", false)
	item1 := textInputItem("first")
	item2 := textInputItem("second")

	filterFn := func(_ context.Context, data CallModelData) (*ModelInputData, error) {
		return &ModelInputData{
			Input:        []TResponseInputItem{data.ModelData.Input[0]},
			Instructions: data.ModelData.Instructions,
		}, nil
	}

	runConfig := RunConfig{CallModelInputFilter: filterFn}
	runner := Runner{}
	_, err := runner.getNewResponse(
		t.Context(),
		agent,
		param.Opt[string]{},
		[]TResponseInputItem{item1, item2},
		nil,
		nil,
		nil,
		runConfig,
		NewAgentToolUseTracker(),
		tracker,
		"",
		responses.ResponsePromptParam{},
	)
	require.NoError(t, err)

	input := model.lastArgs.Input.(InputItems)
	require.Len(t, input, 1)
	assertInputItemEqual(t, item1, input[0])

	fp1 := mustFingerprint(t, item1)
	fp2 := mustFingerprint(t, item2)
	_, ok := tracker.sentItemFingerprints[fp1]
	assert.True(t, ok)
	_, ok = tracker.sentItemFingerprints[fp2]
	assert.False(t, ok)
}

func TestServerConversationTrackerRunSingleTurnStreamedMarksFilteredInputAsSent(t *testing.T) {
	model := newFakeModel()
	model.SetNextOutput([]TResponseOutputItem{textOutputMessage("ok")})
	agent := New("test").WithModelInstance(model)

	tracker := NewOpenAIServerConversationTracker("conv6", "", false)
	item1 := textInputItem("first")
	item2 := textInputItem("second")

	filterFn := func(_ context.Context, data CallModelData) (*ModelInputData, error) {
		return &ModelInputData{
			Input:        []TResponseInputItem{data.ModelData.Input[0]},
			Instructions: data.ModelData.Instructions,
		}, nil
	}
	runConfig := RunConfig{CallModelInputFilter: filterFn}

	streamedResult := newRunResultStreaming(t.Context())
	streamedResult.setInput(InputItems{item1, item2})
	streamedResult.setNewItems(nil)

	_, err := Runner{}.runSingleTurnStreamed(
		t.Context(),
		streamedResult,
		agent,
		NoOpRunHooks{},
		runConfig,
		false,
		NewAgentToolUseTracker(),
		NewRunContextWrapper[any](nil),
		nil,
		"",
		tracker,
	)
	require.NoError(t, err)

	input := model.lastArgs.Input.(InputItems)
	require.Len(t, input, 1)
	assertInputItemEqual(t, item1, input[0])
	require.Len(t, tracker.remainingInitialInput, 1)
	assertInputItemEqual(t, item2, tracker.remainingInitialInput[0])
}

func inputItemFromMap(t *testing.T, payload map[string]any) TResponseInputItem {
	t.Helper()
	raw, err := json.Marshal(payload)
	require.NoError(t, err)
	var item TResponseInputItem
	param.SetJSON(raw, &item)
	return item
}

func outputItemFromMap(t *testing.T, payload map[string]any) TResponseOutputItem {
	t.Helper()
	raw, err := json.Marshal(payload)
	require.NoError(t, err)
	var item responses.ResponseOutputItemUnion
	require.NoError(t, json.Unmarshal(raw, &item))
	return item
}

func textInputItem(content string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: responses.EasyInputMessageRoleUser,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func textOutputMessage(content string) TResponseOutputItem {
	return responses.ResponseOutputItemUnion{
		ID:   "1",
		Type: "message",
		Role: constant.ValueOf[constant.Assistant](),
		Content: []responses.ResponseOutputMessageContentUnion{{
			Text:        content,
			Type:        "output_text",
			Annotations: nil,
		}},
		Status: string(responses.ResponseOutputMessageStatusCompleted),
	}
}

func assertInputItemEqual(t *testing.T, expected, actual TResponseInputItem) {
	t.Helper()
	assert.Equal(t, mustFingerprint(t, expected), mustFingerprint(t, actual))
}

func mustFingerprint(t *testing.T, item TResponseInputItem) string {
	t.Helper()
	fp, ok := fingerprintForTracker(item)
	require.True(t, ok)
	return fp
}
