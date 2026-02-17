package agents

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamingContext(t *testing.T) {
	model := newStreamingTestModel()
	agent := New("Assistant").
		WithInstructions("You are a helpful assistant.").
		WithModelInstance(model)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		result, err := Runner{}.RunStreamed(context.Background(), agent, "Tell me a joke")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, _ := w.(http.Flusher)
		_ = result.StreamEvents(func(event StreamEvent) error {
			if _, err := fmt.Fprintf(w, "%s\n\n", streamEventType(event)); err != nil {
				return err
			}
			if flusher != nil {
				flusher.Flush()
			}
			return nil
		})
	}))
	defer server.Close()

	resp, err := http.Post(server.URL+"/stream", "application/json", strings.NewReader("{}"))
	require.NoError(t, err)
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	lines := make([]string, 0)
	for _, line := range strings.Split(string(body), "\n") {
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}

	assert.Equal(t, []string{
		"agent_updated_stream_event",
		"raw_response_event",
		"raw_response_event",
		"raw_response_event",
		"raw_response_event",
		"raw_response_event",
		"raw_response_event",
		"raw_response_event",
		"raw_response_event",
		"raw_response_event",
		"run_item_stream_event",
	}, lines)
}

func streamEventType(event StreamEvent) string {
	switch v := event.(type) {
	case RawResponsesStreamEvent:
		return v.Type
	case RunItemStreamEvent:
		return v.Type
	case AgentUpdatedStreamEvent:
		return v.Type
	default:
		return ""
	}
}

type streamingTestModel struct {
	events []responses.ResponseStreamEventUnion
}

func newStreamingTestModel() *streamingTestModel {
	output := []TResponseOutputItem{textOutputMessageForStreaming("done")}
	response := responses.Response{
		ID:     "resp-test",
		Output: output,
	}

	events := []responses.ResponseStreamEventUnion{
		{Type: "response.created", SequenceNumber: 1},
		{Type: "response.in_progress", SequenceNumber: 2},
		{Type: "response.output_item.added", SequenceNumber: 3},
		{Type: "response.content_part.added", SequenceNumber: 4},
		{Type: "response.output_text.delta", SequenceNumber: 5},
		{Type: "response.output_text.done", SequenceNumber: 6},
		{Type: "response.content_part.done", SequenceNumber: 7},
		{Type: "response.output_item.done", SequenceNumber: 8},
		{Type: "response.completed", SequenceNumber: 9, Response: response},
	}

	return &streamingTestModel{events: events}
}

func (m *streamingTestModel) GetResponse(context.Context, ModelResponseParams) (*ModelResponse, error) {
	return nil, fmt.Errorf("streamingTestModel.GetResponse not implemented")
}

func (m *streamingTestModel) StreamResponse(
	ctx context.Context,
	_ ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	for _, event := range m.events {
		if err := yield(ctx, event); err != nil {
			return err
		}
	}
	return nil
}

func textOutputMessageForStreaming(content string) TResponseOutputItem {
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
