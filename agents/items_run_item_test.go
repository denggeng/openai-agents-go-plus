package agents

import (
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"
)

func TestTResponseInputItemFromToolCallItemType_FileSearchCall(t *testing.T) {
	input := ResponseFileSearchToolCall(responses.ResponseFileSearchToolCall{
		ID:      "fs_1",
		Queries: []string{"hello"},
	})

	require.NotPanics(t, func() {
		out := TResponseInputItemFromToolCallItemType(input)
		require.NotNil(t, out.OfFileSearchCall)
		require.Equal(t, "fs_1", out.OfFileSearchCall.ID)
	})
}

func TestTResponseInputItemFromToolCallItemType_WebSearchCall(t *testing.T) {
	input := ResponseFunctionWebSearch(responses.ResponseFunctionWebSearch{
		ID: "ws_1",
		Action: responses.ResponseFunctionWebSearchActionUnion{
			Type:  "search",
			Query: "hello",
		},
	})

	require.NotPanics(t, func() {
		out := TResponseInputItemFromToolCallItemType(input)
		require.NotNil(t, out.OfWebSearchCall)
		require.Equal(t, "ws_1", out.OfWebSearchCall.ID)
		require.NotNil(t, out.OfWebSearchCall.Action.OfSearch)
		require.Equal(t, "hello", out.OfWebSearchCall.Action.OfSearch.Query)
	})
}
