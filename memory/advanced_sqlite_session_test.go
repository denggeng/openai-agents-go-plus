// Copyright 2025 The NLP Odyssey Authors
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

package memory

import (
	"path/filepath"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func advancedTestMessage(role responses.EasyInputMessageRole, text string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
			Role:    role,
			Type:    responses.EasyInputMessageTypeMessage,
		},
	}
}

func advancedFunctionCall(callID, name string) TResponseInputItem {
	return TResponseInputItem{
		OfFunctionCall: &responses.ResponseFunctionToolCallParam{
			Arguments: "{}",
			CallID:    callID,
			Name:      name,
		},
	}
}

func advancedCustomToolCall(callID, name string) TResponseInputItem {
	return TResponseInputItem{
		OfCustomToolCall: &responses.ResponseCustomToolCallParam{
			CallID: callID,
			Input:  "{}",
			Name:   name,
		},
	}
}

func advancedMcpCall(id, serverLabel, name string) TResponseInputItem {
	return TResponseInputItem{
		OfMcpCall: &responses.ResponseInputItemMcpCallParam{
			ID:          id,
			Arguments:   "{}",
			Name:        name,
			ServerLabel: serverLabel,
		},
	}
}

func TestAdvancedSQLiteSession_BasicOps(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-basic",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_basic.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	items := []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Hello"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Hi there!"),
	}
	require.NoError(t, session.AddItems(ctx, items))

	retrieved, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Equal(t, items, retrieved)

	popped, err := session.PopItem(ctx)
	require.NoError(t, err)
	require.NotNil(t, popped)
	assert.Equal(t, items[1], *popped)

	require.NoError(t, session.ClearSession(ctx))
	retrieved, err = session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, retrieved)
}

func TestAdvancedSQLiteSession_BranchingFlow(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-branching",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_branching.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	turns := []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Q1"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "A1"),
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Q2"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "A2"),
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Q3"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "A3"),
	}
	require.NoError(t, session.AddItems(ctx, turns))

	branchName, err := session.CreateBranchFromTurn(ctx, 2, "experiment")
	require.NoError(t, err)
	assert.Equal(t, "experiment", branchName)
	assert.Equal(t, "experiment", session.CurrentBranchID())

	branchItems, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	require.Len(t, branchItems, 2)
	assert.Equal(t, "Q1", branchItems[0].OfMessage.Content.OfString.Value)
	assert.Equal(t, "A1", branchItems[1].OfMessage.Content.OfString.Value)

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Branch Q2"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Branch A2"),
	}))

	require.NoError(t, session.SwitchToBranch(MainBranchID))
	assert.Equal(t, MainBranchID, session.CurrentBranchID())
	mainItems, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	require.Len(t, mainItems, 6)
	assert.Equal(t, "Q3", mainItems[4].OfMessage.Content.OfString.Value)

	require.NoError(t, session.SwitchToBranch("experiment"))
	branchItems, err = session.GetItems(ctx, 0)
	require.NoError(t, err)
	require.Len(t, branchItems, 4)
	assert.Equal(t, "Branch Q2", branchItems[2].OfMessage.Content.OfString.Value)

	branches := session.ListBranches()
	assert.Equal(t, []string{"experiment", "main"}, branches)

	branchInfos, err := session.ListBranchInfos(ctx)
	require.NoError(t, err)
	require.Len(t, branchInfos, 2)
	assert.Equal(t, "experiment", branchInfos[0].BranchID)
	assert.Equal(t, 4, branchInfos[0].MessageCount)
	assert.Equal(t, 2, branchInfos[0].UserTurns)
	assert.True(t, branchInfos[0].IsCurrent)
	assert.NotEmpty(t, branchInfos[0].CreatedAt)
	assert.Equal(t, "main", branchInfos[1].BranchID)
	assert.Equal(t, 6, branchInfos[1].MessageCount)
	assert.Equal(t, 3, branchInfos[1].UserTurns)
	assert.False(t, branchInfos[1].IsCurrent)
	assert.NotEmpty(t, branchInfos[1].CreatedAt)

	require.NoError(t, session.SwitchToBranch(MainBranchID))
	require.NoError(t, session.DeleteBranch(ctx, "experiment"))
	branches = session.ListBranches()
	assert.Equal(t, []string{"main"}, branches)
	assert.Equal(t, MainBranchID, session.CurrentBranchID())
}

func TestAdvancedSQLiteSession_ListBranchInfos_EmptyBranch(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-branch-infos-empty",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_branch_infos_empty.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Q1"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "A1"),
	}))

	_, err = session.CreateBranchFromTurn(ctx, 1, "empty_branch")
	require.NoError(t, err)

	infos, err := session.ListBranchInfos(ctx)
	require.NoError(t, err)
	require.Len(t, infos, 2)

	assert.Equal(t, "empty_branch", infos[0].BranchID)
	assert.Equal(t, 0, infos[0].MessageCount)
	assert.Equal(t, 0, infos[0].UserTurns)
	assert.True(t, infos[0].IsCurrent)
	assert.Equal(t, "", infos[0].CreatedAt)

	assert.Equal(t, "main", infos[1].BranchID)
	assert.Equal(t, 2, infos[1].MessageCount)
	assert.Equal(t, 1, infos[1].UserTurns)
	assert.False(t, infos[1].IsCurrent)
	assert.NotEmpty(t, infos[1].CreatedAt)
}

func TestAdvancedSQLiteSession_BranchValidation(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-validation",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_validation.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	_, err = session.CreateBranchFromTurn(ctx, 0, "bad")
	require.ErrorContains(t, err, "turn must be > 0")

	_, err = session.CreateBranchFromTurn(ctx, 1, "")
	require.ErrorContains(t, err, "branch id is required")

	_, err = session.CreateBranchFromTurn(ctx, 1, "new")
	require.NoError(t, err)
	_, err = session.CreateBranchFromTurn(ctx, 1, "new")
	require.ErrorContains(t, err, "already exists")

	err = session.SwitchToBranch("missing")
	require.ErrorContains(t, err, "does not exist")

	err = session.DeleteBranch(ctx, "")
	require.ErrorContains(t, err, "branch id cannot be empty")

	err = session.DeleteBranch(ctx, "missing")
	require.ErrorContains(t, err, "does not exist")

	err = session.DeleteBranch(ctx, "new")
	require.ErrorContains(t, err, "without force")

	err = session.DeleteBranchForce(ctx, "new")
	require.NoError(t, err)
	assert.Equal(t, MainBranchID, session.CurrentBranchID())

	err = session.DeleteBranch(ctx, MainBranchID)
	require.ErrorContains(t, err, "cannot delete main branch")
}

func TestAdvancedSQLiteSession_ConversationAnalytics(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-analytics",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_analytics.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Tell me about cats"),
		advancedFunctionCall("call-1", "web_search"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Cats are great pets"),
	}))
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "What about dogs?"),
		advancedCustomToolCall("call-2", "animal_facts"),
		advancedMcpCall("mcp-1", "filesystem", "read_file"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Dogs are great pets too"),
	}))

	turns, err := session.GetConversationTurns(ctx, "")
	require.NoError(t, err)
	require.Len(t, turns, 2)
	assert.Equal(t, 1, turns[0].Turn)
	assert.Equal(t, "Tell me about cats", turns[0].Content)
	assert.Equal(t, "Tell me about cats", turns[0].FullContent)
	assert.NotEmpty(t, turns[0].Timestamp)
	assert.True(t, turns[0].CanBranch)
	assert.Equal(t, 2, turns[1].Turn)
	assert.Equal(t, "What about dogs?", turns[1].Content)
	assert.NotEmpty(t, turns[1].Timestamp)

	catTurns, err := session.FindTurnsByContent(ctx, "cats", "")
	require.NoError(t, err)
	require.Len(t, catTurns, 1)
	assert.Equal(t, 1, catTurns[0].Turn)
	assert.Equal(t, "Tell me about cats", catTurns[0].Content)
	assert.NotEmpty(t, catTurns[0].Timestamp)

	conversationByTurn, err := session.GetConversationByTurns(ctx, "")
	require.NoError(t, err)
	require.Len(t, conversationByTurn, 2)
	assert.Equal(t, "user", conversationByTurn[1][0].Type)
	assert.Equal(t, "function_call", conversationByTurn[1][1].Type)
	assert.Equal(t, "web_search", conversationByTurn[1][1].ToolName)
	assert.Equal(t, "assistant", conversationByTurn[1][2].Type)
	assert.Equal(t, "custom_tool_call", conversationByTurn[2][1].Type)
	assert.Equal(t, "animal_facts", conversationByTurn[2][1].ToolName)
	assert.Equal(t, "mcp_call", conversationByTurn[2][2].Type)
	assert.Equal(t, "filesystem.read_file", conversationByTurn[2][2].ToolName)

	toolUsage, err := session.GetToolUsage(ctx, "")
	require.NoError(t, err)
	assert.Equal(t, []AdvancedToolUsage{
		{ToolName: "web_search", UsageCount: 1, Turn: 1},
		{ToolName: "animal_facts", UsageCount: 1, Turn: 2},
		{ToolName: "filesystem.read_file", UsageCount: 1, Turn: 2},
	}, toolUsage)
}

func TestAdvancedSQLiteSession_CreateBranchFromContent(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-branch-content",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_branch_content.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "First question about math"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Math answer"),
	}))
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Second question about science"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Science answer"),
	}))
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Another math question"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Another math answer"),
	}))

	branchID, err := session.CreateBranchFromContent(ctx, "math", "math_branch")
	require.NoError(t, err)
	assert.Equal(t, "math_branch", branchID)
	assert.Equal(t, "math_branch", session.CurrentBranchID())

	items, err := session.GetItems(ctx, 0)
	require.NoError(t, err)
	assert.Empty(t, items)

	_, err = session.CreateBranchFromContent(ctx, "nonexistent", "missing_branch")
	require.ErrorContains(t, err, `no user turns found containing "nonexistent"`)
}

func TestAdvancedSQLiteSession_BranchScopedQueries(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-branch-scoped",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_branch_scoped.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Main turn one"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Main answer one"),
	}))
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Main turn two"),
		advancedFunctionCall("call-main", "main_tool"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Main answer two"),
	}))

	_, err = session.CreateBranchFromTurn(ctx, 2, "branch")
	require.NoError(t, err)
	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Branch-only turn"),
		advancedCustomToolCall("call-branch", "branch_tool"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Branch answer"),
	}))

	mainTurns, err := session.GetConversationTurns(ctx, MainBranchID)
	require.NoError(t, err)
	require.Len(t, mainTurns, 2)
	assert.Equal(t, "Main turn two", mainTurns[1].Content)

	branchTurns, err := session.GetConversationTurns(ctx, "")
	require.NoError(t, err)
	require.Len(t, branchTurns, 2)
	assert.Equal(t, "Branch-only turn", branchTurns[1].Content)

	mainUsage, err := session.GetToolUsage(ctx, MainBranchID)
	require.NoError(t, err)
	assert.Equal(t, []AdvancedToolUsage{
		{ToolName: "main_tool", UsageCount: 1, Turn: 2},
	}, mainUsage)

	branchUsage, err := session.GetToolUsage(ctx, "")
	require.NoError(t, err)
	assert.Equal(t, []AdvancedToolUsage{
		{ToolName: "branch_tool", UsageCount: 1, Turn: 2},
	}, branchUsage)
}

func TestAdvancedSQLiteSession_UsageTracking(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-usage",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_usage.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "First turn"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "First response"),
	}))
	require.NoError(t, session.StoreRunUsage(ctx, &usage.Usage{
		Requests:     1,
		InputTokens:  50,
		OutputTokens: 30,
		TotalTokens:  80,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: 10,
		},
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: 5,
		},
	}))

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Second turn"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Second response"),
	}))
	require.NoError(t, session.StoreRunUsage(ctx, &usage.Usage{
		Requests:     2,
		InputTokens:  75,
		OutputTokens: 45,
		TotalTokens:  120,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: 20,
		},
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: 15,
		},
	}))

	sessionUsage, err := session.GetSessionUsage(ctx, "")
	require.NoError(t, err)
	require.NotNil(t, sessionUsage)
	assert.Equal(t, uint64(3), sessionUsage.Requests)
	assert.Equal(t, uint64(125), sessionUsage.InputTokens)
	assert.Equal(t, uint64(75), sessionUsage.OutputTokens)
	assert.Equal(t, uint64(200), sessionUsage.TotalTokens)
	assert.Equal(t, 2, sessionUsage.TotalTurns)

	turnOneUsage, err := session.GetTurnUsage(ctx, "", 1)
	require.NoError(t, err)
	require.Len(t, turnOneUsage, 1)
	assert.Equal(t, 1, turnOneUsage[0].UserTurnNumber)
	assert.Equal(t, uint64(1), turnOneUsage[0].Requests)
	assert.Equal(t, uint64(80), turnOneUsage[0].TotalTokens)
	assert.Equal(t, int64(10), turnOneUsage[0].InputTokensDetails.CachedTokens)
	assert.Equal(t, int64(5), turnOneUsage[0].OutputTokensDetails.ReasoningTokens)

	turnTwoUsage, err := session.GetTurnUsage(ctx, "", 2)
	require.NoError(t, err)
	require.Len(t, turnTwoUsage, 1)
	assert.Equal(t, 2, turnTwoUsage[0].UserTurnNumber)
	assert.Equal(t, uint64(2), turnTwoUsage[0].Requests)
	assert.Equal(t, uint64(120), turnTwoUsage[0].TotalTokens)
	assert.Equal(t, int64(20), turnTwoUsage[0].InputTokensDetails.CachedTokens)
	assert.Equal(t, int64(15), turnTwoUsage[0].OutputTokensDetails.ReasoningTokens)

	allTurnUsage, err := session.GetTurnUsage(ctx, "")
	require.NoError(t, err)
	require.Len(t, allTurnUsage, 2)
	assert.Equal(t, 1, allTurnUsage[0].UserTurnNumber)
	assert.Equal(t, 2, allTurnUsage[1].UserTurnNumber)
}

func TestAdvancedSQLiteSession_UsageTrackingAcrossBranches(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-usage-branch",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_usage_branch.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Main question"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Main answer"),
	}))
	require.NoError(t, session.StoreRunUsage(ctx, &usage.Usage{
		Requests:     1,
		InputTokens:  50,
		OutputTokens: 30,
		TotalTokens:  80,
	}))

	_, err = session.CreateBranchFromTurn(ctx, 1, "usage_branch")
	require.NoError(t, err)

	require.NoError(t, session.AddItems(ctx, []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Branch question"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "Branch answer"),
	}))
	require.NoError(t, session.StoreRunUsage(ctx, &usage.Usage{
		Requests:     2,
		InputTokens:  100,
		OutputTokens: 60,
		TotalTokens:  160,
	}))

	mainUsage, err := session.GetSessionUsage(ctx, MainBranchID)
	require.NoError(t, err)
	require.NotNil(t, mainUsage)
	assert.Equal(t, uint64(1), mainUsage.Requests)
	assert.Equal(t, uint64(80), mainUsage.TotalTokens)
	assert.Equal(t, 1, mainUsage.TotalTurns)

	branchUsage, err := session.GetSessionUsage(ctx, "usage_branch")
	require.NoError(t, err)
	require.NotNil(t, branchUsage)
	assert.Equal(t, uint64(2), branchUsage.Requests)
	assert.Equal(t, uint64(160), branchUsage.TotalTokens)
	assert.Equal(t, 1, branchUsage.TotalTurns)

	totalUsage, err := session.GetSessionUsage(ctx, "")
	require.NoError(t, err)
	require.NotNil(t, totalUsage)
	assert.Equal(t, uint64(3), totalUsage.Requests)
	assert.Equal(t, uint64(240), totalUsage.TotalTokens)
	assert.Equal(t, 2, totalUsage.TotalTurns)

	branchTurnUsage, err := session.GetTurnUsage(ctx, "usage_branch")
	require.NoError(t, err)
	require.Len(t, branchTurnUsage, 1)
	assert.Equal(t, uint64(2), branchTurnUsage[0].Requests)
}

func TestAdvancedSQLiteSession_StoreRunUsageNoCurrentTurn(t *testing.T) {
	ctx := t.Context()
	session, err := NewAdvancedSQLiteSession(ctx, AdvancedSQLiteSessionParams{
		SessionID:        "advanced-usage-empty",
		DBDataSourceName: filepath.Join(t.TempDir(), "advanced_usage_empty.db"),
	})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, session.Close()) })

	require.NoError(t, session.StoreRunUsage(ctx, nil))
	require.NoError(t, session.StoreRunUsage(ctx, &usage.Usage{
		Requests:     1,
		InputTokens:  10,
		OutputTokens: 5,
		TotalTokens:  15,
	}))

	sessionUsage, err := session.GetSessionUsage(ctx, "")
	require.NoError(t, err)
	assert.Nil(t, sessionUsage)
}

func TestItemsBeforeTurn(t *testing.T) {
	items := []TResponseInputItem{
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Q1"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "A1"),
		advancedTestMessage(responses.EasyInputMessageRoleUser, "Q2"),
		advancedTestMessage(responses.EasyInputMessageRoleAssistant, "A2"),
	}

	assert.Nil(t, itemsBeforeTurn(items, 1))
	before2 := itemsBeforeTurn(items, 2)
	require.Len(t, before2, 2)
	assert.Equal(t, "Q1", before2[0].OfMessage.Content.OfString.Value)
}
