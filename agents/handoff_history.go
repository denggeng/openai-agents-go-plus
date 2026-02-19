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

package agents

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

const (
	defaultConversationHistoryStart = "<CONVERSATION HISTORY>"
	defaultConversationHistoryEnd   = "</CONVERSATION HISTORY>"
)

var (
	conversationHistoryMu    sync.RWMutex
	conversationHistoryStart = defaultConversationHistoryStart
	conversationHistoryEnd   = defaultConversationHistoryEnd
)

// SetConversationHistoryWrappers overrides the markers used to wrap conversation summaries.
// Pass nil to leave either side unchanged.
func SetConversationHistoryWrappers(start *string, end *string) {
	conversationHistoryMu.Lock()
	defer conversationHistoryMu.Unlock()
	if start != nil {
		conversationHistoryStart = *start
	}
	if end != nil {
		conversationHistoryEnd = *end
	}
}

// ResetConversationHistoryWrappers restores the default <CONVERSATION HISTORY> markers.
func ResetConversationHistoryWrappers() {
	conversationHistoryMu.Lock()
	defer conversationHistoryMu.Unlock()
	conversationHistoryStart = defaultConversationHistoryStart
	conversationHistoryEnd = defaultConversationHistoryEnd
}

// GetConversationHistoryWrappers returns the current summary markers.
func GetConversationHistoryWrappers() (string, string) {
	conversationHistoryMu.RLock()
	defer conversationHistoryMu.RUnlock()
	return conversationHistoryStart, conversationHistoryEnd
}

// DefaultHandoffHistoryMapper returns a single assistant message summarizing the transcript.
func DefaultHandoffHistoryMapper(transcript []TResponseInputItem) []TResponseInputItem {
	return []TResponseInputItem{buildSummaryMessage(transcript)}
}

// NestHandoffHistory summarizes the previous transcript for the next agent.
func NestHandoffHistory(
	handoffInputData HandoffInputData,
	historyMapper HandoffHistoryMapper,
) HandoffInputData {
	normalizedHistory := normalizeInputHistory(handoffInputData.InputHistory)
	flattenedHistory := flattenNestedHistoryMessages(normalizedHistory)

	preItemsAsInputs := make([]TResponseInputItem, 0, len(handoffInputData.PreHandoffItems))
	filteredPreItems := make([]RunItem, 0, len(handoffInputData.PreHandoffItems))
	for _, runItem := range handoffInputData.PreHandoffItems {
		if shouldSkipHandoffItem(runItem) {
			continue
		}
		plainInput := runItem.ToInputItem()
		preItemsAsInputs = append(preItemsAsInputs, plainInput)
		if shouldForwardPreItem(plainInput) {
			filteredPreItems = append(filteredPreItems, runItem)
		}
	}

	newItemsAsInputs := make([]TResponseInputItem, 0, len(handoffInputData.NewItems))
	filteredInputItems := make([]RunItem, 0, len(handoffInputData.NewItems))
	for _, runItem := range handoffInputData.NewItems {
		if shouldSkipHandoffItem(runItem) {
			continue
		}
		plainInput := runItem.ToInputItem()
		newItemsAsInputs = append(newItemsAsInputs, plainInput)
		if shouldForwardNewItem(plainInput) {
			filteredInputItems = append(filteredInputItems, runItem)
		}
	}

	transcript := append(append(flattenedHistory, preItemsAsInputs...), newItemsAsInputs...)
	mapper := historyMapper
	if mapper == nil {
		mapper = DefaultHandoffHistoryMapper
	}
	historyItems := mapper(transcript)

	inputItems := make([]RunItem, 0, len(filteredInputItems))
	inputItems = append(inputItems, filteredInputItems...)

	return HandoffInputData{
		InputHistory:    InputItems(cloneInputItems(historyItems)),
		PreHandoffItems: slicesCloneRunItems(filteredPreItems),
		NewItems:        slicesCloneRunItems(handoffInputData.NewItems),
		InputItems:      inputItems,
		RunContext:      handoffInputData.RunContext,
	}
}

var summaryOnlyInputTypes = map[string]struct{}{
	"function_call":        {},
	"function_call_output": {},
}

func normalizeInputHistory(inputHistory Input) []TResponseInputItem {
	switch history := inputHistory.(type) {
	case nil:
		return nil
	case InputString:
		return ItemHelpers().InputToNewInputList(history)
	case InputItems:
		return cloneInputItems(history)
	default:
		panic(UserErrorf("unexpected Input type %T", history))
	}
}

func cloneInputItems(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	out := make([]TResponseInputItem, len(items))
	copy(out, items)
	return out
}

func slicesCloneRunItems(items []RunItem) []RunItem {
	if len(items) == 0 {
		return nil
	}
	out := make([]RunItem, len(items))
	copy(out, items)
	return out
}

func shouldSkipHandoffItem(item RunItem) bool {
	switch item.(type) {
	case ToolApprovalItem, *ToolApprovalItem:
		return true
	default:
		return false
	}
}

func buildSummaryMessage(transcript []TResponseInputItem) TResponseInputItem {
	transcriptCopy := cloneInputItems(transcript)
	var summaryLines []string
	if len(transcriptCopy) == 0 {
		summaryLines = []string{"(no previous turns recorded)"}
	} else {
		summaryLines = make([]string, 0, len(transcriptCopy))
		for idx, item := range transcriptCopy {
			summaryLines = append(summaryLines, formatTranscriptItem(idx+1, item))
		}
	}

	startMarker, endMarker := GetConversationHistoryWrappers()
	contentLines := []string{
		"For context, here is the conversation so far between the user and the previous agent:",
		startMarker,
	}
	contentLines = append(contentLines, summaryLines...)
	contentLines = append(contentLines, endMarker)
	content := strings.Join(contentLines, "\n")

	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: responses.EasyInputMessageRoleAssistant,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func formatTranscriptItem(index int, item TResponseInputItem) string {
	role := roleFromInputItem(item)
	if role != "" {
		content := stringifyInputItemContent(item)
		if content == "" {
			return formatSummaryLine(index, role)
		}
		return formatSummaryLine(index, role+": "+content)
	}
	itemType := inputItemType(item)
	rest := serializeInputItem(item)
	if rest == "" {
		return formatSummaryLine(index, itemType)
	}
	return formatSummaryLine(index, itemType+": "+rest)
}

func formatSummaryLine(index int, text string) string {
	return fmt.Sprintf("%d. %s", index, text)
}

func stringifyInputItemContent(item TResponseInputItem) string {
	switch {
	case item.OfMessage != nil:
		if item.OfMessage.Content.OfString.Valid() {
			return item.OfMessage.Content.OfString.Value
		}
		return serializeValue(item.OfMessage.Content)
	case item.OfInputMessage != nil:
		return serializeValue(item.OfInputMessage.Content)
	case item.OfOutputMessage != nil:
		return serializeValue(item.OfOutputMessage.Content)
	default:
		return ""
	}
}

func roleFromInputItem(item TResponseInputItem) string {
	switch {
	case item.OfMessage != nil:
		return string(item.OfMessage.Role)
	case item.OfInputMessage != nil:
		return item.OfInputMessage.Role
	case item.OfOutputMessage != nil:
		return string(item.OfOutputMessage.Role)
	default:
		return ""
	}
}

func inputItemType(item TResponseInputItem) string {
	switch {
	case item.OfFunctionCall != nil:
		return "function_call"
	case item.OfFunctionCallOutput != nil:
		return "function_call_output"
	case item.OfComputerCall != nil:
		return "computer_call"
	case item.OfComputerCallOutput != nil:
		return "computer_call_output"
	case item.OfFileSearchCall != nil:
		return "file_search_call"
	case item.OfWebSearchCall != nil:
		return "web_search_call"
	case item.OfShellCall != nil:
		return "shell_call"
	case item.OfShellCallOutput != nil:
		return "shell_call_output"
	case item.OfApplyPatchCall != nil:
		return "apply_patch_call"
	case item.OfApplyPatchCallOutput != nil:
		return "apply_patch_call_output"
	case item.OfMcpListTools != nil:
		return "mcp_list_tools"
	case item.OfMcpApprovalRequest != nil:
		return "mcp_approval_request"
	case item.OfMcpApprovalResponse != nil:
		return "mcp_approval_response"
	case item.OfReasoning != nil:
		return "reasoning"
	case item.OfCompaction != nil:
		return "compaction"
	case item.OfItemReference != nil:
		return "item_reference"
	default:
		return "item"
	}
}

func serializeInputItem(item TResponseInputItem) string {
	raw, err := json.Marshal(item)
	if err != nil {
		return serializeValue(item)
	}
	return string(raw)
}

func serializeValue(value any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	return string(raw)
}

func shouldForwardPreItem(inputItem TResponseInputItem) bool {
	if role := roleFromInputItem(inputItem); role == "assistant" {
		return false
	}
	itemType := inputItemType(inputItem)
	_, skip := summaryOnlyInputTypes[itemType]
	return !skip
}

func shouldForwardNewItem(inputItem TResponseInputItem) bool {
	if role := roleFromInputItem(inputItem); role != "" {
		return true
	}
	itemType := inputItemType(inputItem)
	_, skip := summaryOnlyInputTypes[itemType]
	return !skip
}

func flattenNestedHistoryMessages(items []TResponseInputItem) []TResponseInputItem {
	if len(items) == 0 {
		return nil
	}
	flattened := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		if nested := extractNestedHistoryTranscript(item); nested != nil {
			flattened = append(flattened, nested...)
			continue
		}
		flattened = append(flattened, item)
	}
	return flattened
}

func extractNestedHistoryTranscript(item TResponseInputItem) []TResponseInputItem {
	content, ok := messageStringContent(item)
	if !ok {
		return nil
	}
	startMarker, endMarker := GetConversationHistoryWrappers()
	startIdx := strings.Index(content, startMarker)
	endIdx := strings.Index(content, endMarker)
	if startIdx == -1 || endIdx == -1 || endIdx <= startIdx {
		return nil
	}
	startIdx += len(startMarker)
	body := content[startIdx:endIdx]
	lines := strings.Split(body, "\n")
	parsed := make([]TResponseInputItem, 0, len(lines))
	for _, line := range lines {
		parsedItem := parseSummaryLine(line)
		if parsedItem != (TResponseInputItem{}) {
			parsed = append(parsed, parsedItem)
		}
	}
	return parsed
}

func messageStringContent(item TResponseInputItem) (string, bool) {
	if item.OfMessage == nil {
		return "", false
	}
	if !item.OfMessage.Content.OfString.Valid() {
		return "", false
	}
	return item.OfMessage.Content.OfString.Value, true
}

func parseSummaryLine(line string) TResponseInputItem {
	stripped := strings.TrimSpace(line)
	if stripped == "" {
		return TResponseInputItem{}
	}
	if dotIndex := strings.Index(stripped, "."); dotIndex != -1 {
		if allDigits(stripped[:dotIndex]) {
			stripped = strings.TrimSpace(stripped[dotIndex+1:])
		}
	}
	rolePart, remainder, ok := strings.Cut(stripped, ":")
	if !ok {
		return TResponseInputItem{}
	}
	roleText := strings.TrimSpace(rolePart)
	if roleText == "" {
		return TResponseInputItem{}
	}
	role, _ := splitRoleAndName(roleText)
	content := strings.TrimSpace(remainder)
	return messageInputItem(role, content)
}

func splitRoleAndName(roleText string) (string, string) {
	if strings.HasSuffix(roleText, ")") && strings.Contains(roleText, "(") {
		openIdx := strings.LastIndex(roleText, "(")
		if openIdx != -1 {
			possibleName := strings.TrimSpace(roleText[openIdx+1 : len(roleText)-1])
			roleCandidate := strings.TrimSpace(roleText[:openIdx])
			if possibleName != "" {
				return ensureRole(roleCandidate), possibleName
			}
		}
	}
	return ensureRole(roleText), ""
}

func ensureRole(role string) string {
	role = strings.TrimSpace(role)
	if role == "" {
		return "developer"
	}
	return role
}

func messageInputItem(role string, content string) TResponseInputItem {
	normalized := normalizeRole(role)
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: normalized,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func normalizeRole(role string) responses.EasyInputMessageRole {
	switch strings.ToLower(role) {
	case "user":
		return responses.EasyInputMessageRoleUser
	case "assistant":
		return responses.EasyInputMessageRoleAssistant
	case "system":
		return responses.EasyInputMessageRoleSystem
	case "developer":
		return responses.EasyInputMessageRoleDeveloper
	default:
		return responses.EasyInputMessageRoleDeveloper
	}
}

func allDigits(text string) bool {
	if text == "" {
		return false
	}
	for _, r := range text {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
