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

package codex

import "github.com/denggeng/openai-agents-go-plus/agents"

// ThreadItem is an item emitted inside item.* Codex stream events.
type ThreadItem interface {
	Type() string
}

type CommandExecutionItem struct {
	ID               string
	Command          string
	Status           string
	AggregatedOutput string
	ExitCode         *int
}

func (CommandExecutionItem) Type() string { return "command_execution" }

type FileUpdateChange struct {
	Path string
	Kind string
}

type FileChangeItem struct {
	ID      string
	Changes []FileUpdateChange
	Status  string
}

func (FileChangeItem) Type() string { return "file_change" }

type McpToolCallResult struct {
	Content           []any
	StructuredContent any
}

type McpToolCallError struct {
	Message string
}

type McpToolCallItem struct {
	ID        string
	Server    string
	Tool      string
	Arguments any
	Status    string
	Result    *McpToolCallResult
	Error     *McpToolCallError
}

func (McpToolCallItem) Type() string { return "mcp_tool_call" }

type AgentMessageItem struct {
	ID   string
	Text string
}

func (AgentMessageItem) Type() string { return "agent_message" }

type ReasoningItem struct {
	ID   string
	Text string
}

func (ReasoningItem) Type() string { return "reasoning" }

type WebSearchItem struct {
	ID    string
	Query string
}

func (WebSearchItem) Type() string { return "web_search" }

type ErrorItem struct {
	ID      string
	Message string
}

func (ErrorItem) Type() string { return "error" }

type TodoItem struct {
	Text      string
	Completed bool
}

type TodoListItem struct {
	ID    string
	Items []TodoItem
}

func (TodoListItem) Type() string { return "todo_list" }

type UnknownThreadItem struct {
	ItemType string
	Payload  map[string]any
	ID       *string
}

func (item UnknownThreadItem) Type() string { return item.ItemType }

func IsAgentMessageItem(item ThreadItem) bool {
	_, ok := item.(AgentMessageItem)
	return ok
}

func CoerceThreadItem(raw any) (ThreadItem, error) {
	mapping, ok := toStringAnyMap(raw)
	if !ok {
		return nil, agents.NewUserError("thread item payload must be a mapping")
	}

	itemType, _ := mapping["type"].(string)
	switch itemType {
	case "command_execution":
		id, _ := mapping["id"].(string)
		command, _ := mapping["command"].(string)
		status, _ := mapping["status"].(string)
		output, _ := mapping["aggregated_output"].(string)
		exitCode, hasExitCode := numericToInt(mapping["exit_code"])
		var exitCodePtr *int
		if hasExitCode {
			exitCodePtr = &exitCode
		}
		return CommandExecutionItem{
			ID:               id,
			Command:          command,
			Status:           status,
			AggregatedOutput: output,
			ExitCode:         exitCodePtr,
		}, nil
	case "file_change":
		id, _ := mapping["id"].(string)
		status, _ := mapping["status"].(string)
		changes := make([]FileUpdateChange, 0)
		for _, rawChange := range toAnySlice(mapping["changes"]) {
			changeMap, ok := toStringAnyMap(rawChange)
			if !ok {
				continue
			}
			path, _ := changeMap["path"].(string)
			kind, _ := changeMap["kind"].(string)
			changes = append(changes, FileUpdateChange{Path: path, Kind: kind})
		}
		return FileChangeItem{ID: id, Changes: changes, Status: status}, nil
	case "mcp_tool_call":
		id, _ := mapping["id"].(string)
		server, _ := mapping["server"].(string)
		tool, _ := mapping["tool"].(string)
		status, _ := mapping["status"].(string)
		var result *McpToolCallResult
		if rawResult, ok := toStringAnyMap(mapping["result"]); ok {
			result = &McpToolCallResult{
				Content:           toAnySlice(rawResult["content"]),
				StructuredContent: rawResult["structured_content"],
			}
		}
		var itemError *McpToolCallError
		if rawError, ok := toStringAnyMap(mapping["error"]); ok {
			message, _ := rawError["message"].(string)
			itemError = &McpToolCallError{Message: message}
		}
		return McpToolCallItem{
			ID:        id,
			Server:    server,
			Tool:      tool,
			Arguments: mapping["arguments"],
			Status:    status,
			Result:    result,
			Error:     itemError,
		}, nil
	case "agent_message":
		id, _ := mapping["id"].(string)
		text, _ := mapping["text"].(string)
		return AgentMessageItem{ID: id, Text: text}, nil
	case "reasoning":
		id, _ := mapping["id"].(string)
		text, _ := mapping["text"].(string)
		return ReasoningItem{ID: id, Text: text}, nil
	case "web_search":
		id, _ := mapping["id"].(string)
		query, _ := mapping["query"].(string)
		return WebSearchItem{ID: id, Query: query}, nil
	case "todo_list":
		id, _ := mapping["id"].(string)
		items := make([]TodoItem, 0)
		for _, rawTodo := range toAnySlice(mapping["items"]) {
			todoMap, ok := toStringAnyMap(rawTodo)
			if !ok {
				continue
			}
			text, _ := todoMap["text"].(string)
			completed, _ := todoMap["completed"].(bool)
			items = append(items, TodoItem{
				Text:      text,
				Completed: completed,
			})
		}
		return TodoListItem{ID: id, Items: items}, nil
	case "error":
		id, _ := mapping["id"].(string)
		message, _ := mapping["message"].(string)
		return ErrorItem{ID: id, Message: message}, nil
	default:
		var idPtr *string
		if id, ok := mapping["id"].(string); ok {
			idPtr = &id
		}
		return UnknownThreadItem{
			ItemType: itemType,
			Payload:  mapping,
			ID:       idPtr,
		}, nil
	}
}
