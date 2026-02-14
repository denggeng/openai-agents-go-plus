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

import "github.com/nlpodyssey/openai-agents-go/agents"

// ThreadEvent is one JSONL event emitted by the Codex CLI.
type ThreadEvent interface {
	EventType() string
}

type ThreadStartedEvent struct {
	ThreadID string
}

func (ThreadStartedEvent) EventType() string { return "thread.started" }

type TurnStartedEvent struct{}

func (TurnStartedEvent) EventType() string { return "turn.started" }

type Usage struct {
	InputTokens       int
	CachedInputTokens int
	OutputTokens      int
}

type TurnCompletedEvent struct {
	Usage *Usage
}

func (TurnCompletedEvent) EventType() string { return "turn.completed" }

type ThreadError struct {
	Message string
}

type TurnFailedEvent struct {
	Error ThreadError
}

func (TurnFailedEvent) EventType() string { return "turn.failed" }

type ItemStartedEvent struct {
	Item ThreadItem
}

func (ItemStartedEvent) EventType() string { return "item.started" }

type ItemUpdatedEvent struct {
	Item ThreadItem
}

func (ItemUpdatedEvent) EventType() string { return "item.updated" }

type ItemCompletedEvent struct {
	Item ThreadItem
}

func (ItemCompletedEvent) EventType() string { return "item.completed" }

type ThreadErrorEvent struct {
	Message string
}

func (ThreadErrorEvent) EventType() string { return "error" }

type UnknownThreadEvent struct {
	Type    string
	Payload map[string]any
}

func (event UnknownThreadEvent) EventType() string { return event.Type }

func coerceUsage(raw any) (*Usage, error) {
	if raw == nil {
		return nil, nil
	}
	switch typed := raw.(type) {
	case Usage:
		value := typed
		return &value, nil
	case *Usage:
		if typed == nil {
			return nil, nil
		}
		value := *typed
		return &value, nil
	}

	mapping, ok := toStringAnyMap(raw)
	if !ok {
		return nil, agents.NewUserError("usage payload must be a mapping")
	}
	inputTokens, _ := numericToInt(mapping["input_tokens"])
	cachedInputTokens, _ := numericToInt(mapping["cached_input_tokens"])
	outputTokens, _ := numericToInt(mapping["output_tokens"])
	return &Usage{
		InputTokens:       inputTokens,
		CachedInputTokens: cachedInputTokens,
		OutputTokens:      outputTokens,
	}, nil
}

func coerceThreadError(raw any) (ThreadError, error) {
	switch typed := raw.(type) {
	case ThreadError:
		return typed, nil
	case *ThreadError:
		if typed == nil {
			return ThreadError{}, nil
		}
		return *typed, nil
	}
	mapping, ok := toStringAnyMap(raw)
	if !ok {
		return ThreadError{}, agents.NewUserError("thread error payload must be a mapping")
	}
	message, _ := mapping["message"].(string)
	return ThreadError{Message: message}, nil
}

// CoerceThreadEvent parses a typed or map payload into one event object.
func CoerceThreadEvent(raw any) (ThreadEvent, error) {
	switch typed := raw.(type) {
	case ThreadStartedEvent:
		return typed, nil
	case *ThreadStartedEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case TurnStartedEvent:
		return typed, nil
	case *TurnStartedEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case TurnCompletedEvent:
		return typed, nil
	case *TurnCompletedEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case TurnFailedEvent:
		return typed, nil
	case *TurnFailedEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case ItemStartedEvent:
		return typed, nil
	case *ItemStartedEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case ItemUpdatedEvent:
		return typed, nil
	case *ItemUpdatedEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case ItemCompletedEvent:
		return typed, nil
	case *ItemCompletedEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case ThreadErrorEvent:
		return typed, nil
	case *ThreadErrorEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	case UnknownThreadEvent:
		return typed, nil
	case *UnknownThreadEvent:
		if typed == nil {
			return nil, agents.NewUserError("thread event payload must be a mapping")
		}
		return *typed, nil
	}

	mapping, ok := toStringAnyMap(raw)
	if !ok {
		return nil, agents.NewUserError("thread event payload must be a mapping")
	}
	eventType, _ := mapping["type"].(string)
	switch eventType {
	case "thread.started":
		threadID, _ := mapping["thread_id"].(string)
		return ThreadStartedEvent{ThreadID: threadID}, nil
	case "turn.started":
		return TurnStartedEvent{}, nil
	case "turn.completed":
		usage, err := coerceUsage(mapping["usage"])
		if err != nil {
			return nil, err
		}
		return TurnCompletedEvent{Usage: usage}, nil
	case "turn.failed":
		threadError, err := coerceThreadError(mapping["error"])
		if err != nil {
			return nil, err
		}
		return TurnFailedEvent{Error: threadError}, nil
	case "item.started":
		item, err := coerceThreadEventItem(mapping)
		if err != nil {
			return nil, err
		}
		return ItemStartedEvent{Item: item}, nil
	case "item.updated":
		item, err := coerceThreadEventItem(mapping)
		if err != nil {
			return nil, err
		}
		return ItemUpdatedEvent{Item: item}, nil
	case "item.completed":
		item, err := coerceThreadEventItem(mapping)
		if err != nil {
			return nil, err
		}
		return ItemCompletedEvent{Item: item}, nil
	case "error":
		message, _ := mapping["message"].(string)
		return ThreadErrorEvent{Message: message}, nil
	default:
		return UnknownThreadEvent{
			Type:    eventTypeOrUnknown(eventType),
			Payload: mapping,
		}, nil
	}
}

func coerceThreadEventItem(mapping map[string]any) (ThreadItem, error) {
	if rawItem, ok := mapping["item"]; ok && rawItem != nil {
		return CoerceThreadItem(rawItem)
	}
	return CoerceThreadItem(map[string]any{"type": "unknown"})
}

func eventTypeOrUnknown(value string) string {
	if value == "" {
		return "unknown"
	}
	return value
}
