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

package realtime

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func stringPtr(value string) *string {
	return &value
}

func TestTryConvertRawMessageValidSessionUpdate(t *testing.T) {
	rawMessage := RealtimeModelSendRawMessage{
		Message: RealtimeModelRawClientMessage{
			Type: "session.update",
			OtherData: map[string]any{
				"session": map[string]any{
					"model":      "gpt-realtime",
					"type":       "realtime",
					"modalities": []string{"text", "audio"},
					"voice":      "ash",
				},
			},
		},
	}

	result := TryConvertRawMessage(rawMessage)
	require.NotNil(t, result)
	assert.Equal(t, "session.update", result["type"])
}

func TestTryConvertRawMessageValidResponseCreate(t *testing.T) {
	rawMessage := RealtimeModelSendRawMessage{
		Message: RealtimeModelRawClientMessage{
			Type:      "response.create",
			OtherData: map[string]any{},
		},
	}

	result := TryConvertRawMessage(rawMessage)
	require.NotNil(t, result)
	assert.Equal(t, "response.create", result["type"])
}

func TestTryConvertRawMessageValidConversationItemRetrieve(t *testing.T) {
	rawMessage := RealtimeModelSendRawMessage{
		Message: RealtimeModelRawClientMessage{
			Type: "conversation.item.retrieve",
			OtherData: map[string]any{
				"item_id": "item_123",
			},
		},
	}

	result := TryConvertRawMessage(rawMessage)
	require.NotNil(t, result)
	assert.Equal(t, "conversation.item.retrieve", result["type"])
	assert.Equal(t, "item_123", result["item_id"])
}

func TestTryConvertRawMessageInvalidType(t *testing.T) {
	rawMessage := RealtimeModelSendRawMessage{
		Message: RealtimeModelRawClientMessage{
			Type:      "invalid.message.type",
			OtherData: map[string]any{},
		},
	}

	assert.Nil(t, TryConvertRawMessage(rawMessage))
}

func TestTryConvertRawMessageMalformedData(t *testing.T) {
	rawMessage := RealtimeModelSendRawMessage{
		Message: RealtimeModelRawClientMessage{
			Type: "session.update",
			OtherData: map[string]any{
				"session": "invalid_session_data",
			},
		},
	}

	assert.Nil(t, TryConvertRawMessage(rawMessage))
}

func TestTryConvertRawMessageMissingType(t *testing.T) {
	rawMessage := RealtimeModelSendRawMessage{
		Message: RealtimeModelRawClientMessage{
			Type:      "missing.type.test",
			OtherData: map[string]any{"some": "data"},
		},
	}

	assert.Nil(t, TryConvertRawMessage(rawMessage))
}

func TestConvertTracingConfigNone(t *testing.T) {
	assert.Nil(t, ConvertTracingConfig(nil))
}

func TestConvertTracingConfigAuto(t *testing.T) {
	result := ConvertTracingConfig("auto")
	require.NotNil(t, result)
	assert.Nil(t, result.OfTracingConfiguration)

	serialized, err := json.Marshal(result)
	require.NoError(t, err)
	assert.Contains(t, string(serialized), "auto")
}

func TestConvertTracingConfigStructFull(t *testing.T) {
	tracingConfig := RealtimeModelTracingConfig{
		GroupID:      stringPtr("test-group"),
		Metadata:     map[string]any{"env": "test"},
		WorkflowName: stringPtr("test-workflow"),
	}

	result := ConvertTracingConfig(tracingConfig)
	require.NotNil(t, result)
	require.NotNil(t, result.OfTracingConfiguration)

	assert.True(t, result.OfTracingConfiguration.GroupID.Valid())
	assert.Equal(t, "test-group", result.OfTracingConfiguration.GroupID.Value)
	assert.Equal(t, map[string]any{"env": "test"}, result.OfTracingConfiguration.Metadata)
	assert.True(t, result.OfTracingConfiguration.WorkflowName.Valid())
	assert.Equal(t, "test-workflow", result.OfTracingConfiguration.WorkflowName.Value)
}

func TestConvertTracingConfigStructPartial(t *testing.T) {
	tracingConfig := RealtimeModelTracingConfig{
		GroupID: stringPtr("test-group"),
	}

	result := ConvertTracingConfig(tracingConfig)
	require.NotNil(t, result)
	require.NotNil(t, result.OfTracingConfiguration)

	assert.True(t, result.OfTracingConfiguration.GroupID.Valid())
	assert.Equal(t, "test-group", result.OfTracingConfiguration.GroupID.Value)
	assert.Nil(t, result.OfTracingConfiguration.Metadata)
	assert.False(t, result.OfTracingConfiguration.WorkflowName.Valid())
}

func TestConvertTracingConfigEmptyStruct(t *testing.T) {
	tracingConfig := RealtimeModelTracingConfig{}

	result := ConvertTracingConfig(tracingConfig)
	require.NotNil(t, result)
	require.NotNil(t, result.OfTracingConfiguration)
	assert.False(t, result.OfTracingConfiguration.GroupID.Valid())
	assert.Nil(t, result.OfTracingConfiguration.Metadata)
	assert.False(t, result.OfTracingConfiguration.WorkflowName.Valid())
}

func TestConvertTracingConfigMap(t *testing.T) {
	result := ConvertTracingConfig(map[string]any{
		"group_id":      "test-group",
		"workflow_name": "test-workflow",
		"metadata": map[string]any{
			"env": "test",
		},
	})
	require.NotNil(t, result)
	require.NotNil(t, result.OfTracingConfiguration)

	assert.True(t, result.OfTracingConfiguration.GroupID.Valid())
	assert.Equal(t, "test-group", result.OfTracingConfiguration.GroupID.Value)
	assert.True(t, result.OfTracingConfiguration.WorkflowName.Valid())
	assert.Equal(t, "test-workflow", result.OfTracingConfiguration.WorkflowName.Value)
	assert.Equal(t, map[string]any{"env": "test"}, result.OfTracingConfiguration.Metadata)
}

func TestConvertTracingConfigMapPartial(t *testing.T) {
	result := ConvertTracingConfig(map[string]any{
		"group_id": "test-group",
	})
	require.NotNil(t, result)
	require.NotNil(t, result.OfTracingConfiguration)

	assert.True(t, result.OfTracingConfiguration.GroupID.Valid())
	assert.Equal(t, "test-group", result.OfTracingConfiguration.GroupID.Value)
	assert.Nil(t, result.OfTracingConfiguration.Metadata)
	assert.False(t, result.OfTracingConfiguration.WorkflowName.Valid())
}

func TestConvertTracingConfigMapEmpty(t *testing.T) {
	result := ConvertTracingConfig(map[string]any{})
	require.NotNil(t, result)
	require.NotNil(t, result.OfTracingConfiguration)
	assert.False(t, result.OfTracingConfiguration.GroupID.Valid())
	assert.Nil(t, result.OfTracingConfiguration.Metadata)
	assert.False(t, result.OfTracingConfiguration.WorkflowName.Valid())
}

func TestConvertUserInputToConversationItemString(t *testing.T) {
	event := RealtimeModelSendUserInput{UserInput: "Hello, world!"}

	result := ConvertUserInputToConversationItem(event)
	assert.Equal(t, "message", result["type"])
	assert.Equal(t, "user", result["role"])

	content, ok := result["content"].([]map[string]any)
	require.True(t, ok)
	require.Len(t, content, 1)
	assert.Equal(t, "input_text", content[0]["type"])
	assert.Equal(t, "Hello, world!", content[0]["text"])
}

func TestConvertUserInputToConversationItemDict(t *testing.T) {
	userInputDict := map[string]any{
		"type": "message",
		"role": "user",
		"content": []any{
			map[string]any{"type": "input_text", "text": "Hello"},
			map[string]any{"type": "input_text", "text": "World"},
		},
	}
	event := RealtimeModelSendUserInput{UserInput: userInputDict}

	result := ConvertUserInputToConversationItem(event)
	assert.Equal(t, "message", result["type"])
	assert.Equal(t, "user", result["role"])

	content, ok := result["content"].([]map[string]any)
	require.True(t, ok)
	require.Len(t, content, 2)
	assert.Equal(t, "input_text", content[0]["type"])
	assert.Equal(t, "Hello", content[0]["text"])
	assert.Equal(t, "input_text", content[1]["type"])
	assert.Equal(t, "World", content[1]["text"])
}

func TestConvertUserInputToConversationItemDictSkipsUnknownParts(t *testing.T) {
	userInputDict := map[string]any{
		"type": "message",
		"role": "user",
		"content": []any{
			map[string]any{"type": "input_text", "text": "Hello"},
			map[string]any{
				"type":      "input_image",
				"image_url": "http://x/y.png",
				"detail":    "auto",
			},
			map[string]any{"type": "bogus", "x": 1},
		},
	}
	event := RealtimeModelSendUserInput{UserInput: userInputDict}

	result := ConvertUserInputToConversationItem(event)
	assert.Equal(t, "message", result["type"])
	assert.Equal(t, "user", result["role"])

	content, ok := result["content"].([]map[string]any)
	require.True(t, ok)
	require.Len(t, content, 2)
	assert.Equal(t, "input_text", content[0]["type"])
	assert.Equal(t, "Hello", content[0]["text"])
	assert.Equal(t, "input_image", content[1]["type"])
	assert.Equal(t, "http://x/y.png", content[1]["image_url"])
	assert.Equal(t, "auto", content[1]["detail"])
}

func TestConvertUserInputToConversationItemDictEmptyContent(t *testing.T) {
	userInputDict := map[string]any{
		"type":    "message",
		"role":    "user",
		"content": []any{},
	}
	event := RealtimeModelSendUserInput{UserInput: userInputDict}

	result := ConvertUserInputToConversationItem(event)
	content, ok := result["content"].([]map[string]any)
	require.True(t, ok)
	assert.Len(t, content, 0)
}

func TestConvertUserInputToConversationItemDictContentAsTypedMapSlice(t *testing.T) {
	userInputDict := map[string]any{
		"type": "message",
		"role": "user",
		"content": []map[string]any{
			{"type": "input_text", "text": "Typed"},
		},
	}
	event := RealtimeModelSendUserInput{UserInput: userInputDict}

	result := ConvertUserInputToConversationItem(event)
	content, ok := result["content"].([]map[string]any)
	require.True(t, ok)
	require.Len(t, content, 1)
	assert.Equal(t, "Typed", content[0]["text"])
}

func TestConvertUserInputToItemCreate(t *testing.T) {
	event := RealtimeModelSendUserInput{UserInput: "Test message"}

	result := ConvertUserInputToItemCreate(event)
	assert.Equal(t, "conversation.item.create", result["type"])

	item, ok := result["item"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "message", item["type"])
	assert.Equal(t, "user", item["role"])
}

func TestConvertAudioToInputAudioBufferAppend(t *testing.T) {
	audioData := []byte("test audio data")
	event := RealtimeModelSendAudio{Audio: audioData, Commit: false}

	result := ConvertAudioToInputAudioBufferAppend(event)
	assert.Equal(t, "input_audio_buffer.append", result["type"])

	expectedB64 := base64.StdEncoding.EncodeToString(audioData)
	assert.Equal(t, expectedB64, result["audio"])
}

func TestConvertAudioToInputAudioBufferAppendEmpty(t *testing.T) {
	event := RealtimeModelSendAudio{Audio: []byte{}, Commit: true}
	result := ConvertAudioToInputAudioBufferAppend(event)
	assert.Equal(t, "input_audio_buffer.append", result["type"])
	assert.Equal(t, "", result["audio"])
}

func TestConvertAudioToInputAudioBufferAppendLargeData(t *testing.T) {
	audioData := make([]byte, 10000)
	for i := range audioData {
		audioData[i] = 'x'
	}
	event := RealtimeModelSendAudio{Audio: audioData, Commit: false}

	result := ConvertAudioToInputAudioBufferAppend(event)
	assert.Equal(t, "input_audio_buffer.append", result["type"])

	audioBase64, ok := result["audio"].(string)
	require.True(t, ok)
	decoded, err := base64.StdEncoding.DecodeString(audioBase64)
	require.NoError(t, err)
	assert.Equal(t, audioData, decoded)
}

func TestConvertToolOutput(t *testing.T) {
	toolCall := RealtimeModelToolCallEvent{CallID: "call_123"}
	event := RealtimeModelSendToolOutput{
		ToolCall:      toolCall,
		Output:        "Function executed successfully",
		StartResponse: false,
	}

	result := ConvertToolOutput(event)
	assert.Equal(t, "conversation.item.create", result["type"])

	item, ok := result["item"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function_call_output", item["type"])
	assert.Equal(t, "Function executed successfully", item["output"])
	assert.Equal(t, "call_123", item["call_id"])
}

func TestConvertToolOutputEmptyOutput(t *testing.T) {
	toolCall := RealtimeModelToolCallEvent{CallID: "call_456"}
	event := RealtimeModelSendToolOutput{
		ToolCall:      toolCall,
		Output:        "",
		StartResponse: true,
	}

	result := ConvertToolOutput(event)
	item, ok := result["item"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "", item["output"])
	assert.Equal(t, "call_456", item["call_id"])
}

func TestConvertToolOutputMissingCallID(t *testing.T) {
	event := RealtimeModelSendToolOutput{
		ToolCall: RealtimeModelToolCallEvent{CallID: ""},
		Output:   "x",
	}
	assert.Nil(t, ConvertToolOutput(event))
}

func TestConvertInterrupt(t *testing.T) {
	result := ConvertInterrupt("item_789", 2, 1500)

	assert.Equal(t, "conversation.item.truncate", result["type"])
	assert.Equal(t, "item_789", result["item_id"])
	assert.Equal(t, 2, result["content_index"])
	assert.Equal(t, 1500, result["audio_end_ms"])
}

func TestConvertInterruptZeroTime(t *testing.T) {
	result := ConvertInterrupt("item_1", 0, 0)
	assert.Equal(t, "conversation.item.truncate", result["type"])
	assert.Equal(t, "item_1", result["item_id"])
	assert.Equal(t, 0, result["content_index"])
	assert.Equal(t, 0, result["audio_end_ms"])
}

func TestConvertInterruptLargeValues(t *testing.T) {
	result := ConvertInterrupt("item_xyz", 99, 999999)
	assert.Equal(t, "conversation.item.truncate", result["type"])
	assert.Equal(t, "item_xyz", result["item_id"])
	assert.Equal(t, 99, result["content_index"])
	assert.Equal(t, 999999, result["audio_end_ms"])
}

func TestConvertInterruptEmptyItemID(t *testing.T) {
	result := ConvertInterrupt("", 1, 100)
	assert.Equal(t, "conversation.item.truncate", result["type"])
	assert.Equal(t, "", result["item_id"])
	assert.Equal(t, 1, result["content_index"])
	assert.Equal(t, 100, result["audio_end_ms"])
}
