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
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	oairealtime "github.com/openai/openai-go/v3/realtime"
)

const (
	realtimeSessionType      = "realtime"
	transcriptionSessionType = "transcription"
)

// NormalizeSessionPayload normalizes user-provided session payloads into a realtime session
// create request object. Transcription-only payloads are ignored and return nil.
func NormalizeSessionPayload(session any) *oairealtime.RealtimeSessionCreateRequestParam {
	switch v := session.(type) {
	case nil:
		return nil
	case oairealtime.RealtimeSessionCreateRequestParam:
		normalized := v
		return &normalized
	case *oairealtime.RealtimeSessionCreateRequestParam:
		return v
	case oairealtime.RealtimeTranscriptionSessionCreateRequestParam:
		return nil
	case *oairealtime.RealtimeTranscriptionSessionCreateRequestParam:
		return nil
	}

	payload, ok := toStringAnyMap(session)
	if !ok {
		return nil
	}

	if isTranscriptionSessionPayload(payload) {
		return nil
	}

	if typeValue, ok := payload["type"].(string); ok {
		normalizedType := strings.ToLower(strings.TrimSpace(typeValue))
		if normalizedType != "" && normalizedType != realtimeSessionType {
			return nil
		}
	}

	rawPayload, err := json.Marshal(payload)
	if err != nil {
		return nil
	}

	var normalized oairealtime.RealtimeSessionCreateRequestParam
	if err := json.Unmarshal(rawPayload, &normalized); err != nil {
		return nil
	}

	return &normalized
}

// ExtractSessionAudioFormat returns the normalized output audio format configured on a session.
func ExtractSessionAudioFormat(
	session oairealtime.RealtimeSessionCreateRequestParam,
) *string {
	format := session.Audio.Output.Format
	if format.OfAudioPCM == nil && format.OfAudioPCMU == nil && format.OfAudioPCMA == nil &&
		format.GetType() == nil {
		return nil
	}

	normalized := NormalizeAudioFormat(format)
	if strings.TrimSpace(normalized) == "" {
		return nil
	}

	return &normalized
}

// NormalizeAudioFormat normalizes different audio format representations into a single string.
func NormalizeAudioFormat(format any) string {
	switch v := format.(type) {
	case oairealtime.RealtimeAudioFormatsAudioPCM, *oairealtime.RealtimeAudioFormatsAudioPCM,
		oairealtime.RealtimeAudioFormatsAudioPCMParam, *oairealtime.RealtimeAudioFormatsAudioPCMParam:
		return "pcm16"
	case oairealtime.RealtimeAudioFormatsAudioPCMU, *oairealtime.RealtimeAudioFormatsAudioPCMU,
		oairealtime.RealtimeAudioFormatsAudioPCMUParam, *oairealtime.RealtimeAudioFormatsAudioPCMUParam:
		return "g711_ulaw"
	case oairealtime.RealtimeAudioFormatsAudioPCMA, *oairealtime.RealtimeAudioFormatsAudioPCMA,
		oairealtime.RealtimeAudioFormatsAudioPCMAParam, *oairealtime.RealtimeAudioFormatsAudioPCMAParam:
		return "g711_alaw"
	case oairealtime.RealtimeAudioFormatsUnionParam:
		return normalizeAudioFormatUnionParam(v)
	case *oairealtime.RealtimeAudioFormatsUnionParam:
		if v == nil {
			return fmt.Sprint(v)
		}
		return normalizeAudioFormatUnionParam(*v)
	case oairealtime.RealtimeAudioFormatsUnion:
		if formatType := strings.TrimSpace(v.Type); formatType != "" {
			return formatType
		}
	case *oairealtime.RealtimeAudioFormatsUnion:
		if v == nil {
			return fmt.Sprint(v)
		}
		if formatType := strings.TrimSpace(v.Type); formatType != "" {
			return formatType
		}
	}

	if formatType, ok := readAudioFormatType(format); ok && strings.TrimSpace(formatType) != "" {
		return formatType
	}

	return fmt.Sprint(format)
}

// NormalizeTurnDetectionConfig converts known camelCase keys into snake_case while preserving
// existing snake_case values.
func NormalizeTurnDetectionConfig(config any) any {
	mapping, ok := toStringAnyMap(config)
	if !ok {
		return config
	}

	keyMap := map[string]string{
		"createResponse":    "create_response",
		"interruptResponse": "interrupt_response",
		"prefixPaddingMs":   "prefix_padding_ms",
		"silenceDurationMs": "silence_duration_ms",
		"idleTimeoutMs":     "idle_timeout_ms",
		"modelVersion":      "model_version",
	}

	for camelKey, snakeKey := range keyMap {
		if camelValue, exists := mapping[camelKey]; exists {
			if _, hasSnakeValue := mapping[snakeKey]; !hasSnakeValue {
				mapping[snakeKey] = camelValue
			}
			delete(mapping, camelKey)
		}
	}

	return mapping
}

func normalizeAudioFormatUnionParam(format oairealtime.RealtimeAudioFormatsUnionParam) string {
	switch {
	case format.OfAudioPCM != nil:
		return "pcm16"
	case format.OfAudioPCMU != nil:
		return "g711_ulaw"
	case format.OfAudioPCMA != nil:
		return "g711_alaw"
	}

	if formatType := format.GetType(); formatType != nil && strings.TrimSpace(*formatType) != "" {
		return *formatType
	}

	return fmt.Sprint(format)
}

func readAudioFormatType(format any) (string, bool) {
	if format == nil {
		return "", false
	}

	if formatType, ok := format.(string); ok {
		return formatType, true
	}

	if typeValue, ok := mapValue(format, "type"); ok {
		if formatType, ok := typeValue.(string); ok {
			return formatType, true
		}
	}

	value := reflect.ValueOf(format)
	if !value.IsValid() {
		return "", false
	}

	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return "", false
		}
		value = value.Elem()
	}

	if value.Kind() == reflect.Struct {
		typeField := value.FieldByName("Type")
		if typeField.IsValid() && typeField.Kind() == reflect.String {
			return typeField.String(), true
		}
	}

	return "", false
}

func isTranscriptionSessionPayload(payload map[string]any) bool {
	typeValue, ok := payload["type"].(string)
	if !ok {
		return false
	}

	return strings.ToLower(strings.TrimSpace(typeValue)) == transcriptionSessionType
}

func toStringAnyMap(input any) (map[string]any, bool) {
	if input == nil {
		return nil, false
	}

	value := reflect.ValueOf(input)
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return nil, false
		}
		value = value.Elem()
	}

	if value.Kind() != reflect.Map {
		return nil, false
	}

	if value.Type().Key().Kind() != reflect.String {
		return nil, false
	}

	out := make(map[string]any, value.Len())
	iter := value.MapRange()
	for iter.Next() {
		out[iter.Key().String()] = iter.Value().Interface()
	}

	return out, true
}
