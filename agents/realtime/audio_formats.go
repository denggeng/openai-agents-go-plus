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
	"reflect"
	"strings"

	oairealtime "github.com/openai/openai-go/v3/realtime"
)

const defaultPCMRate int64 = 24000

// ToRealtimeAudioFormat normalizes user-provided audio format values into
// OpenAI Realtime audio format params.
func ToRealtimeAudioFormat(inputAudioFormat any) *oairealtime.RealtimeAudioFormatsUnionParam {
	switch v := inputAudioFormat.(type) {
	case nil:
		return nil
	case string:
		return realtimeAudioFormatFromString(v)
	case oairealtime.RealtimeAudioFormatsUnionParam:
		format := v
		return &format
	case *oairealtime.RealtimeAudioFormatsUnionParam:
		return v
	case oairealtime.RealtimeAudioFormatsAudioPCMParam:
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCM: &v,
		}
	case *oairealtime.RealtimeAudioFormatsAudioPCMParam:
		if v == nil {
			return nil
		}
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCM: v,
		}
	case oairealtime.RealtimeAudioFormatsAudioPCMUParam:
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMU: &v,
		}
	case *oairealtime.RealtimeAudioFormatsAudioPCMUParam:
		if v == nil {
			return nil
		}
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMU: v,
		}
	case oairealtime.RealtimeAudioFormatsAudioPCMAParam:
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMA: &v,
		}
	case *oairealtime.RealtimeAudioFormatsAudioPCMAParam:
		if v == nil {
			return nil
		}
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMA: v,
		}
	default:
		return realtimeAudioFormatFromMapping(v)
	}
}

func realtimeAudioFormatFromString(input string) *oairealtime.RealtimeAudioFormatsUnionParam {
	switch strings.ToLower(strings.TrimSpace(input)) {
	case "pcm16", "audio/pcm", "pcm":
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCM: &oairealtime.RealtimeAudioFormatsAudioPCMParam{
				Type: "audio/pcm",
				Rate: defaultPCMRate,
			},
		}
	case "g711_ulaw", "audio/pcmu", "pcmu":
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMU: &oairealtime.RealtimeAudioFormatsAudioPCMUParam{
				Type: "audio/pcmu",
			},
		}
	case "g711_alaw", "audio/pcma", "pcma":
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMA: &oairealtime.RealtimeAudioFormatsAudioPCMAParam{
				Type: "audio/pcma",
			},
		}
	default:
		return nil
	}
}

func realtimeAudioFormatFromMapping(input any) *oairealtime.RealtimeAudioFormatsUnionParam {
	typeValue, ok := mapValue(input, "type")
	if !ok {
		return nil
	}

	formatType, ok := typeValue.(string)
	if !ok {
		return nil
	}

	switch strings.ToLower(strings.TrimSpace(formatType)) {
	case "audio/pcm":
		rate, _ := mapValue(input, "rate")
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCM: &oairealtime.RealtimeAudioFormatsAudioPCMParam{
				Type: "audio/pcm",
				Rate: normalizedPCMRate(rate),
			},
		}
	case "audio/pcmu":
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMU: &oairealtime.RealtimeAudioFormatsAudioPCMUParam{
				Type: "audio/pcmu",
			},
		}
	case "audio/pcma":
		return &oairealtime.RealtimeAudioFormatsUnionParam{
			OfAudioPCMA: &oairealtime.RealtimeAudioFormatsAudioPCMAParam{
				Type: "audio/pcma",
			},
		}
	default:
		return nil
	}
}

func normalizedPCMRate(raw any) int64 {
	if raw == nil {
		return defaultPCMRate
	}
	value, ok := numericToInt64(raw)
	if !ok {
		return defaultPCMRate
	}
	if value == defaultPCMRate {
		return defaultPCMRate
	}
	return defaultPCMRate
}

func mapValue(input any, key string) (any, bool) {
	if input == nil {
		return nil, false
	}
	rv := reflect.ValueOf(input)
	if rv.Kind() != reflect.Map {
		return nil, false
	}
	if rv.Type().Key().Kind() != reflect.String {
		return nil, false
	}
	value := rv.MapIndex(reflect.ValueOf(key))
	if !value.IsValid() {
		return nil, false
	}
	return value.Interface(), true
}

func numericToInt64(value any) (int64, bool) {
	switch v := value.(type) {
	case int:
		return int64(v), true
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	case uint:
		return int64(v), true
	case uint8:
		return int64(v), true
	case uint16:
		return int64(v), true
	case uint32:
		return int64(v), true
	case uint64:
		if v > uint64(^uint64(0)>>1) {
			return 0, false
		}
		return int64(v), true
	case float32:
		return int64(v), true
	case float64:
		return int64(v), true
	default:
		return 0, false
	}
}
