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

package agents

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

func convertToolOutput(output any) responses.ResponseInputItemFunctionCallOutputOutputUnionParam {
	if list, ok := convertToolOutputList(output); ok {
		return responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfResponseFunctionCallOutputItemArray: list,
		}
	}
	if single, ok := convertSingleToolOutput(output); ok {
		return responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfResponseFunctionCallOutputItemArray: responses.ResponseFunctionCallOutputItemListParam{single},
		}
	}
	return responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
		OfString: param.NewOpt(stringifyToolOutput(output)),
	}
}

func convertToolOutputList(output any) (responses.ResponseFunctionCallOutputItemListParam, bool) {
	if output == nil {
		return nil, false
	}
	if _, ok := output.([]byte); ok {
		return nil, false
	}
	rv := reflect.ValueOf(output)
	if !rv.IsValid() {
		return nil, false
	}
	if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
		return nil, false
	}
	if rv.Type().Elem().Kind() == reflect.Uint8 {
		return nil, false
	}
	items := make(responses.ResponseFunctionCallOutputItemListParam, 0, rv.Len())
	for i := 0; i < rv.Len(); i++ {
		item, ok := convertSingleToolOutput(rv.Index(i).Interface())
		if !ok {
			return nil, false
		}
		items = append(items, item)
	}
	return items, true
}

func convertSingleToolOutput(output any) (responses.ResponseFunctionCallOutputItemUnionParam, bool) {
	switch v := output.(type) {
	case responses.ResponseFunctionCallOutputItemUnionParam:
		return v, true
	case *responses.ResponseFunctionCallOutputItemUnionParam:
		if v == nil {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		return *v, true
	case responses.ResponseInputTextContentParam:
		return responses.ResponseFunctionCallOutputItemUnionParam{OfInputText: &v}, true
	case *responses.ResponseInputTextContentParam:
		if v == nil {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		return responses.ResponseFunctionCallOutputItemUnionParam{OfInputText: v}, true
	case responses.ResponseInputImageContentParam:
		return responses.ResponseFunctionCallOutputItemUnionParam{OfInputImage: &v}, true
	case *responses.ResponseInputImageContentParam:
		if v == nil {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		return responses.ResponseFunctionCallOutputItemUnionParam{OfInputImage: v}, true
	case responses.ResponseInputFileContentParam:
		return responses.ResponseFunctionCallOutputItemUnionParam{OfInputFile: &v}, true
	case *responses.ResponseInputFileContentParam:
		if v == nil {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		return responses.ResponseFunctionCallOutputItemUnionParam{OfInputFile: v}, true
	case ToolOutputText:
		return toolOutputTextToParam(v), true
	case *ToolOutputText:
		if v == nil {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		return toolOutputTextToParam(*v), true
	case ToolOutputImage:
		item, ok := toolOutputImageToParam(v)
		return item, ok
	case *ToolOutputImage:
		if v == nil {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		item, ok := toolOutputImageToParam(*v)
		return item, ok
	case ToolOutputFileContent:
		item, ok := toolOutputFileContentToParam(v)
		return item, ok
	case *ToolOutputFileContent:
		if v == nil {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		item, ok := toolOutputFileContentToParam(*v)
		return item, ok
	case map[string]any:
		return toolOutputMapToParam(v)
	case map[string]string:
		converted := make(map[string]any, len(v))
		for key, value := range v {
			converted[key] = value
		}
		return toolOutputMapToParam(converted)
	default:
		return responses.ResponseFunctionCallOutputItemUnionParam{}, false
	}
}

func toolOutputTextToParam(output ToolOutputText) responses.ResponseFunctionCallOutputItemUnionParam {
	item := responses.ResponseInputTextContentParam{
		Text: output.Text,
		Type: constant.ValueOf[constant.InputText](),
	}
	return responses.ResponseFunctionCallOutputItemUnionParam{OfInputText: &item}
}

func toolOutputImageToParam(output ToolOutputImage) (responses.ResponseFunctionCallOutputItemUnionParam, bool) {
	if output.ImageURL == "" && output.FileID == "" {
		return responses.ResponseFunctionCallOutputItemUnionParam{}, false
	}
	item := responses.ResponseInputImageContentParam{
		Type: constant.ValueOf[constant.InputImage](),
	}
	if output.ImageURL != "" {
		item.ImageURL = param.NewOpt(output.ImageURL)
	}
	if output.FileID != "" {
		item.FileID = param.NewOpt(output.FileID)
	}
	if output.Detail != "" {
		detail, ok := parseImageDetail(output.Detail)
		if !ok {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		item.Detail = detail
	}
	return responses.ResponseFunctionCallOutputItemUnionParam{OfInputImage: &item}, true
}

func toolOutputFileContentToParam(output ToolOutputFileContent) (responses.ResponseFunctionCallOutputItemUnionParam, bool) {
	if output.FileData == "" && output.FileURL == "" && output.FileID == "" {
		return responses.ResponseFunctionCallOutputItemUnionParam{}, false
	}
	item := responses.ResponseInputFileContentParam{
		Type: constant.ValueOf[constant.InputFile](),
	}
	if output.FileData != "" {
		item.FileData = param.NewOpt(output.FileData)
	}
	if output.FileURL != "" {
		item.FileURL = param.NewOpt(output.FileURL)
	}
	if output.FileID != "" {
		item.FileID = param.NewOpt(output.FileID)
	}
	if output.Filename != "" {
		item.Filename = param.NewOpt(output.Filename)
	}
	return responses.ResponseFunctionCallOutputItemUnionParam{OfInputFile: &item}, true
}

func toolOutputMapToParam(values map[string]any) (responses.ResponseFunctionCallOutputItemUnionParam, bool) {
	typeValue, ok := getStringField(values, "type")
	if !ok || typeValue == "" {
		return responses.ResponseFunctionCallOutputItemUnionParam{}, false
	}

	switch strings.ToLower(typeValue) {
	case "text":
		text, ok := getStringField(values, "text")
		if !ok {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		return toolOutputTextToParam(ToolOutputText{Text: text}), true
	case "image":
		imageURL, hasImageURL := getStringField(values, "image_url")
		fileID, hasFileID := getStringField(values, "file_id")
		if !hasImageURL && !hasFileID {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		output := ToolOutputImage{
			ImageURL: imageURL,
			FileID:   fileID,
		}
		if detail, ok := getStringField(values, "detail"); ok {
			output.Detail = detail
		}
		return toolOutputImageToParam(output)
	case "file":
		fileData, hasFileData := getStringField(values, "file_data")
		fileURL, hasFileURL := getStringField(values, "file_url")
		fileID, hasFileID := getStringField(values, "file_id")
		if !hasFileData && !hasFileURL && !hasFileID {
			return responses.ResponseFunctionCallOutputItemUnionParam{}, false
		}
		output := ToolOutputFileContent{
			FileData: fileData,
			FileURL:  fileURL,
			FileID:   fileID,
		}
		if filename, ok := getStringField(values, "filename"); ok {
			output.Filename = filename
		}
		return toolOutputFileContentToParam(output)
	default:
		return responses.ResponseFunctionCallOutputItemUnionParam{}, false
	}
}

func getStringField(values map[string]any, key string) (string, bool) {
	raw, ok := values[key]
	if !ok {
		return "", false
	}
	value, ok := raw.(string)
	if !ok {
		return "", false
	}
	return value, true
}

func parseImageDetail(detail string) (responses.ResponseInputImageContentDetail, bool) {
	switch strings.ToLower(strings.TrimSpace(detail)) {
	case string(responses.ResponseInputImageContentDetailLow):
		return responses.ResponseInputImageContentDetailLow, true
	case string(responses.ResponseInputImageContentDetailHigh):
		return responses.ResponseInputImageContentDetailHigh, true
	case string(responses.ResponseInputImageContentDetailAuto):
		return responses.ResponseInputImageContentDetailAuto, true
	default:
		return "", false
	}
}

func stringifyToolOutput(output any) string {
	switch v := output.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		if b, err := json.Marshal(v); err == nil {
			return string(b)
		}
		return fmt.Sprintf("%v", output)
	}
}
