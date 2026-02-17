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

package agents_test

import (
	"encoding/json"
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolCallOutputItemTextModel(t *testing.T) {
	call := makeToolCall()
	out := agents.ToolOutputText{Text: "hello"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputText)
	assert.Equal(t, "hello", items[0].OfInputText.Text)
}

func TestToolCallOutputItemImageModel(t *testing.T) {
	call := makeToolCall()
	out := agents.ToolOutputImage{ImageURL: "data:image/png;base64,AAAA"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputImage)
	assert.True(t, items[0].OfInputImage.ImageURL.Valid())
	assert.Equal(t, "data:image/png;base64,AAAA", items[0].OfInputImage.ImageURL.Value)
}

func TestToolCallOutputItemFileModel(t *testing.T) {
	call := makeToolCall()
	out := agents.ToolOutputFileContent{FileData: "ZmFrZS1kYXRh", Filename: "foo.txt"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputFile)
	assert.True(t, items[0].OfInputFile.FileData.Valid())
	assert.Equal(t, "ZmFrZS1kYXRh", items[0].OfInputFile.FileData.Value)
}

func TestToolCallOutputItemMixedList(t *testing.T) {
	call := makeToolCall()
	outputs := []any{
		agents.ToolOutputText{Text: "a"},
		agents.ToolOutputImage{ImageURL: "http://example/img.png"},
		agents.ToolOutputFileContent{FileData: "ZmlsZS1kYXRh"},
	}

	payload := agents.ItemHelpers().ToolCallOutputItem(call, outputs)
	items := requireOutputList(t, payload)
	require.Len(t, items, 3)

	require.NotNil(t, items[0].OfInputText)
	assert.Equal(t, "a", items[0].OfInputText.Text)
	require.NotNil(t, items[1].OfInputImage)
	assert.Equal(t, "http://example/img.png", items[1].OfInputImage.ImageURL.Value)
	require.NotNil(t, items[2].OfInputFile)
	assert.Equal(t, "ZmlsZS1kYXRh", items[2].OfInputFile.FileData.Value)
}

func TestToolCallOutputItemImageForwardsFileIDAndDetail(t *testing.T) {
	call := makeToolCall()
	out := agents.ToolOutputImage{FileID: "file_123", Detail: "high"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputImage)
	assert.True(t, items[0].OfInputImage.FileID.Valid())
	assert.Equal(t, "file_123", items[0].OfInputImage.FileID.Value)
	assert.Equal(t, responses.ResponseInputImageContentDetailHigh, items[0].OfInputImage.Detail)
}

func TestToolCallOutputItemFileForwardsFileIDAndFilename(t *testing.T) {
	call := makeToolCall()
	out := agents.ToolOutputFileContent{FileID: "file_456", Filename: "report.pdf"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputFile)
	assert.True(t, items[0].OfInputFile.FileID.Valid())
	assert.Equal(t, "file_456", items[0].OfInputFile.FileID.Value)
	assert.True(t, items[0].OfInputFile.Filename.Valid())
	assert.Equal(t, "report.pdf", items[0].OfInputFile.Filename.Value)
}

func TestToolCallOutputItemFileForwardsFileURL(t *testing.T) {
	call := makeToolCall()
	out := agents.ToolOutputFileContent{FileURL: "https://example.com/report.pdf"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputFile)
	assert.True(t, items[0].OfInputFile.FileURL.Valid())
	assert.Equal(t, "https://example.com/report.pdf", items[0].OfInputFile.FileURL.Value)
}

func TestToolCallOutputItemTextDictVariant(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "text", "text": "hey"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputText)
	assert.Equal(t, "hey", items[0].OfInputText.Text)
}

func TestToolCallOutputItemImageDictVariant(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "image", "image_url": "http://example.com/img.png", "detail": "auto"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputImage)
	assert.True(t, items[0].OfInputImage.ImageURL.Valid())
	assert.Equal(t, "http://example.com/img.png", items[0].OfInputImage.ImageURL.Value)
	assert.Equal(t, responses.ResponseInputImageContentDetailAuto, items[0].OfInputImage.Detail)
}

func TestToolCallOutputItemImageDictVariantWithFileID(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "image", "file_id": "file_123"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputImage)
	assert.True(t, items[0].OfInputImage.FileID.Valid())
	assert.Equal(t, "file_123", items[0].OfInputImage.FileID.Value)
}

func TestToolCallOutputItemFileDictVariantWithFileData(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "file", "file_data": "foobar", "filename": "report.pdf"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputFile)
	assert.True(t, items[0].OfInputFile.FileData.Valid())
	assert.Equal(t, "foobar", items[0].OfInputFile.FileData.Value)
	assert.True(t, items[0].OfInputFile.Filename.Valid())
	assert.Equal(t, "report.pdf", items[0].OfInputFile.Filename.Value)
}

func TestToolCallOutputItemFileDictVariantWithFileURL(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "file", "file_url": "https://example.com/report.pdf", "filename": "report.pdf"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputFile)
	assert.True(t, items[0].OfInputFile.FileURL.Valid())
	assert.Equal(t, "https://example.com/report.pdf", items[0].OfInputFile.FileURL.Value)
	assert.True(t, items[0].OfInputFile.Filename.Valid())
	assert.Equal(t, "report.pdf", items[0].OfInputFile.Filename.Value)
}

func TestToolCallOutputItemFileDictVariantWithFileID(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "file", "file_id": "file_123", "filename": "report.pdf"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputFile)
	assert.True(t, items[0].OfInputFile.FileID.Valid())
	assert.Equal(t, "file_123", items[0].OfInputFile.FileID.Value)
	assert.True(t, items[0].OfInputFile.Filename.Valid())
	assert.Equal(t, "report.pdf", items[0].OfInputFile.Filename.Value)
}

func TestToolCallOutputItemImageWithExtraFields(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "image", "image_url": "http://example.com/img.png", "foobar": 213}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 1)

	decoded := decodeOutputItem(t, items[0])
	assert.Equal(t, "input_image", decoded["type"])
	assert.Equal(t, "http://example.com/img.png", decoded["image_url"])
	_, ok := decoded["foobar"]
	assert.False(t, ok)
}

func TestToolCallOutputItemMixedListWithValidDicts(t *testing.T) {
	call := makeToolCall()
	out := []any{
		map[string]any{"type": "text", "text": "hello"},
		map[string]any{"type": "image", "image_url": "http://example.com/img.png"},
		map[string]any{"type": "file", "file_id": "file_123"},
	}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	items := requireOutputList(t, payload)
	require.Len(t, items, 3)

	require.NotNil(t, items[0].OfInputText)
	assert.Equal(t, "hello", items[0].OfInputText.Text)
	require.NotNil(t, items[1].OfInputImage)
	assert.Equal(t, "http://example.com/img.png", items[1].OfInputImage.ImageURL.Value)
	require.NotNil(t, items[2].OfInputFile)
	assert.Equal(t, "file_123", items[2].OfInputFile.FileID.Value)
}

func TestToolCallOutputItemTextTypeOnlyNotConverted(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "text"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemImageTypeOnlyNotConverted(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "image"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemFileTypeOnlyNotConverted(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "file"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemEmptyDictNotConverted(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemDictWithoutTypeNotConverted(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"msg": "1234"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemImageDictVariantWithLocationNotConverted(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "image", "location": "/path/to/img.png"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemFileDictVariantWithPathNotConverted(t *testing.T) {
	call := makeToolCall()
	out := map[string]any{"type": "file", "path": "/path/to/file.txt"}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemListWithoutTypeNotConverted(t *testing.T) {
	call := makeToolCall()
	out := []any{map[string]any{"msg": "foobar"}}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func TestToolCallOutputItemMixedListPartialInvalidNotConverted(t *testing.T) {
	call := makeToolCall()
	out := []any{
		map[string]any{"type": "text", "text": "hello"},
		map[string]any{"msg": "foobar"},
	}
	payload := agents.ItemHelpers().ToolCallOutputItem(call, out)

	assert.Equal(t, mustJSON(t, out), requireOutputString(t, payload))
}

func makeToolCall() agents.ResponseFunctionToolCall {
	return agents.ResponseFunctionToolCall{
		ID:        "call-1",
		Arguments: "{}",
		CallID:    "call-1",
		Name:      "dummy",
		Type:      constant.ValueOf[constant.FunctionCall](),
	}
}

func requireOutputList(t *testing.T, payload responses.ResponseInputItemFunctionCallOutputParam) responses.ResponseFunctionCallOutputItemListParam {
	t.Helper()
	require.False(t, param.IsOmitted(payload.Output.OfResponseFunctionCallOutputItemArray))
	return payload.Output.OfResponseFunctionCallOutputItemArray
}

func requireOutputString(t *testing.T, payload responses.ResponseInputItemFunctionCallOutputParam) string {
	t.Helper()
	require.True(t, payload.Output.OfString.Valid())
	return payload.Output.OfString.Value
}

func decodeOutputItem(t *testing.T, item responses.ResponseFunctionCallOutputItemUnionParam) map[string]any {
	t.Helper()
	raw, err := json.Marshal(item)
	require.NoError(t, err)
	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))
	return decoded
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	require.NoError(t, err)
	return string(raw)
}
