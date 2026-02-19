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
	"testing"

	"github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
)

type recordingEditor struct {
	operations []agents.ApplyPatchOperation
}

func (r *recordingEditor) CreateFile(operation agents.ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return agents.ApplyPatchResult{Output: "Created " + operation.Path, Status: agents.ApplyPatchResultStatusCompleted}, nil
}

func (r *recordingEditor) UpdateFile(operation agents.ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return agents.ApplyPatchResult{Output: "Updated " + operation.Path, Status: agents.ApplyPatchResultStatusCompleted}, nil
}

func (r *recordingEditor) DeleteFile(operation agents.ApplyPatchOperation) (any, error) {
	r.operations = append(r.operations, operation)
	return agents.ApplyPatchResult{Output: "Deleted " + operation.Path, Status: agents.ApplyPatchResultStatusCompleted}, nil
}

func TestRecordingEditorRecordsOperations(t *testing.T) {
	editor := &recordingEditor{}
	operation := agents.ApplyPatchOperation{Path: "file.txt"}

	_, _ = editor.CreateFile(operation)
	_, _ = editor.UpdateFile(operation)
	_, _ = editor.DeleteFile(operation)

	assert.Equal(t, []agents.ApplyPatchOperation{operation, operation, operation}, editor.operations)
}
